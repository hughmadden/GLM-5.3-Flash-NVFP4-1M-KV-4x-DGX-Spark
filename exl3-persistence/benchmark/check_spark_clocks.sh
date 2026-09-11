#!/usr/bin/env bash
# =============================================================================
# check_spark_clocks.sh — pre-benchmark SM-clock / power-latch gate, GB10 Sparks
# =============================================================================
#
# WHAT THIS DETECTS
# -----------------
# The recorded "stuck clock" failure on this fleet: after an unclean power event
# (reboot, crash, OOM, unplug), a DGX Spark (GB10) node comes back with the
# graphics/SM clock collapsed to ~500-860 MHz *while the GPU is busy*, staying in
# P0, with `clocks_event_reasons.active == 0x0` ("nothing is limiting the
# clocks"). It still serves traffic and still passes /health. The only symptom is
# silently depressed throughput; a benchmark run on such a node under-reports by
# roughly 3-4x and nothing in the harness notices.
#
# On this fleet the fault band was recorded as 611-728 MHz (2026-09-02). The
# community band is 481-858 MHz; the two overlap, so this is the same failure
# mode. The underlying cause established by the community is a stuck *platform
# power ceiling* (USB-PD / PSU safety latch) — the clock is low because the power
# the board is allowed to draw is low. That is why this gate checks power and the
# power-cap governor alongside the clock, and why "reduce the clock with
# nvidia-smi" is not a fix.
#
# WHY THE CHECK MUST BE DONE UNDER LOAD — and why power alone is NOT enough
# -------------------------------------------------------------------------
# The fault is defined as "low clock WHILE BUSY". Utilisation is therefore the
# load gate, and the clock is only judged on samples taken while busy.
#
# Power draw is the corroborating signal, but it is ONLY meaningful under load:
# measured on this fleet 2026-09-11, an *idle* GB10 draws 13.8-17.2 W, and a
# degraded (latched) node draws ~5-16 W under load. Those bands overlap. A
# power-only test — e.g. the community rule "util >= 80% and power <= 25 W" —
# would therefore fire on an idle node if the utilisation gate were dropped.
# Utilisation is what separates them. Do not simplify this gate to a power check.
#
# Interesting fleet-specific detail: on this driver (580.173.02) the GB10 here
# does NOT down-clock at idle — idle reads 2405-2411 MHz, essentially the same as
# busy. So the clock value alone discriminates on this fleet today. The gate
# still requires load, because (a) that is the fault's definition, (b) a future
# driver or power state may idle low, and (c) the power corroboration only means
# anything under load.
#
# WHAT IS CHECKED
#   1. Governor / policy state — is a clock *policy* stuck, rather than a value?
#      * clocks_event_reasons.active decoded from nvml.h bit values
#      * persistence_mode, pstate
#      * clocks.applications.graphics (a `nvidia-smi -lgc/-ac` cap would show
#        here or in `nvidia-smi -q -d CLOCK`; a mis-set cap *imitates* this fault)
#      * the `spbm` hwmon power caps, if the host exposes them (the community's
#        direct readout of the latched power ceiling). ABSENT on this fleet —
#        reported as such rather than silently skipped.
#   2. Value under load — min/median SM clock on samples where util >= LOAD_MIN_PCT.
#   3. Throttle state — HW slowdown / HW thermal / HW power brake are hard FAILs;
#      sw_power_cap / sw_thermal / app-clocks-locked are WARNs.
#
# VERDICTS / EXIT CODES
#   0  PASS          every node was observed busy with SM clocks above the floor
#   1  FAIL          a busy node is below the clock floor, or is in a hardware
#                    throttle state. DO NOT TRUST THE BENCH.
#   2  INCONCLUSIVE  a node was never observed busy, so it could not be tested.
#                    This is non-zero on purpose: a gate that passes an untested
#                    node is worse than no gate.
#   3  ERROR         SSH or parse failure for at least one node
#
# READ-ONLY GUARANTEE
# -------------------
# Issues ONLY `nvidia-smi --query-gpu=...` and reads a few /sys/class/hwmon files
# over SSH. It never writes to a node, never sets clocks, power limits or
# persistence mode, never restarts or reconfigures anything, and never touches a
# container or an inference endpoint. Safe during a live benchmark.
# Do NOT add -lgc / -ac / -pl / --gpu-reset to this script: a gate must not
# change the thing it measures.
#
# THE SYNTHETIC-LOAD OPTION (documented, deliberately not automated)
# -----------------------------------------------------------------
# If no node is under load, the honest answer is INCONCLUSIVE. The right fix is
# to run the gate while a benchmark is active (the ideal condition). A synthetic
# load that does not touch the model endpoint would be a saturating FP32 GEMM,
# e.g. a ~10 s loop of large cuBLAS SGEMM via torch on each node — but that
# consumes GPU and UMA memory and is NOT safe to start unsolicited on a node
# running a live benchmark, so this script does not do it. See
# SPARK-CLOCK-GATE-20260911.md §4 for the exact snippet.
#
# USAGE
#   probes/check_spark_clocks.sh                      # gate all four nodes
#   probes/check_spark_clocks.sh --nodes node0,node2
#   SPARK_NODES="node0 node1" probes/check_spark_clocks.sh
#   probes/check_spark_clocks.sh --selftest           # prove the verdict logic
#   probes/check_spark_clocks.sh --verbose            # per-sample raw lines
#
# TUNING (environment variables)
#   SPARK_NODES       node list, space or comma separated (default node0..4)
#   CLOCK_SAMPLES     samples per node (default 15)
#   CLOCK_INTERVAL    seconds between samples, fractions ok (default 1.5)
#   LOAD_MIN_PCT      utilisation % required to count a sample as busy (50)
#   SM_FLOOR_MHZ      absolute SM-clock floor under load (default 1500)
#   SM_RATIO_MIN_PCT  SM clock as % of clocks.max.sm under load (default 50)
#   SPARK_SSH_OPTS    extra ssh options
#
# THRESHOLD RATIONALE
#   Fault band observed here: 611-728 MHz. Community band: 481-858 MHz.
#   Healthy on this fleet: 2385-2509 MHz (2026-09-11) and 2171-2190 MHz (prior
#   campaign). A 1500 MHz floor sits ~2x above every fault value ever reported
#   and ~30% below the lowest healthy value recorded here, so it separates them
#   with wide margin and does not depend on which engine is running. The ratio
#   floor guards a future driver reporting a different clocks.max.sm.
#
# Author: campaign agent, 2026-09-11 AEST. See SPARK-CLOCK-GATE-20260911.md.
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------- configuration
SPARK_NODES_RAW="${SPARK_NODES:-node0 node1 node2 node3}"
CLOCK_SAMPLES="${CLOCK_SAMPLES:-15}"
CLOCK_INTERVAL="${CLOCK_INTERVAL:-1.5}"
LOAD_MIN_PCT="${LOAD_MIN_PCT:-50}"
SM_FLOOR_MHZ="${SM_FLOOR_MHZ:-1500}"
SM_RATIO_MIN_PCT="${SM_RATIO_MIN_PCT:-50}"
PWR_LOW_W="${PWR_LOW_W:-30}"
SPARK_SSH_OPTS="${SPARK_SSH_OPTS:-}"

SSH_BASE_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)
# shellcheck disable=SC2206
[ -n "$SPARK_SSH_OPTS" ] && SSH_BASE_OPTS+=($SPARK_SSH_OPTS)

FIELDS="clocks.current.sm,clocks.current.graphics,clocks.max.sm,clocks.applications.graphics,utilization.gpu,power.draw.average,temperature.gpu,pstate,persistence_mode,clocks_event_reasons.active"

VERBOSE=0
SELFTEST=0
NODES=()

# ------------------------------------------------------------------- arg parsing
while [ $# -gt 0 ]; do
  case "$1" in
    --nodes)      IFS=', ' read -r -a NODES <<< "${2:-}"; shift 2 ;;
    --nodes=*)    IFS=', ' read -r -a NODES <<< "${1#*=}"; shift ;;
    --samples)    CLOCK_SAMPLES="${2:-}"; shift 2 ;;
    --interval)   CLOCK_INTERVAL="${2:-}"; shift 2 ;;
    --floor)      SM_FLOOR_MHZ="${2:-}"; shift 2 ;;
    --load-min)   LOAD_MIN_PCT="${2:-}"; shift 2 ;;
    --verbose|-v) VERBOSE=1; shift ;;
    --selftest)   SELFTEST=1; shift ;;
    -h|--help)    sed -n '2,120p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 3 ;;
  esac
done

if [ "${#NODES[@]}" -eq 0 ]; then
  IFS=', ' read -r -a NODES <<< "$SPARK_NODES_RAW"
fi

# ------------------------------------------------------------------- utilities
hex_to_dec() {
  local h="${1,,}"; h="${h#0x}"; h="${h// /}"
  [ -z "$h" ] && { echo 0; return; }
  echo $(( 16#$h )) 2>/dev/null || echo 0
}

# Decode an nvmlClocksEventReason bitmask. Bit values taken verbatim from
# /usr/local/cuda/include/nvml.h on node0 (driver 580.173.02):
#   nvmlClocksEventReasonGpuIdle                   0x001
#   nvmlClocksEventReasonApplicationsClocksSetting 0x002
#   nvmlClocksEventReasonSwPowerCap                0x004
#   nvmlClocksThrottleReasonHwSlowdown             0x008
#   nvmlClocksEventReasonSyncBoost                 0x010
#   nvmlClocksEventReasonSwThermalSlowdown         0x020
#   nvmlClocksThrottleReasonHwThermalSlowdown      0x040
#   nvmlClocksThrottleReasonHwPowerBrakeSlowdown   0x080
#   nvmlClocksEventReasonDisplayClockSetting       0x100
# There is deliberately no "platform power / PD" bit in NVML: the fault this
# script exists to catch is invisible to the throttle reason mask.
decode_reasons() {
  local v; v=$(hex_to_dec "$1")
  local -a out=()
  (( v & 0x001 )) && out+=(gpu_idle)
  (( v & 0x002 )) && out+=(app_clocks_locked)
  (( v & 0x004 )) && out+=(sw_power_cap)
  (( v & 0x008 )) && out+=(HW_SLOWDOWN)
  (( v & 0x010 )) && out+=(sync_boost)
  (( v & 0x020 )) && out+=(sw_thermal)
  (( v & 0x040 )) && out+=(HW_THERMAL)
  (( v & 0x080 )) && out+=(HW_POWER_BRAKE)
  (( v & 0x100 )) && out+=(display_clock)
  if [ "${#out[@]}" -eq 0 ]; then echo "none"; else (IFS=','; echo "${out[*]}"); fi
}

median_of() {
  local n=$#; [ "$n" -eq 0 ] && { echo "N/A"; return; }
  local sorted; sorted=$(printf '%s\n' "$@" | sort -n)
  printf '%s\n' "$sorted" | sed -n "$(( (n + 1) / 2 ))p"
}

# --------------------------------------------------------- the verdict function
# Reads the collector stream on stdin: first line "#SPBM <p11> <p13>" (or
# "#SPBM absent"), then nvidia-smi CSV lines (noheader,nounits). Sets V_* globals.
# This is the single decision point: main and --selftest both call it.
compute_verdict() {
  local node="$1"
  local -a sm=() gr=() maxsm=() appgr=() util=() pwr=() temp=() reasons=()
  local pstate="N/A" persist="N/A" spbm="absent"
  local n=0
  local -a busy_sm=() busy_pwr=()
  local -a raw_lines=()

  local line
  while IFS= read -r line; do
    [ -z "${line// /}" ] && continue
    case "$line" in
      '#SPBM'*) spbm="${line#\#SPBM }"; continue ;;
      '#'*) continue ;;
    esac
    IFS=',' read -r f_sm f_gr f_maxsm f_appgr f_util f_pwr f_temp f_pstate f_persist f_reasons _rest <<< "$line"
    f_sm="${f_sm// /}"; f_gr="${f_gr// /}"; f_maxsm="${f_maxsm// /}"; f_appgr="${f_appgr// /}"
    f_util="${f_util// /}"; f_pwr="${f_pwr// /}"; f_temp="${f_temp// /}"
    f_pstate="${f_pstate// /}"; f_persist="${f_persist// /}"; f_reasons="${f_reasons// /}"

    # malformed / failed query -> not a sample
    case "$f_sm" in ''|*[!0-9]*) continue ;; esac
    case "$f_util" in ''|*[!0-9]*) continue ;; esac

    n=$((n+1))
    raw_lines+=("$line")
    sm+=("$f_sm"); gr+=("$f_gr"); maxsm+=("$f_maxsm"); appgr+=("$f_appgr")
    util+=("$f_util"); pwr+=("$f_pwr"); temp+=("$f_temp"); reasons+=("$f_reasons")
    pstate="$f_pstate"; persist="$f_persist"

    if [ "$f_util" -ge "$LOAD_MIN_PCT" ] 2>/dev/null; then
      busy_sm+=("$f_sm")
      case "$f_pwr" in ''|*[!0-9.]*) busy_pwr+=("");; *) busy_pwr+=("$f_pwr");; esac
    fi
  done

  V_NODE="$node"; V_N="$n"; V_VERDICT="ERROR"; V_NOTE=""
  V_RAW=("${raw_lines[@]}"); V_SPBM="$spbm"
  if [ "$n" -eq 0 ]; then V_NOTE="no parseable nvidia-smi samples"; return; fi

  # aggregates over ALL samples
  V_UTIL_MAX=0; V_TEMP_MAX=0; V_PWR_SUM=0; V_PWR_N=0
  for i in "${!sm[@]}"; do
    [ "${util[$i]}" -gt "$V_UTIL_MAX" ] 2>/dev/null && V_UTIL_MAX="${util[$i]}"
    [ "${temp[$i]}" -gt "$V_TEMP_MAX" ] 2>/dev/null && V_TEMP_MAX="${temp[$i]}"
    case "${pwr[$i]}" in ''|*[!0-9.]*) ;; *) V_PWR_SUM=$(awk -v a="$V_PWR_SUM" -v b="${pwr[$i]}" 'BEGIN{print a+b}'); V_PWR_N=$((V_PWR_N+1));; esac
  done
  V_PWR_AVG="N/A"
  [ "$V_PWR_N" -gt 0 ] && V_PWR_AVG=$(awk -v s="$V_PWR_SUM" -v n="$V_PWR_N" 'BEGIN{printf "%.1f", s/n}')
  V_MAXSM="${maxsm[0]}"; V_APPSM="${appgr[0]}"; V_GR="${gr[0]}"
  V_PSTATE="$pstate"; V_PERSIST="$persist"
  V_SM_MIN=$(printf '%s\n' "${sm[@]}" | sort -n | head -1)
  V_SM_MAX=$(printf '%s\n' "${sm[@]}" | sort -n | tail -1)
  V_SM_MED=$(median_of "${sm[@]}")
  V_REASONS=$(printf '%s\n' "${reasons[@]}" | sort -u | tr '\n' ' ' | sed 's/ *$//')
  V_NBUSY="${#busy_sm[@]}"

  if [ "$V_NBUSY" -eq 0 ]; then
    V_VERDICT="IDLE"
    V_NOTE="never busy (>=$LOAD_MIN_PCT% util) in $n samples; clock under load not observed -> cannot judge"
    V_BUSY_MIN="N/A"; V_RATIO="N/A"; V_BUSY_PWR="N/A"
    return
  fi

  V_BUSY_MIN=$(printf '%s\n' "${busy_sm[@]}" | sort -n | head -1)
  V_BUSY_MED=$(median_of "${busy_sm[@]}")
  local bp=()
  for p in "${busy_pwr[@]}"; do [ -n "$p" ] && bp+=("$p"); done
  if [ "${#bp[@]}" -gt 0 ]; then
    V_BUSY_PWR=$(awk -v s="$(printf '%s\n' "${bp[@]}" | awk '{t+=$1} END{print t}')" -v n="${#bp[@]}" 'BEGIN{printf "%.1f", s/n}')
  else
    V_BUSY_PWR="N/A"
  fi
  if [ "${V_MAXSM:-0}" -gt 0 ] 2>/dev/null; then
    V_RATIO=$(awk -v a="$V_BUSY_MIN" -v b="$V_MAXSM" 'BEGIN{printf "%.0f", 100*a/b}')
  else
    V_RATIO=0
  fi

  # ---- FAIL 1: hardware throttle states
  local i v hw_bad=0
  for i in "${!reasons[@]}"; do
    v=$(hex_to_dec "${reasons[$i]}")
    (( v & 0x008 )) && hw_bad=1
    (( v & 0x040 )) && hw_bad=1
    (( v & 0x080 )) && hw_bad=1
  done
  if [ "$hw_bad" -eq 1 ]; then
    V_VERDICT="FAIL"; V_VERDICT_KIND="hw-throttle"
    V_NOTE="hardware throttle active while busy: $V_REASONS"
    return
  fi

  # ---- FAIL 2: busy but clock below the floor -> the stuck-clock fault
  if [ "$V_BUSY_MIN" -lt "$SM_FLOOR_MHZ" ] 2>/dev/null || [ "$V_RATIO" -lt "$SM_RATIO_MIN_PCT" ] 2>/dev/null; then
    V_VERDICT="FAIL"
    local lowpwr=0
    if [ "$V_BUSY_PWR" != "N/A" ]; then
      lowpwr=$(awk -v p="$V_BUSY_PWR" -v lim="$PWR_LOW_W" 'BEGIN{print (p<=lim)?1:0}')
    fi
    if [ "$V_BUSY_MIN" -le 1000 ] 2>/dev/null && [ "$lowpwr" -eq 1 ]; then
      V_VERDICT_KIND="power-latch"
      V_NOTE="STUCK CLOCK: busy at ${V_BUSY_MIN} MHz (${V_RATIO}% of max) with only ${V_BUSY_PWR} W draw and governor '$V_REASONS' -> platform power-delivery latch signature. Clock is low because the allowed power is low, not because of a throttle. Fix is an AC power drain, not nvidia-smi."
    else
      V_VERDICT_KIND="low-clock"
      V_NOTE="LOW CLOCK UNDER LOAD: busy at ${V_BUSY_MIN} MHz (${V_RATIO}% of max), ${V_BUSY_PWR} W, governor '$V_REASONS'. Check for a mis-set clock cap (nvidia-smi -q -d CLOCK; nvidia-smi -rgc) before assuming the power latch."
    fi
    return
  fi

  # ---- WARN: policy / soft-throttle states that do not themselves invalidate
  local soft=""
  for i in "${!reasons[@]}"; do
    v=$(hex_to_dec "${reasons[$i]}")
    (( v & 0x002 )) && soft="${soft}app_clocks_locked "
    (( v & 0x004 )) && soft="${soft}sw_power_cap "
    (( v & 0x020 )) && soft="${soft}sw_thermal "
  done
  case "$V_REASONS" in *gpu_idle*) soft="${soft}idle_samples_present " ;; esac
  if [ -n "$soft" ]; then
    V_VERDICT="WARN"
    V_NOTE="clocks healthy but flags present: $(echo "$soft" | sed 's/ *$//' | tr ' ' ',')"
    return
  fi

  V_VERDICT="PASS"
  V_NOTE="busy at ${V_BUSY_MIN}-${V_SM_MAX} MHz (${V_RATIO}% of max), ${V_BUSY_PWR} W, no throttle"
}

# --------------------------------------------------------------------- selftest
run_selftest() {
  local rc=0
  echo "=== selftest: verdict logic against synthetic fixtures ==="
  echo "The real compute_verdict() is driven with fake collector streams."
  echo "The fault cannot be injected on a live node without breaking the"
  echo "read-only rule, so this is how the gate's FAIL path is proven."
  echo
  _st() {
    local name="$1" expect="$2" csv="$3"
    compute_verdict "fixture" <<< "$csv"
    if [ "$V_VERDICT" = "$expect" ]; then
      printf '  ok   %-52s -> %s\n' "$name" "$V_VERDICT"
    else
      printf '  FAIL %-52s -> got %s, expected %s\n' "$name" "$V_VERDICT" "$expect"
      rc=1
    fi
  }
  # fields: sm,gr,maxsm,appgr,util,pwr,temp,pstate,persist,reasons
  local healthy_busy="#SPBM absent
2418, 2418, 3003, 2418, 96, 62.0, 81, P0, Enabled, 0x0000000000000000
2398, 2398, 3003, 2418, 96, 58.0, 82, P0, Enabled, 0x0000000000000000"
  local fleet_fault="#SPBM absent
611, 611, 3003, 2418, 96, 14.0, 80, P0, Enabled, 0x0000000000000000
728, 728, 3003, 2418, 95, 15.0, 80, P0, Enabled, 0x0000000000000000"
  local lowclock_highpwr="#SPBM absent
721, 721, 3003, 2418, 96, 70.0, 55, P0, Enabled, 0x0000000000000000"
  local idle_lowclock="#SPBM absent
210, 210, 3003, 2418, 0, 8.0, 45, P8, Enabled, 0x0000000000000001
180, 180, 3003, 2418, 0, 7.0, 44, P8, Enabled, 0x0000000000000001"
  local hw_thermal="#SPBM absent
2400, 2400, 3003, 2418, 96, 60.0, 95, P0, Enabled, 0x0000000000000040"
  local idle_only_healthy_clock="#SPBM absent
2405, 2405, 3003, 2418, 0, 15.0, 45, P8, Enabled, 0x0000000000000001
2405, 2405, 3003, 2418, 0, 15.0, 45, P8, Enabled, 0x0000000000000001"
  local mixed_busy_and_idle="#SPBM absent
2398, 2398, 3003, 2418, 96, 60.0, 81, P0, Enabled, 0x0000000000000000
2405, 2405, 3003, 2418, 0, 15.0, 45, P8, Enabled, 0x0000000000000001"

  _st "healthy busy -> PASS"                              "PASS" "$healthy_busy"
  _st "fleet fault band 611-728 MHz + 15 W -> FAIL"        "FAIL" "$fleet_fault"
  _st "721 MHz but 70 W (not the power latch) -> FAIL"     "FAIL" "$lowclock_highpwr"
  _st "idle low clock 210 MHz -> IDLE (not tested)"        "IDLE" "$idle_lowclock"
  _st "HW thermal slowdown under load -> FAIL"             "FAIL" "$hw_thermal"
  _st "idle at healthy 2405 MHz only -> IDLE (not tested)" "IDLE" "$idle_only_healthy_clock"
  _st "busy + idle mix -> PASS on the busy samples"        "PASS" "$mixed_busy_and_idle"
  echo
  if [ "$rc" -eq 0 ]; then echo "selftest: ALL PASS"; else echo "selftest: FAILURES PRESENT"; fi
  return "$rc"
}

if [ "$SELFTEST" -eq 1 ]; then run_selftest; exit $?; fi

# ---------------------------------------------------------------------- driver
TMPDIR_RUN=$(mktemp -d)
trap 'rm -rf "$TMPDIR_RUN"' EXIT

echo "Spark GB10 clock / power-latch gate — $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "nodes: ${NODES[*]} | ${CLOCK_SAMPLES} samples/node every ${CLOCK_INTERVAL}s | load>=${LOAD_MIN_PCT}% | SM floor ${SM_FLOOR_MHZ} MHz or ${SM_RATIO_MIN_PCT}% of max"
echo "mode: READ-ONLY (nvidia-smi --query-gpu + /sys/class/hwmon reads). Safe during a live benchmark."
echo

REMOTE_CMD="H=''; for f in /sys/class/hwmon/hwmon*/; do [ \"\$(cat \$f/name 2>/dev/null)\" = spbm ] && H=\"\$f\"; done; \
if [ -n \"\$H\" ]; then echo \"#SPBM \$(cat \$H/power11_cap 2>/dev/null) \$(cat \$H/power13_cap 2>/dev/null)\"; else echo '#SPBM absent'; fi; \
for i in \$(seq 1 $CLOCK_SAMPLES); do \
nvidia-smi --query-gpu=$FIELDS --format=csv,noheader,nounits 2>&1 || echo QUERY_FAILED; \
sleep $CLOCK_INTERVAL; done"

for node in "${NODES[@]}"; do
  (
    timeout "$(awk -v s="$CLOCK_SAMPLES" -v i="$CLOCK_INTERVAL" 'BEGIN{printf "%d", s*i+40}')" \
      ssh "${SSH_BASE_OPTS[@]}" "$node" "$REMOTE_CMD" \
      > "$TMPDIR_RUN/$node.csv" 2> "$TMPDIR_RUN/$node.err"
    echo $? > "$TMPDIR_RUN/$node.rc"
  ) &
done
wait

# ------------------------------------------------------------------- report
overall=0
printf '%-8s %-6s %-8s %-8s %-8s %-7s %-7s %-6s %-7s %-9s %s\n' \
  NODE VERDICT SM_MIN SM_MED SM_MAX BUSY UTILMAX TEMP PWRBUSY APPSM GOVERNOR
printf '%-8s %-6s %-8s %-8s %-8s %-7s %-7s %-6s %-7s %-9s %s\n' \
  -------- ------ -------- -------- -------- ------- ------- ------ ------- --------- --------

for node in "${NODES[@]}"; do
  rc=$(cat "$TMPDIR_RUN/$node.rc" 2>/dev/null || echo 3)
  if [ "$rc" != "0" ] || [ ! -s "$TMPDIR_RUN/$node.csv" ]; then
    printf '%-8s %-6s %s\n' "$node" "ERROR" "ssh/query failed (rc=$rc): $(tr '\n' ' ' < "$TMPDIR_RUN/$node.err" 2>/dev/null | head -c 120)"
    overall=3; continue
  fi
  if grep -q QUERY_FAILED "$TMPDIR_RUN/$node.csv" 2>/dev/null; then
    printf '%-8s %-6s %s\n' "$node" "ERROR" "nvidia-smi query failed on node"
    overall=3; continue
  fi

  compute_verdict "$node" < "$TMPDIR_RUN/$node.csv"

  printf '%-8s %-6s %-8s %-8s %-8s %-7s %-7s %-6s %-7s %-9s %s\n' \
    "$V_NODE" "$V_VERDICT" "${V_SM_MIN:-?}" "${V_SM_MED:-?}" "${V_SM_MAX:-?}" \
    "${V_NBUSY:-0}/${V_N:-0}" "${V_UTIL_MAX:-?}" "${V_TEMP_MAX:-?}" \
    "${V_BUSY_PWR:-?}" "${V_APPSM:-?}" "$(decode_reasons "${V_REASONS%% *}")"
  [ -n "$V_NOTE" ] && printf '%-8s   └─ %s\n' "" "$V_NOTE"
  printf '%-8s   └─ pstate=%s persistence=%s gr=%s max_sm=%s spbm_caps=%s reasons=%s\n' \
    "" "${V_PSTATE:-?}" "${V_PERSIST:-?}" "${V_GR:-?}" "${V_MAXSM:-?}" "${V_SPBM:-?}" "${V_REASONS:-?}"

  if [ "$VERBOSE" -eq 1 ]; then
    i=0
    while IFS= read -r l; do
      case "$l" in '#'*) continue ;; esac
      i=$((i+1)); printf '%-8s      sample %d: %s\n' "" "$i" "$l"
    done < "$TMPDIR_RUN/$node.csv"
  fi

  case "$V_VERDICT" in
    PASS|WARN) [ "$overall" -eq 0 ] && overall=0 ;;
    FAIL)      [ "$overall" -lt 1 ] && overall=1 ;;
    IDLE)      [ "$overall" -lt 2 ] && overall=2 ;;
    *)         overall=3 ;;
  esac
done

echo
case "$overall" in
  0) echo "GATE: PASS — every node was observed busy with SM clocks above the floor." ;;
  1) echo "GATE: FAIL — a busy node is below the clock floor, or is in HW throttle. DO NOT TRUST THIS BENCH." ;;
  2) echo "GATE: INCONCLUSIVE — a node was never observed busy, so it was not tested. Re-run while load is active, or run a synthetic GPU load first (see the findings note; not safe to start during someone else's benchmark)." ;;
  3) echo "GATE: ERROR — could not complete the check on at least one node." ;;
esac

exit "$overall"
