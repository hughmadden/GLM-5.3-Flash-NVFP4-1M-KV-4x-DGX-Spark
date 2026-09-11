#!/usr/bin/env bash
# run_lil_c1c6.sh -- run the Local Inference Lab standard C1..C6 decode bench
# against a live vLLM endpoint and capture the verbatim output, the
# persistence/offload metric deltas that frame it, and a quantified record of
# any ambient (non-bench) traffic that shared the engine during the run.
#
# Usage: ./run_lil_c1c6.sh <engine-url> <model-id> <outdir> [rounds] [max-tokens]
#
# Bench rules enforced before measuring:
#   * endpoint must answer /health with 200
#   * num_requests_running == 0 and num_requests_waiting_by_reason{capacity} == 0
#     (a request parked in reason="deferred" holds no GPU slot and is not a
#      blocker; that parked state is a separate defect, see the report)
#   * no leftover bench containers  (checked by the caller)
#   * the persistence flusher must be stopped (caller's job)
#
# Ambient traffic: this deployment is fronted by a model gateway, so the engine
# can receive traffic that is not ours. Set AMBIENT_LOG to a shell command that
# prints gateway log lines; the runner snapshots it before and after the bench
# and keeps only the lines that appeared during the run.
set -u
URL=${1:?engine url, e.g. http://HEAD:8000}
MODEL=${2:?served model id}
OUT=${3:?output dir}
ROUNDS=${4:-3}
MAXTOK=${5:-700}
AMBIENT_LOG=${AMBIENT_LOG:-}
mkdir -p "$OUT"

say() { printf '%s %s\n' "$(TZ=Australia/Sydney date '+%Y-%m-%dT%H:%M:%S%z')" "$*"; }
scrape() { curl -s --max-time 20 "$URL/metrics" > "$1"; }

metric() {  # metric <awk-regex-prefix> -> summed value across label sets
  curl -s --max-time 10 "$URL/metrics" \
    | awk -v k="^$1" '$0 ~ k {s+=$NF} END{printf "%d", s+0}'
}

idle_wait() {  # running must be 0; capacity-waiting must be 0; deferred ignored
  local r w i
  for i in $(seq 1 40); do
    r=$(metric 'vllm:num_requests_running[{]')
    w=$(metric 'vllm:num_requests_waiting_by_reason[{].*reason="capacity"')
    if [ "${r:-1}" = "0" ] && [ "${w:-1}" = "0" ]; then
      echo "idle: running=$r capacity_waiting=$w"; return 0
    fi
    sleep 5
  done
  echo "NOT IDLE after 200s (running=$r capacity_waiting=$w) -- proceeding, recorded as a caveat"
}

say "run_lil_c1c6 start url=$URL model=$MODEL rounds=$ROUNDS max_tokens=$MAXTOK out=$OUT"

code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$URL/health")
say "health=$code"
[ "$code" = "200" ] || { say "ABORT: endpoint not healthy"; exit 2; }

say "waiting for idle"; idle_wait
scrape "$OUT/metrics-before.txt"

# --- ambient traffic bookkeeping -------------------------------------------
if [ -n "$AMBIENT_LOG" ]; then
  eval "$AMBIENT_LOG" 2>/dev/null | wc -l > "$OUT/ambient-lines-before.txt" || echo 0 > "$OUT/ambient-lines-before.txt"
  say "ambient log baseline: $(cat "$OUT/ambient-lines-before.txt") lines"
fi

# --- occupancy sampler ------------------------------------------------------
(
  echo "utc,running,waiting,waiting_capacity,waiting_deferred"
  while :; do
    ts=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    vals=$(curl -s --max-time 8 "$URL/metrics" | awk '
      /^vllm:num_requests_running\{/ {r+=$NF}
      /^vllm:num_requests_waiting\{/ {w+=$NF}
      /^vllm:num_requests_waiting_by_reason\{.*reason="capacity"/ {c+=$NF}
      /^vllm:num_requests_waiting_by_reason\{.*reason="deferred"/ {d+=$NF}
      END{printf "%d,%d,%d,%d", r+0, w+0, c+0, d+0}')
    echo "$ts,$vals"
    sleep 3
  done
) > "$OUT/occupancy.csv" 2>/dev/null &
SAMPLER=$!

# --- the bench itself -------------------------------------------------------
say "running LIL standard bench (verbatim output -> bench-c1c6.txt)"
python3 "$(dirname "$0")/bench_c1c6.py" \
  --url "$URL" --model "$MODEL" --rounds "$ROUNDS" --max-tokens "$MAXTOK" \
  > "$OUT/bench-c1c6.txt" 2>&1
rc=$?
say "bench exit=$rc"

kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null

sleep 5
scrape "$OUT/metrics-after.txt"

# --- ambient traffic during the run ----------------------------------------
if [ -n "$AMBIENT_LOG" ]; then
  eval "$AMBIENT_LOG" 2>/dev/null > "$OUT/ambient-lines-after.txt" || true
  before=$(cat "$OUT/ambient-lines-before.txt" 2>/dev/null || echo 0)
  total=$(wc -l < "$OUT/ambient-lines-after.txt" 2>/dev/null || echo 0)
  : > "$OUT/ambient-during.txt"
  if [ "${total:-0}" -gt "${before:-0}" ]; then
    n=$(( total - before ))
    tail -n "$n" "$OUT/ambient-lines-after.txt" > "$OUT/ambient-during.txt"
  fi
  say "ambient requests during bench: $(grep -c 'request_id=' "$OUT/ambient-during.txt" 2>/dev/null || echo 0)"
fi

# --- framing counters -------------------------------------------------------
for f in before after; do
  grep -E '^vllm:(prefix_cache_(queries|hits)_total|kv_offload_(store_bytes|load_bytes|store_time|lookup_sync_delay_seconds_(count|sum)|allocation_failure)_total|num_requests_(running|waiting))' \
    "$OUT/metrics-$f.txt" | grep -v transfer_type | sed "s/^/$f /" > "$OUT/offload-$f.txt"
done
say "offload counters -> offload-before.txt / offload-after.txt"

# --- occupancy summary ------------------------------------------------------
if [ -s "$OUT/occupancy.csv" ]; then
  awk -F, 'NR>1 {n++; if ($2+0>0) busy++; if ($2+0>1) multi++; s+=$2; if ($2+0>mx) mx=$2}
    END {printf "occupancy samples=%d running>0 in %d (%.0f%%), running>1 in %d (%.0f%%), max_running=%d, mean_running=%.2f\n",
      n, busy, (n?100*busy/n:0), multi, (n?100*multi/n:0), mx, (n?s/n:0)}' \
    "$OUT/occupancy.csv" | tee "$OUT/occupancy-summary.txt"
fi

say "run_lil_c1c6 done rc=$rc"
exit $rc
