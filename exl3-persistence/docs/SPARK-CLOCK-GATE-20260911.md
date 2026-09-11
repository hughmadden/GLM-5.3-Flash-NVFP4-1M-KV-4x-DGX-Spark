# SPARK CLOCK GATE: the "stuck clock" failure mode and a pre-bench gate (2026-09-11 AEST)

Investigation of the known GB10 "stuck clock" fault on `node0`–`node3`, plus the
detection gate the prior campaign said was missing. **All fleet access was read-only.**
No node was rebooted, restarted, reconfigured or disturbed; no container was touched;
no inference request was made against the model endpoint.

**Verdict in one line: no node shows the fault. Across three gate runs all four nodes were
busy at 2,320–2,489 MHz drawing 52.9–62.9 W — roughly 2.7–2.9× above the top of the fault
band. Two runs were clean; the third caught the fleet warmer and soft-throttling (SW
thermal / SW power cap) with clocks still healthy. The running benchmark is not at risk
from this cause within the evidence gathered here, with the caveats in §5.**

Deliverable: [`probes/check_spark_clocks.sh`](probes/check_spark_clocks.sh) — run it
before trusting a benchmark. Exit 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE, 3 = ERROR.

---

## 1. What the prior campaign recorded, and what it does not say

The only local record of the fault is prose in a session artefact — there is **no raw
`nvidia-smi` capture of the fault anywhere on this host**. Extracted verbatim from
`~/.dsh/sessions/--home-the operator-dev-recipes--/session-f10eda46-…/session.jsonl.zstd`:

> **Clocks.** Healthy GB10 decode sits near **2,180 MHz** at 95% utilization. **After a
> reboot, two nodes were pinned at 611 to 728 MHz under load.** All four verified at
> 2,171 to 2,190 MHz under a real request before this run, and again during the power
> sampling.

> **Part one, clocks.** Our first run had EXL3 winning every metric by wide margins.
> Two of the four Sparks had come out of a reboot with GPU clocks pinned between 611 and
> 728 MHz. The tell was NVFP4's numbers not matching its own recipe's published peak.
> **Restart, verify clocks under load, run everything again.**

Established from this record (EVIDENCE):

| Fact | Value |
|---|---|
| Fault band observed on **this fleet** | 611–728 MHz under load |
| Healthy band on this fleet, prior campaign | 2,171–2,190 MHz |
| Detection used at the time | eyeball a real request; no gate |
| Recovery used at the time | "restart", then re-verify |
| Which two nodes failed | **not recorded** |
| Whether "restart" meant warm reboot or AC drain | **not recorded** |
| Raw nvidia-smi output of the failing state | **not recorded — MISS** |

Note also that the prior "healthy" 2,171–2,190 MHz and this campaign's 2,385–2,489 MHz
are different engines at different points in their run, so they are not in conflict;
both are healthy. The gate floor (§3) is set below both.

---

## 2. What the failure mode actually is

### 2.1 Established from evidence on these hosts

| Question | Answer | Evidence |
|---|---|---|
| What is stuck — SM clock, memory clock, or a policy? | The **graphics/SM clock domain**, as a **value**. `clocks.current.graphics` and `clocks.current.sm` move together. | `nvidia-smi -q -d CLOCK` on all four: Graphics == SM (2,398/2,398, 2,450/2,450, 2,450/2,450, 2,385/2,385) |
| Is the memory clock involved? | **Cannot be determined on GB10 — MISS.** | `clocks.current.memory` returns `[N/A]`; `clocks.max.memory` `[N/A]` |
| Is a clock *policy* stuck (locked clocks / applications clocks / persistence)? | **No, on this fleet.** App clocks equal the *default* app clocks (2,418 MHz), the event-reason flag is Not Active, persistence is Enabled, and **no clock-cap systemd unit exists**. | §4.3, §4.4 |
| Is the fault visible in the throttle mask? | **No — it is silent.** All clock event reasons read "Not Active" while the clock is low. | §4.3; community reports agree |
| Is it a reboot-specific trigger? | **No — reboot is a recurring antecedent, not the cause.** On this fleet the event followed a **kernel + driver OTA upgrade plus reboot** on 2026-09-02. | §4.5 |
| Does anything *in the boot path* set clocks? | **No.** The only boot-path GPU agent is `nvidia-persistenced`, which enables persistence mode and nothing else. | §4.4 |

### 2.2 What is actually stuck (INFERENCE, well-supported by external evidence)

This is a **platform power-delivery latch, not a driver or governor fault**. The clock is
low *because the power the board is allowed to draw is low*. Labelled INFERENCE because I
could not read the power cap directly on this fleet (the `spbm` hwmon node is absent — §4.6).

Supporting external evidence (community reports, not this fleet):

- `clocks_throttle_reasons.active` reads `0x0` while the clock is stuck at 513–721 MHz —
  "the platform is limiting power delivery below the level where NVML would report a
  throttle reason" (parallelArchitect).
  [361296](https://forums.developer.nvidia.com/t/investigating-513mhz-cap-for-gpu/361296)
- The cap is readable *outside* NVML: "In the `spbm` hwmon node, pl1 was 20 W and syspl1
  30 W, against a 250/300 W hardware max. After a reboot: 140 W and 231 W, same burn
  98.6 TFLOPS. 10.5x." (vladtemian). [361296](https://forums.developer.nvidia.com/t/investigating-513mhz-cap-for-gpu/361296)
- Mechanism: "power control circuits in the PSU … limits amount of power/voltage supplied
  … stuck in some safety protocol."
  [376239](https://forums.developer.nvidia.com/t/gpu-clock-bug-looks-like-5-min-wait-is-enough/376239)
- `nvidia-smi -lgc 3003` "silently has no effect; clock stays 721 MHz", and
  `-q -d SUPPORTED_CLOCKS` reports N/A — i.e. clock commands are not the lever.
  [376039](https://forums.developer.nvidia.com/t/dgx-spark-gb10-gpu-clock-pinned-at-721-mhz-under-full-load-no-throttling-not-liftable-via-nvidia-smi/376039)

Community fault band: **481–858 MHz**, most commonly 507–721 MHz, at **~5–16 W** under
load (one report measured a board flat at 35 W while still capped, so treat the power band
as wider than the clock band). The fleet's own 611–728 MHz sits inside that band, so the
two are the same failure mode. (The community band is wider than the fleet's; the fleet's
"728 MHz" figure has no external source — that is this fleet's own observation, not a
literature value.)

### 2.3 Consequence for diagnosis: three separate things get conflated

| # | Thing | How it presents | How to tell it apart |
|---|---|---|---|
| 1 | **Platform power latch** (this fault) | Low clock **+ low power** under load, P0, mask `0x0`, `-lgc` no-op | Clock **and** power both collapsed; AC drain fixes it |
| 2 | **A mis-set clock cap** (`nvidia-smi -lgc/-ac`, e.g. the tonyd2wild hard-poweroff mitigation) | Low clock, P0, mask `0x0` — **looks identical** | Check `nvidia-smi -q -d CLOCK`; `nvidia-smi -rgc` reverts. **Not present on this fleet** (§4.4) |
| 3 | **A real throttle** (thermal/power/brake) | Low clock with a **non-zero** mask | Mask bit set; §3.3 decodes it |

The gate distinguishes all three, and labels which one it thinks it found.

---

## 3. Detection method

### 3.1 Why idle sampling is insufficient

An idle GPU legitimately reports a low SM clock, so **a single idle sample proves
nothing**: by clock value alone it is indistinguishable from the fault. The fault is
defined as *low clock while busy*. The gate therefore samples each node repeatedly,
requires the GPU to be demonstrably busy, and judges the clock **only on the samples
taken while busy**.

There is a second, sharper reason on this fleet, and it is the reason **power draw must
not be used as the primary test**:

| State | SM clock | Power draw | Util |
|---|---|---|---|
| Idle (measured 2026-09-11) | **2,405–2,411 MHz** | **13.8–17.2 W** | 0% |
| Loaded, healthy (measured) | 2,385–2,489 MHz | 52.9–62.9 W | 93–96% |
| Degraded, community | 481–858 MHz | ~5–16 W | 80–96% |

**Idle power (13.8–17.2 W) and degraded power (~5–16 W) overlap.** The community rule
"util ≥ 80% and power ≤ 25 W" is safe only *because* it also requires utilisation. A
power-only gate would fire on an idle node. Utilisation is what separates them.

A useful fleet-specific observation: on driver 580.173.02 these GB10s **do not down-clock
at idle** — idle reads ~2,405 MHz, essentially the same as busy. So today the clock value
alone discriminates here. The gate still demands load, because that is the fault's
definition, because a future driver or power state may idle low, and because the power
corroboration is meaningless without it.

### 3.2 What `nvidia-smi` on these hosts does and does not expose

Verified against `nvidia-smi --help-query-gpu` and `/usr/local/cuda/include/nvml.h`
(driver 580.173.02) on `node0`:

| Field | Available? | Notes |
|---|---|---|
| `clocks.current.sm` / `clocks.current.graphics` | **yes** | the fault signal |
| `clocks.max.sm` | **yes** | 3,003 MHz |
| `clocks.applications.graphics` | **yes** | 2,418 MHz; equals the *default* app clock |
| `clocks.applications.sm` | **NO** | `Field "clocks.applications.sm" is not a valid field to query.` — **the field named in the task does not exist** |
| `clocks.current.memory` / `clocks.max.memory` | **N/A** | memory clock is unreadable on GB10 |
| `clocks_event_reasons.active` | **yes** | the governor readout; **`0x0` during the fault** |
| `clocks_event_reasons.{gpu_idle,applications_clocks_setting,sw_power_cap,hw_slowdown,hw_thermal_slowdown,sw_thermal_slowdown,hw_power_brake_slowdown}` | **yes** | decoded from `nvml.h` bit values (§3.3) |
| `clocks_event_reasons_counters.*` | **yes** | cumulative µs; **not** a fault indicator (see below) |
| `power.draw` / `power.draw.average` / `power.draw.instant` | **yes** | the corroborating signal |
| `power.limit`, `power.default_limit`, `enforced.power.limit` | **N/A** | **power limits are not readable on GB10** |
| `pstate` | **yes** | `P0` under load |
| `persistence_mode` | **yes** | `Enabled` on all four |
| `compute_mode` | **yes** | `Default` |
| `spbm` hwmon power caps | **ABSENT on this fleet** | §4.6 — MISS |

Trap to avoid: the cumulative `clocks_event_reasons_counters.sw_thermal_slowdown` reads
274,753,154 µs (~275 s) on `node0` while the *active* flag reads "Not Active". The
counters are lifetime totals since driver load and accumulate during normal boost
transients; **they are not evidence of current throttling**. Only the `active` bitmask is.

### 3.3 Throttle bitmask, from `nvml.h` (not inferred)

```
nvmlClocksEventReasonGpuIdle                   0x001
nvmlClocksEventReasonApplicationsClocksSetting 0x002   (== "user defined clocks")
nvmlClocksEventReasonSwPowerCap                0x004
nvmlClocksThrottleReasonHwSlowdown             0x008
nvmlClocksEventReasonSyncBoost                 0x010
nvmlClocksEventReasonSwThermalSlowdown         0x020
nvmlClocksThrottleReasonHwThermalSlowdown      0x040
nvmlClocksThrottleReasonHwPowerBrakeSlowdown   0x080
nvmlClocksEventReasonDisplayClockSetting       0x100
```

Note there is **no "platform power / PD" bit**. That is exactly why this fault is silent:
the thing that is stuck is not represented in the mask.

Observed during the run: one transient `0x4` (SW power cap) sample on `node2` at a
**healthy** 2,470 MHz. A non-zero bit is therefore not by itself a fault; the gate treats
`sw_power_cap`/`sw_thermal`/`app_clocks_locked` as WARN and only `hw_slowdown`,
`hw_thermal_slowdown` and `hw_power_brake_slowdown` as FAIL.

### 3.4 Verdict logic and thresholds

- A sample is **busy** when `utilization.gpu >= LOAD_MIN_PCT` (default 50).
- If no sample is busy → **INCONCLUSIVE** (exit 2). A gate that passes an untested node is
  worse than no gate.
- On busy samples, `SM_MIN = min(clocks.current.sm)`.
- **FAIL** if `SM_MIN < 1500 MHz` or `SM_MIN < 50%` of `clocks.max.sm`, or if any hardware
  throttle bit is active.
- On FAIL, the gate names the likely variant: **power-latch signature** when
  `SM_MIN <= 1000 MHz` *and* mean busy power `<= 30 W`; otherwise **low-clock** with a
  prompt to check for a mis-set `-lgc` cap first.

Threshold rationale: every fault value ever reported (fleet 611–728; community 481–858)
is below 900 MHz, and every healthy value recorded on this fleet is at or above
2,171 MHz. A 1,500 MHz floor sits ~2× above the highest fault value and ~30% below the
lowest healthy value, so it separates them with wide margin and does not depend on which
engine is running.

### 3.5 The gate cannot inject the fault, so the FAIL path is proven by selftest

The fault cannot be reproduced on a live node without writing to it (`-lgc`), which would
violate the read-only rule and disturb the benchmark. The verdict function is therefore
proven against synthetic fixtures with `--selftest`, driving the **real** decision code:

```
$ probes/check_spark_clocks.sh --selftest
=== selftest: verdict logic against synthetic fixtures ===
The real compute_verdict() is driven with fake collector streams.
The fault cannot be injected on a live node without breaking the
read-only rule, so this is how the gate's FAIL path is proven.

  ok   healthy busy -> PASS                                 -> PASS
  ok   fleet fault band 611-728 MHz + 15 W -> FAIL          -> FAIL
  ok   721 MHz but 70 W (not the power latch) -> FAIL       -> FAIL
  ok   idle low clock 210 MHz -> IDLE (not tested)          -> IDLE
  ok   HW thermal slowdown under load -> FAIL               -> FAIL
  ok   idle at healthy 2405 MHz only -> IDLE (not tested)   -> IDLE
  ok   busy + idle mix -> PASS on the busy samples          -> PASS

selftest: ALL PASS
selftest exit=0
```

The third fixture matters: it proves the gate still FAILs a low clock when power is
*normal*, so it is not merely a power test. The fourth and sixth prove it refuses to
judge an idle node.

### 3.6 Synthetic load — documented, deliberately not run

If no node is busy, the honest answer is INCONCLUSIVE. The correct fix is to run the gate
while a benchmark is active. A synthetic load that does not touch the model endpoint would
be a saturating FP32 GEMM, e.g. per node:

```bash
ssh sparkN 'python3 - <<EOF
import torch, time
a = torch.randn(8192, 8192, device="cuda"); b = torch.randn(8192, 8192, device="cuda")
t = time.time()
while time.time() - t < 10: a @ b
torch.cuda.synchronize()
EOF'
```

**This was not run.** It allocates GPU and UMA memory on a node already running a live
benchmark, which is exactly the kind of disturbance the brief forbids. It is recorded here
as the documented option only, for use on a released node. The community equivalent
(hoesing's `spark-gpu-throttle-check`) uses a 4096×4096 SGEMM with a 1,400 MHz FAIL
threshold.

---

## 4. Per-node results with raw evidence

### 4.1 The gate, run for real (run 2 of 3; runs 1–2 clean PASS)

```
$ TZ=Australia/Sydney date   # 2026-09-11 16:13:02 AEST
$ probes/check_spark_clocks.sh
Spark GB10 clock / power-latch gate — 2026-09-11 06:13:02 UTC
nodes: node0 node1 node2 node3 | 15 samples/node every 1.5s | load>=50% | SM floor 1500 MHz or 50% of max
mode: READ-ONLY (nvidia-smi --query-gpu + /sys/class/hwmon reads). Safe during a live benchmark.

NODE     VERDICT SM_MIN   SM_MED   SM_MAX   BUSY    UTILMAX TEMP   PWRBUSY APPSM     GOVERNOR
-------- ------ -------- -------- -------- ------- ------- ------ ------- --------- --------
node0   PASS   2398     2424     2437     15/15   96      84     58.2    2418      none
           └─ busy at 2398-2437 MHz (80% of max), 58.2 W, no throttle
           └─ pstate=P0 persistence=Enabled gr=2424 max_sm=3003 spbm_caps=absent reasons=0x0000000000000000
node1   PASS   2444     2470     2483     15/15   96      87     56.7    2418      none
           └─ busy at 2444-2483 MHz (81% of max), 56.7 W, no throttle
           └─ pstate=P0 persistence=Enabled gr=2470 max_sm=3003 spbm_caps=absent reasons=0x0000000000000000
node2   PASS   2444     2457     2489     15/15   96      84     53.3    2418      none
           └─ busy at 2444-2489 MHz (81% of max), 53.3 W, no throttle
           └─ pstate=P0 persistence=Enabled gr=2450 max_sm=3003 spbm_caps=absent reasons=0x0000000000000000
node3   PASS   2385     2398     2424     15/15   96      87     52.9    2418      none
           └─ busy at 2385-2424 MHz (79% of max), 52.9 W, no throttle
           └─ pstate=P0 persistence=Enabled gr=2398 max_sm=3003 spbm_caps=absent reasons=0x0000000000000000

GATE: PASS — every node was observed busy with SM clocks above the floor.
GATE EXIT=0
```

Run 1 (identical command, `--verbose`, ~2 minutes earlier) gave the same verdicts with
`SM_MIN`/`SM_MAX` of 2398–2437 (80%, 62.9 W), 2444–2489 (81%, 60.7 W), 2450–2489 (82%,
57.9 W), 2385–2424 (79%, 58.2 W). Two independent runs, 15/15 busy on every node, zero
failures. Per-sample raw lines for run 1 are in the gate's `--verbose` output.

### 4.1b Run 3 — the WARN path fired in the field, as the nodes warmed

A third run several minutes later caught the fleet drifting warmer (peak 89 °C) and the
gate's WARN path exercised for real, on two different soft-throttle bits:

```
node0   PASS   2379     2398     2437     15/15   96      87     60.6    2418      none
node1   WARN   2333     2450     2470     15/15   96      89     59.4    2418      none
           └─ clocks healthy but flags present: sw_thermal
           └─ ... reasons=0x0000000000000000 0x0000000000000020
node2   WARN   2444     2450     2489     15/15   96      87     55.6    2418      none
           └─ clocks healthy but flags present: sw_power_cap
           └─ ... reasons=0x0000000000000000 0x0000000000000004
node3   PASS   2320     2392     2405     15/15   96      87     55.6    2418      none
GATE: PASS — every node was observed busy with SM clocks above the floor.
```

Three things this shows, and one caveat it creates:

- The WARN path is real and correctly calibrated: it flagged `0x20` (SW thermal) and
  `0x04` (SW power cap) on **different** nodes while still reporting the clocks healthy,
  rather than either ignoring them or failing the nodes.
- All four nodes remained **15/15 busy** and **well above the floor** — `SM_MIN` 2,320 MHz
  is still 2.7× the highest fault value ever reported (858 MHz) and 3.8× this fleet's own
  lowest fault reading (611 MHz). This is ordinary soft throttling, not the fault: the
  mask is **non-zero**, whereas the fault's signature is a non-zero *clock collapse* with a
  **zero** mask.
- Soft throttling *is* depressing clocks a little as the bench proceeds — `SM_MIN` moved
  from 2,385–2,444 MHz (runs 1–2) to 2,320–2,444 MHz (run 3), a few percent. That is a
  visible, gradual thermal effect any harness can track, unlike the silent latch.
- **Caveat:** because the gate WARNs on soft-throttle bits, a WARN is not a clean PASS. It
  still exits 0 (the clocks are healthy and the bench is not invalidated), but on a
  thermal-limited node WARN means "clocks are fine, and also the node is warm enough to be
  soft-throttling" — worth reporting alongside throughput, especially if comparing runs
  taken at different temperatures.

### 4.2 The exact per-node command

```bash
for n in node0 node1 node2 node3; do
  ssh -o BatchMode=yes $n "nvidia-smi --query-gpu=clocks.current.sm,clocks.current.graphics,\
clocks.max.sm,clocks.applications.graphics,utilization.gpu,power.draw.average,temperature.gpu,\
pstate,persistence_mode,clocks_event_reasons.active --format=csv,noheader,nounits"
done
```

### 4.3 Governor state — all four nodes, verbatim

```
=== node0 === kernel/driver: 6.17.0-1029-nvidia / 580.173.02
        Graphics                                       : 2398 MHz
        SM                                             : 2398 MHz
    Applications Clocks
        Graphics                                       : 2418 MHz
    Default Applications Clocks
        Graphics                                       : 2418 MHz
    Max Clocks
        Graphics                                       : 3003 MHz
        SM                                             : 3003 MHz
    Clocks Event Reasons
        Idle                                           : Not Active
        Applications Clocks Setting                    : Not Active
        SW Power Cap                                   : Not Active
        HW Slowdown                                    : Not Active
            HW Thermal Slowdown                        : Not Active
            HW Power Brake Slowdown                    : Not Active
        Sync Boost                                     : Not Active
        SW Thermal Slowdown                            : Not Active
        Display Clock Setting                          : Not Active
```

Identical block on `node1` (SM 2450, kernel 6.17.0-1029), `node2` (SM 2450, kernel
6.17.0-1031) and `node3` (SM 2385, kernel 6.17.0-1031): **every clock event reason
"Not Active" on all four**, Applications Clocks == Default Applications Clocks (2,418 MHz),
Max Clocks 3,003 MHz. No node has a clock policy applied.

Why this matters: this is precisely the signature that makes the fault dangerous — a
healthy node and a faulted node both show a clean throttle mask. **The mask cannot be used
to certify a node; only the clock-under-load can.**

### 4.4 No clock cap is applied anywhere

```
$ systemctl list-unit-files | grep -icE "gb10-clock|lock-gpu|clock-cap"
0        # on node0, node1, node2 and node3
```

`Applications Clocks == Default Applications Clocks` on all four confirms no
`nvidia-smi -ac` deviation. Variant #2 of §2.3 is therefore **ruled out on this fleet**,
which is what lets the gate attribute a future low-clock failure to the power latch rather
than to a mis-set cap.

The only boot-path GPU agent is `nvidia-persistenced`, with a drop-in that switches it
from the packaged default to persistence mode:

```
$ cat /etc/systemd/system/nvidia-persistenced.service.d/nv-persistence-override.conf
[Service]
ExecStart=
ExecStart=/usr/bin/nvidia-persistenced --persistence-mode --verbose

[Install]
WantedBy=
WantedBy=basic.target
```

A search of `/etc/systemd`, `/usr/lib/systemd/system`, `/etc/rc.local` and `/etc/nvidia`
for `applications-clock`, `lock-gpu-clock`, `nvidia-smi -ac`, `--power-limit` and
`persistence-mode` returned **only** these `nvidia-persistenced` lines. **Nothing in the
boot path sets clocks or power limits on this fleet. There is no systemd-ordering bug to
find here.**

### 4.5 The trigger on this fleet: an OTA that replaced the kernel and driver

Boot history (`journalctl --list-boots`, tail) shows the 2026-09-02 morning:

```
node0  -1  Tue 2026-09-01 22:00:14 AEST → Wed 2026-09-02 10:22:08 AEST
         0  Wed 2026-09-02 10:23:31 AEST → still running
node1   0  Tue 2026-09-01 22:00:34 AEST → still running
node2  -2  Wed 2026-09-02 10:40:08 → Wed 2026-09-02 11:56:24   (kernel 6.11.0-1016)
        -1  Wed 2026-09-02 11:57:49 → Wed 2026-09-02 12:19:45   (kernel 6.17.0-1031)
         0  Wed 2026-09-02 12:21:10 → still running
node3  -2  Wed 2026-09-02 10:40:58 → Wed 2026-09-02 11:56:24
        -1  Wed 2026-09-02 11:57:48 → Wed 2026-09-02 12:19:46
         0  Wed 2026-09-02 12:21:09 → still running
```

`/var/log/apt/history.log` on 2026-09-02 shows a DGX Spark OTA that **replaced the kernel
and the NVIDIA driver**:

```
Upgrade: linux-image-nvidia-hwe-24.04        (6.11.0-1016.16  → 6.17.0-1031.31)
         linux-nvidia-hwe-24.04              (6.11.0-1016.16  → 6.17.0-1031.31)
         nvidia-driver-580-open              (580.95.05        → 580.173.02)
         libnvidia-compute-580               (580.95.05        → 580.173.02)
         nvidia-kernel-source-580-open       (580.95.05        → 580.173.02)
         dgx-spark-ota-update-meta           (25.09.3          → 26.03.1)
         nvidia-firmware-580-580.173.02
Install: linux-image-6.17.0-1031-nvidia 6.17.0-1031.31, linux-modules-nvidia-580-open-…
```

Apt ran at 10:17:47, 11:54:39, 12:06:09 and 12:21:11 on 2026-09-02 — matching the reboot
sequence above. Every reboot in that sequence was **clean**: the end of each previous boot
is `Reached target reboot.target - System Reboot` / `Shutting down`, with **no OOM kill,
no panic, no thermal trip and no watchdog event** in the prior boot's journal.

Current kernels: `node0`/`node1` `6.17.0-1029-nvidia`; `node2`/`node3`
`6.17.0-1031-nvidia`. All four run driver `580.173.02`.

**Interpretation (INFERENCE, explicitly):** on this fleet the fault is temporally
associated with a **kernel/driver OTA plus reboot**, matching the community's most common
antecedent class ("I rebooted the machine after some APT package updates"). It is *not* an
association with a crash or a hard power-off, and it is *not* a systemd-ordering bug —
§4.4 shows nothing in the boot path touches clocks. The coherent reading is that the
reboot is the occasion on which the USB-PD controller renegotiates the power contract, and
occasionally that negotiation latches low; the boot path is a witness, not a cause.
**I have no direct measurement proving clocks were low during those specific boots** — the
association is drawn from the prior campaign's prose plus the boot/apt timeline, and the
prior record does not say which two nodes failed or in which boot. Treat this paragraph as
the weakest link in §2, not as an established mechanism.

One recurring message at boot is a red herring and should not be chased:

```
mlx5_core 0000:01:00.0: mlx5_pcie_event:326:(pid 12): Detected insufficient power on the PCIe slot (27W).
```

This is the ConnectX-7 NIC reporting its own slot budget (the GPU talks over NVLink C2C),
and the community consensus is that it is unrelated to the GPU clock fault. It appears on
`node2`/`node3` in boot `-1`.

### 4.6 MISS: the `spbm` power-cap readout does not exist here

The community's only direct readout of the latched power ceiling is the `spbm` hwmon node.
**It is not present on this fleet.**

```
$ for f in /sys/class/hwmon/hwmon*/; do echo "$f -> $(cat $f/name)"; done
/sys/class/hwmon/hwmon0/ -> acpitz
/sys/class/hwmon/hwmon1/ -> nvme
/sys/class/hwmon/hwmon2/ -> mlx5
/sys/class/hwmon/hwmon3/ -> mlx5
/sys/class/hwmon/hwmon4/ -> mlx5
/sys/class/hwmon/hwmon5/ -> mlx5
/sys/class/hwmon/hwmon6/ -> mt7925_phy0
```

No `spbm`; no `power11_cap`/`power13_cap` anywhere under `/sys` (only unrelated
HDA-audio `power_caps` widget attributes). Combined with `power.limit` being `N/A` on
GB10, **there is no way to read the power ceiling on these hosts.** The gate reports
`spbm_caps=absent` rather than silently skipping it. Consequence: the platform-power
*mechanism* cannot be confirmed locally — only its *symptom* (low clock + low power under
load) can be detected.

---

## 5. Risk to the running benchmark

**Verdict: the current measurements are NOT at risk from this cause, on the evidence
gathered here.**

Supporting evidence:

| Node | SM clock under load | vs top of fault band (858 MHz) | Power under load | vs fault band (~5–16 W) | Throttle |
|---|---|---|---|---|---|
| node0 | 2,398–2,437 MHz | **2.8×** | 58.2–62.9 W | 3.6–12× | `0x0` |
| node1 | 2,444–2,489 MHz | **2.9×** | 56.7–60.7 W | 3.5–12× | `0x0` |
| node2 | 2,444–2,489 MHz | **2.9×** | 53.3–57.9 W | 3.3–11× | `0x0` |
| node3 | 2,385–2,424 MHz | **2.8×** | 52.9–58.2 W | 3.3–11× | `0x0` |

All four were observed busy at **15/15 samples in each of two independent runs**, at
93–96% utilisation, in P0, with zero hardware throttle, no clock cap and persistence mode
enabled.

The **clock** evidence is the unambiguous part: 2,385–2,489 MHz is 2.8–2.9× the highest
fault value ever reported (858 MHz) and 3.3–4.1× this fleet's own 611–728 MHz. The
**power** evidence is corroborating but softer, because the degraded power band is wider
than the degraded clock band — most reports are 5–16 W, but one measured a board flat at
35 W while still capped. 53–63 W is therefore above every reported degraded figure, but by
1.5× in the worst case rather than by an order of magnitude. Neither signal alone would
settle it; together with a clean throttle mask, P0 and 93–96% utilisation they do.

Caveats that bound this verdict — read them before quoting it:

1. **This is a snapshot, not continuous monitoring.** My total coverage is roughly two
   25-second gate runs plus ~15 minutes of ad-hoc sampling. The community reports the
   fault can appear on a node that was healthy ("I can leave a Spark idling in a good state
   and still have this pop up"). A PASS at 16:11 AEST does not cover a bench that runs for
   hours afterwards.
2. **The benchmark load is bursty, and the gate can land in a lull.** A separate 30-sample
   / ~60 s duty-cycle run measured a mean utilisation of only **31.8–31.9%**, with
   **10/30** samples at ≥50% util. In that window the load was genuinely idle for stretches
   (0% util, 13.8–17.2 W). Both gate runs happened to land in busy phases. **If the gate is
   run during a lull it returns INCONCLUSIVE (exit 2), not PASS** — which is correct, but
   means a single INCONCLUSIVE is not a fault and should be re-run, ideally with more
   samples (`CLOCK_SAMPLES=40` ≈ 60 s).
3. **A throughput collapse mid-run should still be treated as suspect** even though the
   gate passed, because the fault is silent and can appear without a reboot. The cheap
   mitigation is to re-run the gate at the *end* of the measurement window as well as the
   start, and to cross-check against the engine's own published peak.
4. **The fault band on this fleet (611–728 MHz) was never captured raw.** I am comparing
   against prose from a prior session, not against an `nvidia-smi` capture. The comparison
   is sound in magnitude (2,385 vs 728 MHz is unambiguous) but the fleet-specific band is
   less well evidenced than the community band.
5. **The fleet warmed measurably during the observation and began soft-throttling.**
   Run 3 (§4.1b) caught SW thermal (`0x20`) on `node1` at 89 °C and SW power cap (`0x04`)
   on `node2`, with `SM_MIN` down a few percent (2,320–2,444 MHz vs 2,385–2,444 MHz in
   runs 1–2). This is *not* the fault — the mask is non-zero and the clocks stay far above
   the floor — but it does mean **throughput comparisons between runs taken at different
   node temperatures are not strictly like-for-like**. Report temperature alongside any
   decode number, and prefer comparing runs taken in the same thermal state.

What would change the verdict: any node reporting sub-1,000 MHz under load with a clean
throttle mask. The gate exists to make that a one-command check rather than a hidden
throughput regression.

---

## 6. Using the gate

```bash
# gate all four nodes (defaults: 15 samples/node, 1.5 s apart, ~22 s per node, parallel)
probes/check_spark_clocks.sh

# prove the verdict logic without touching the fleet
probes/check_spark_clocks.sh --selftest

# longer window, to survive a lull in a bursty benchmark
CLOCK_SAMPLES=40 probes/check_spark_clocks.sh

# a subset, or a different fleet
probes/check_spark_clocks.sh --nodes node0,node2
SPARK_NODES="node0 node1" probes/check_spark_clocks.sh

# per-sample raw evidence
probes/check_spark_clocks.sh --verbose
```

| Exit | Meaning | Action |
|---|---|---|
| 0 | PASS — all nodes busy with clocks above the floor | trust the bench (subject to §5 caveats) |
| 1 | FAIL — busy below the floor, or HW throttle | **do not trust the bench**; see the named variant |
| 2 | INCONCLUSIVE — a node was never seen busy | re-run under load, or on a released node run the §3.6 synthetic load |
| 3 | ERROR — SSH/parse failure | fix access; do not treat as a pass |

Tuning: `SPARK_NODES`, `CLOCK_SAMPLES`, `CLOCK_INTERVAL`, `LOAD_MIN_PCT` (50),
`SM_FLOOR_MHZ` (1500), `SM_RATIO_MIN_PCT` (50), `PWR_LOW_W` (30), `SPARK_SSH_OPTS`.

**The gate is strictly read-only.** It issues only `nvidia-smi --query-gpu=…` and reads a
few `/sys/class/hwmon` files over SSH. It never sets clocks, power limits or persistence
mode, never reboots, and never touches a container or the inference endpoint. It is safe to
run against a live benchmark — that is the condition under which it is *most* useful,
because the fault is only visible under load.

If the gate ever reports the power-latch variant, the fix is **not** `nvidia-smi`: the
community remedy is a full AC power drain (unplug the brick at **both** the device and the
wall, wait ≥5 minutes, reconnect, boot, re-check). Note that `-lgc` does not fix this fault
and may silently no-op on GB10. **Any such action is a fleet intervention and is outside
this task's authority and outside this document.**

### Wiring in

The gate is in-repo at `probes/check_spark_clocks.sh` (executable) and is the artefact to
point a pre-bench checklist at. It is deliberately not referenced from `RUNBOOK.md` or
`STATUS.md`, which hold live campaign state owned by the lead lane (per the repo's
standing instruction not to edit mid-flight campaign files); the campaign lead should add
the one-line reference. `public-repo/docs/` mirrors the findings documents — this note was
not copied there, since publication is a separate, sanitised step.

---

## 7. What I could not determine (recorded as MISS, not backfilled)

| Open question | Status |
|---|---|
| Which two nodes were stuck, and in which boot | **MISS** — not in the prior record |
| Whether "restart" meant a warm reboot or an AC drain | **MISS** — not in the prior record |
| Raw `nvidia-smi` output of a faulted node on this fleet | **MISS** — never captured |
| The power ceiling itself (`spbm` caps, `power.limit`) | **MISS** — `spbm` absent; `power.limit` N/A on GB10 |
| Memory clock behaviour during the fault | **MISS** — `clocks.current.memory` N/A on GB10 |
| pstate while idle | **MISS** — idle clock (2,405–2,411 MHz) and power (13.8–17.2 W) at util 0% were captured, but a 60-sample retry to catch idle pstate found no idle samples |
| Direct proof clocks were low during the 2026-09-02 boots | **MISS** — association is temporal, from prose + apt/boot timeline |
| Whether a warm reboot clears the fault on *these* nodes | **MISS** — the fleet's own record shows only that a "restart" was followed by healthy clocks |
| Whether the fault can appear without a reboot on this fleet | **MISS** — no continuous monitoring; a longer observation window would be needed |

## 8. Sources

Local: the prior session record (path above); `nvidia-smi` and `journalctl` on
`node0`–`node3`; `/usr/local/cuda/include/nvml.h` (driver 580.173.02);
`/var/log/apt/history.log`; `/etc/systemd/system/nvidia-persistenced.service.d/`.
External (community, labelled as such throughout):
[361296 (513 MHz cap)](https://forums.developer.nvidia.com/t/investigating-513mhz-cap-for-gpu/361296) ·
[376039 (721 MHz pinned)](https://forums.developer.nvidia.com/t/dgx-spark-gb10-gpu-clock-pinned-at-721-mhz-under-full-load-no-throttling-not-liftable-via-nvidia-smi/376039) ·
[376239 (5-min wait)](https://forums.developer.nvidia.com/t/gpu-clock-bug-looks-like-5-min-wait-is-enough/376239) ·
[366590 (power limited after crash)](https://forums.developer.nvidia.com/t/gb10-is-power-limited-after-crash/366590) ·
[374274 (lower performance after update)](https://forums.developer.nvidia.com/t/suddenly-much-lower-gpu-performance-in-inference/374274) ·
[370304 (15 W / 650 MHz loop)](https://forums.developer.nvidia.com/t/dgx-spark-grace-blackwell-gb10-performance-drop-gpu-trapped-in-15w-650mhz-loop-with-50-c-artificial-t-limit-temp/370304) ·
[hoesing/spark-gpu-throttle-check](https://github.com/hoesing/spark-gpu-throttle-check) ·
[joeynyc/spark-doctor](https://github.com/joeynyc/spark-doctor) ·
[tonyd2wild/DGX-Spark-Hard-Poweroff-Fix](https://github.com/tonyd2wild/DGX-Spark-Hard-Poweroff-Fix)
