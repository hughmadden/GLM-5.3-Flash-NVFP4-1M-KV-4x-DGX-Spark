# TEST ITERATION & PROFILING — how to go faster and see more (2026-09-11 AEST)

Answers "how do we speed up the test/iterate cycle and gather profile/debug data" for the EXL3
persistence lane. Companion to `FIX-PLAN-20260911.md` and `CONCURRENCY-TESTS-20260910.md`.

---

## 1. Where the time actually goes

| Loop | Cost today | Bottleneck |
|---|---|---|
| Unit tests (persistence, no engine) | **0.07 s** to import all 7 modules, **0.70 s** `docker run` + import, **~190 s** for 184 tests | a few fsync-heavy perf tests inside the suite, not the framework |
| Python change → verified | **~1 s** (no rebuild) via the `PYTHONPATH` overlay | nothing — this loop is already fast |
| C/CUDA change → verified | **15–25 min** image rebuild + **4 × 10.6 GB** re-stage + reload | image rebuild and fan-out |
| Engine boot → `/health=200` | **~390 s** | 175 GB weight read + quant/dequant + KV profiling + warmup (`init engine … took 134 s` is only the last part) |
| Engine crash → recovered | **~7 min** (`down` → ritual → flusher → `up`) | same boot cost, plus a manual teardown because the head hangs draining the client socket |

So the unit loop is already near-optimal; **the engine boot dominates everything**. Work on that.

---

## 2. Make the engine loop faster

**2.1 Do not relaunch for request-level questions.** A cap bisect, a decode bench, needle probes and
concurrency sweeps all run against *one* running engine. Sequence them so the destructive step is
last: measure decode first, then walk the prompt-size ladder. Most of the wall-clock lost in this
campaign was relaunching between questions that did not need it.

**2.2 Warm the engine before measuring anything.** The first request after a restore is cold — I
measured **ttft = 108 s** on one first request and 12.9 s on another; both would corrupt a benchmark.
Always send one throwaway small request, then measure.

**2.3 Shrink the boot for iteration.** `MAX_MODEL_LEN` is the lever with the highest ratio: at
850,000 the engine profiles and allocates a ~4.4 M-token KV pool. For config/logic iteration,
`MAX_MODEL_LEN=20000` removes most of the KV profiling and allocation work while changing nothing
about the code paths under test. Keep the full value for final numbers only.

**2.4 Split the env into base + arm overrides.** Right now every A/B is a `sed` on a 250-line env file
and a full relaunch. Keep `env.tp4.fleet` as the base and make the arm a *small diff* applied by a
script (`cycle.sh on|off` already does this and logs the sha256 of the resulting file). A/B then
becomes "one command per arm", and the receipt records which arm produced which number.

**2.5 Consider the flusher's effect on reload.** The recipe's unconditional page-cache flusher runs
every 60 s for the whole boot, which is correct for production qualification — but it deliberately
throws away the page cache that just served the 175 GB weight read, so every relaunch re-reads from
NVMe. For *iteration only*, a variant profile with the flusher off may cut boot time; UMA is 128 GB
against ~175 GB of weights, so expect a partial win, not a dramatic one. Label such runs
"iteration profile" and never quote their throughput as qualification numbers.

**2.6 Don't rebuild for Python changes.** The persistence package is pure Python and imports neither
torch nor vLLM, so mounting the repo over the installed package is enough:

```bash
rsync -a --delete --exclude __pycache__ "$PKG/" node0:~/pkg-src/
# then add to docker run: -v /home/user/pkg-src:/pkg:ro -e PYTHONPATH=/pkg
```
That is how all 8 fixes in `FIX-PLAN-20260911.md` were verified in seconds. Reserve image rebuilds for
C/CUDA or dependency changes.

**2.7 Parallelise what actually parallelises.** Only one TP4 engine arm can run at a time (it owns all
four Sparks), but when the engine is **down** the four Sparks are four independent workers: that is
how the 4-way concurrency fan-out ran. Use them for unit suites, log analysis and profiling
post-processing, not for engine arms.

---

## 3. Crash-resilient experiment design (lessons already paid for)

1. **Run every probe detached, writing to a timestamped log**, then poll the log. Two probe runs were
   lost this campaign because a foreground `ssh`/`docker run` was killed by a tool timeout and the
   output went with it. Copy the on-Spark helper and run it with `setsid nohup`, ending with an
   `AB_DONE` marker line so a poller can tell completion from silence.
2. **Measure before you break.** Put the destructive probe last in the script.
3. **Ladder smallest → largest and stop at the first failure**, with small `max_tokens` for the
   memory probe (16) and a generous per-request timeout.
4. **Record a baseline before each arm** — `dmesg | grep -c NV_ERR_NO_MEMORY`, `/health`, container
   exit code — so "did it crash?" is a delta, not a guess. (Note `dmesg` is a ring buffer: the count
   can *decrease* when it wraps, so also keep the last N NVRM timestamps.)
5. **One arm script** that does set-arm → `down` → ritual → flusher → detached `up` → wait `/health`,
   logging each stage. `cycle.sh on|off` is the current shape; extend it rather than hand-running steps.
6. **Never trust a clean exit code as "no crash".** Every engine death here exited **0** with
   `OOMKilled=false` because vLLM's SIGTERM handler shut down gracefully. The real signals are the
   NVRM lines in `dmesg` and `/health` going dead while PID 1 stays alive.

---

## 4. Profile and debug data — what to collect and how

### 4.1 UMA memory timeline — the instrument for the request-cap crash
The cap is an NVIDIA-driver allocation failure on **unified** memory, so the decisive data is a
1 Hz timeline across a bisect step:

```bash
# sampler (run on each rank during the probe; write to <RECEIPTS>/uma-<rank>-<ts>.csv)
while :; do
  printf '%s,%s,%s,%s\n' "$(date +%s)" \
    "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" \
    "$(awk '/^Cached/{print $2}' /proc/meminfo)" \
    "$(docker stats --no-stream --format '{{.MemUsage}}' glm53-exl3-tp4-head 2>/dev/null)" >> "$CSV"
  sleep 1
done
```
Add `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader` if the GB10 driver exposes
it (verify first — it is a UMA part), and always keep `sudo dmesg -T | grep NVRM` alongside. The last
samples before the failure show which pool grew: weights, KV pool, prefill workspace, or persistence
staging.

### 4.2 vLLM `/metrics` — the cheapest engine telemetry, and the decode answer
Scrape before and after a request and diff. The counters that matter here:
- `vllm:spec_decode_num_accepted_tokens_total`, `…_num_draft_tokens_total`, `…_num_emitted_tokens_total`
  → **acceptance length** (accepted/drafted) and, with `num_speculative_tokens=7`, **steps** and hence
  **ms/step**. This is what exposed the chunk-counting error: our acceptance was 0.43–0.68, i.e. the
  prose regime, where Mia's own published figure is ~27 tok/s.
- `vllm:kv_offload_store_bytes_total`, `…_store_size_count`, `…_lookup_sync_delay_seconds_count`,
  `…_allocation_failure_total` → how much the persistence tier is doing per step.
- `vllm:gpu_cache_usage_perc`, `vllm:num_requests_running`, `vllm:num_requests_waiting` → whether a
  slowdown is queueing or compute.

Always measure decode from **`usage.completion_tokens`** with `stream_options: {"include_usage": true}`,
never from the count of streamed chunks.

### 4.3 Torch profiler inside the engine
Set `VLLM_TORCH_PROFILER_DIR=/profile` (bind-mounted) and drive `POST /start_profile` … `POST /stop_profile`
around a fixed request, then read the Chrome trace. This gives per-kernel decode cost and shows
directly whether the offload worker's lookup/store sits on the critical path — the mechanism behind
the **77 tok/s (persistence off) vs ~12 tok/s (on)** signal.

### 4.4 Nsight Systems for the cap
`nsys profile --cuda-memory-usage=true …` around one large-prompt request gives the CUDA allocation
timeline and the true peak. That is the cleanest way to attribute the UMA exhaustion to a specific
allocation site rather than guessing between weights, KV, workspace and staging.

### 4.5 Engine-side logging
- `--enable-logging-iteration-details` for per-iteration detail.
- `VLLM_LOGGING_LEVEL=DEBUG` when chasing an init or connector problem.
- Keep the existing startup observer receipts (`<RECEIPTS>/observer/`) — metadata only, no
  tensor data, and they already record the KV geometry.

### 4.6 One crash-forensics bundle, always the same shape
On any failure, capture in one timestamped directory: `docker logs <head>`, `docker inspect`
(ExitCode, OOMKilled, FinishedAt), `dmesg -T | grep NVRM`, `/proc/meminfo`, `/metrics` snapshot,
`<RECEIPTS>/up-*.log`, and the observer receipts. Having that bundle from all three outages in
this campaign is what made the diagnosis (NVRM OOM → graceful SIGTERM → hung drain) immediate.

---

## 5. Guardrails — a faster loop must not lie

- The recipe's memory ritual and unconditional flusher, and the full `MAX_MODEL_LEN`, exist for
  production qualification. A reduced boot profile is **iteration only**; every number must carry the
  config it came from.
- Keep the **30 s load budget** and the per-test **watchdog** in the unit harness: they are what make
  "fast" safe (a deadlock reports as `HANG` instead of eating the run).
- Keep dummy data dummy: no test should start an engine, touch a GPU, or open a socket to a real rank.
- The one number worth re-measuring before any optimisation work is the **persistence decode penalty**
  (77 vs 12 tok/s). If it holds on a warmed median-of-5 A/B, it reorders the whole plan: the disk tier
  is currently costing more than it returns at short context.
