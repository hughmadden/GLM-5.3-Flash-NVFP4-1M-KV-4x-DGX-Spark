# benchmark/ — engine-side harnesses

Everything here talks to a **live** vLLM endpoint. Nothing in this directory
starts, stops or reconfigures an engine. Read the measurement rules at the bottom
before using any of it: this campaign produced three separate false conclusions
from benchmarks that violated them.

| Harness | What it measures |
|---|---|
| `bench_c1c6.py` | Decode throughput at concurrency 1..6. For each level, `--rounds` waves of C parallel requests; reports aggregate output tok/s, per-stream decode tok/s and TTFT, and scrapes spec-decode acceptance from `/metrics`. Prompts are code/reasoning-flavoured (realistic acceptance rates) and unique per request to defeat prefix caching. |
| `bench_robust.py` | The statistical upgrade: N waves per level, mean ± stdev and 95 % CI, per-request streaming TTFT + TPOT + decode tok/s, P50/P90/max wall, a mixed code/reasoning/prose prompt set (acceptance varies by type), and **interleaved level order** to decorrelate thermal and scheduler drift. |
| `probe_evict_refill.py` | The persistence tier's eviction/refill behaviour. The disk tier's own 1 TB quota is unreachable at test scale, so the reachable and interesting direction is forced: warm a target prompt, flood the engine with unique long prompts to push the target out of the GPU KV pool (~3.4 M tokens at this configuration), then re-request it and classify the serve path (GPU cache / disk tier / full recompute) from TTFT and the `kv_offload` + prefix-cache counters. |
| `run_lil_c1c6.sh` | Wrapper for the Local Inference Lab standard C1..C6 decode bench: captures verbatim bench output, the persistence/offload metric deltas that frame it, and a quantified record of any ambient (non-bench) traffic that shared the engine during the run. |
| `vllm-ref/offload-modules.txt` | Reference listing of the pinned vLLM offloading modules (paths + file contents) used when porting the persistence ABI. Not executable. |

## Usage

```sh
# C1..C6 decode sweep against your own endpoint
python3 bench_c1c6.py --url http://HEAD:8000 --rounds 3 --max-tokens 700

# robust mixed-prompt sweep
python3 bench_robust.py --url http://HEAD:8000 --waves 4 --levels 1,2,3,4,5,6 --tag <name>

# eviction/refill classification
python3 probe_evict_refill.py --url http://HEAD:8000 --model <served-id> --n 400 --tokens 9000

# Local Inference Lab C1..C6 wrapper
./run_lil_c1c6.sh http://HEAD:8000 <served-id> <outdir> [rounds] [max-tokens]
```

`HEAD:8000` is a sanitization placeholder — substitute your own endpoint. The
endpoints in the source campaign were internal addresses; none are reproduced
here.

`run_lil_c1c6.sh` refuses to measure unless `/health` is 200 and the engine is
idle (`num_requests_running == 0`, and no capacity-waiting request). A request
parked in `reason="deferred"` holds no GPU slot and is not treated as a blocker —
that parked state is itself a separate defect documented in
[`../docs/REQUEST-CAP-LOCATION-20260911.md`](../docs/REQUEST-CAP-LOCATION-20260911.md).

## Measurement rules this campaign had to learn

Reproduced from [`../docs/CORRECTION-LOCKFIX-AND-DECODE-20260911.md`](../docs/CORRECTION-LOCKFIX-AND-DECODE-20260911.md) §5
and [`../docs/TEST-ITERATION-AND-PROFILING.md`](../docs/TEST-ITERATION-AND-PROFILING.md).

1. **Measure decode from `usage.completion_tokens`** with
   `stream_options: {"include_usage": true}`, or from
   `vllm:spec_decode_num_accepted_tokens_total` ÷ steps. **Never** count streamed
   SSE chunks: with DFlash2 speculative decoding each chunk carries several
   accepted tokens, and an early harness undercounted decode by ~5–6× that way.
2. **Report ms/step and acceptance, not raw tok/s**, when comparing builds.
   Throughput is `(acceptance × 7 + 1) / ms_per_step`, so raw tok/s across
   *different prompts* compares prompt difficulty, not engine speed — the same
   build measured 17–72 tok/s at a constant 83 ms/step. Hold the prompt set fixed.
3. **Stop the page-cache flusher** after `/health == 200`, before measuring.
   Leaving it running turned 41–44 tok/s into 13–17 tok/s.
4. **Assert the engine is idle** and check for leftover bench containers. A killed
   `ssh` does not stop the `docker run` it started; one such leftover was still
   issuing requests during a bench and poisoned the run.
5. **Self-limit every runner** (`timeout N docker run --rm --name ...`).
6. **Warm up** — discard ~5 requests. The first request after a boot pays JIT and
   measured `ttft = 108 s` once.
7. **One TP4 engine arm at a time** (it owns all four nodes). Fan out over the
   nodes only while the engine is **down**.
8. **Never trust a clean exit code as "no crash".** Every engine death in this
   campaign exited 0 with `OOMKilled=false`; only `dmesg`'s NVRM lines and
   `/health` going dead distinguish "died" from "fine".
9. **Failed and incomplete cells are disclosed, never retried into a mean or
   filled in.** At most two repetitions per cell were taken; treat the numbers as
   point measurements, not population estimates.
