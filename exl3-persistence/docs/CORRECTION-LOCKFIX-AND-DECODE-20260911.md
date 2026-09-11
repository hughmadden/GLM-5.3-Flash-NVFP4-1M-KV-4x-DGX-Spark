# CORRECTION & METHOD — why the "lock fix regression" was not real (2026-09-11 AEST)

You asked me to troubleshoot why the lock fix made decode worse. **It didn't.** The measurement was
wrong in three independent ways, and chasing it produced a bigger and more useful finding: **there is
no decode bug at all.**

---

## 1. The controlled A/B

Both arms measured with the *same* protocol: engine healthy and idle, flusher stopped, no leftover
containers, 5 warmup requests discarded, then 8 reps of 200 new tokens at ~8k context.

| build | median tok/s | p25–p75 | **ms/step** | accept | alloc_fail | store ms/op |
|---|---|---|---|---|---|---|
| lock fix reverted | **42.1** | 41.7–44.2 | **~83** | 0.35–0.61 | — | — |
| lock fix applied | **41.0** | 35.3–58.2 | **~83** | 0.19–0.75 | **0** | **247** |

**ms/step — the actual engine speed — is identical.** The lock fix neither helps nor hurts decode.

## 2. The three artifacts that made it look worse

**(a) A leftover benchmark container was competing for the engine.** A killed `ssh` does not stop the
`docker run` it started, so an earlier bench (`thirsty_davinci`) was still issuing requests. The tell
was `num_requests_running=1, num_requests_waiting=2` while nothing of mine was supposed to be running.
Fix: bench runners now carry `timeout N docker run --rm --name ...` so they self-terminate.

**(b) The page-cache flusher was still running.** the private deployment runbook §2.4 says stop it once `/health` returns
200; I had been leaving it up. It drops the cache every 60 s, so every persistence read goes to disk:

| with flusher running | with flusher stopped |
|---|---|
| 13–17 tok/s | **41–44 tok/s** |

**(c) Acceptance-length variance — the big one.** Throughput is not a measure of engine speed:

```
tok/s = (acceptance x 7 + 1) / ms_per_step        # 7 speculative tokens (DFlash2)
```

My bench used a different prompt per rep, so acceptance varied 0.19–0.75 and tok/s varied 17–72 **at a
constant 83 ms/step**. Comparing raw tok/s across runs with different prompts compares prompt
difficulty, not engine performance.

| acceptance | tokens/step | tok/s @ 83 ms/step |
|---|---|---|
| 0.19 | 2.33 | 28 |
| 0.42 | 3.94 | 47 |
| 0.75 | 6.25 | 75 |
| 0.959 | 7.71 | 93 |

## 3. Consequence: there is no decode bug

Mia's own published figures, decomposed the same way:

| source | tok/s | acceptance | tokens/step | implied ms/step |
|---|---|---|---|---|
| her "structured" regime | 65.1 | 0.959 | 7.71 | **118** |
| her "prose" regime | 27.1 | 0.341 | 3.39 | **125** |
| **ours** | ~42 | ~0.42 | 3.94 | **83** |

**Our per-step latency is better than the source's own measurements imply.** The tok/s gap against
"65 tok/s" is entirely the structured/high-acceptance prompt regime, not engine speed. The 222/318
tok/s numbers that started this whole line of enquiry are the **TrellisMX vendor's on 4× RTX PRO 6000
@300 W** — different silicon, different quant.

### This also corrects my earlier claims

`FIXES-CAP-AND-PERSISTENCE-20260911.md` reports decode going "12 → 58 tok/s" from the durability
fix. **That comparison was confounded in exactly the same three ways** and should not be quoted as a
4.8x end-to-end win. What is solid is the *isolated* measurement:

- lease-grant latency **8.697 ms → 0.024 ms** (unit harness, dummy data, no engine) — a real 360x;
- the engine metric `vllm:kv_offload_lookup_sync_delay` no longer dominating the step.

The durability fix is still correct and worth keeping — it removes an fsync from the per-step path —
but its end-to-end tok/s effect is masked by acceptance variance, and I overstated it.

## 4. Disposition of the lock fix: KEEP

- Decode: **neutral** (ms/step 83 both ways).
- Store path: **better** — `alloc_fail` 0 and 247 ms/op, against 99–1085 failures and 926–1033 ms/op
  in the confounded runs.
- It also removes the storage suite's D2 defect (a granted write reservation being tombstoned while
  queued behind the global lock) — the payload I/O no longer queues on that lock, and the reservation
  is re-validated under the lock before the index commit.

## 5. Benchmark rules this produced (apply to every future engine measurement)

1. **Stop the flusher** after `/health` == 200, before measuring anything.
2. **Assert the engine is idle**: `num_requests_running == 0` and `num_requests_waiting == 0`, and
   check for leftover bench containers (`docker ps`) — a killed ssh leaks a running `docker run`.
3. **Self-limit every bench runner** (`timeout N docker run --rm --name ...`).
4. **Never compare raw tok/s across different prompts.** Report **ms/step and acceptance**, and derive
   tok/s from them. Hold the prompt set fixed (same seeds) when comparing builds.
5. **Warm up properly** — 5 requests discarded. The first requests after a boot also pay JIT.
6. One 4-node TP4 engine arm at a time; the units fan out over the Sparks only while it is down.
