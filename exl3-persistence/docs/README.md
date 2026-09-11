# docs/ — findings and corrections, in the order they were established

These are the campaign's own working documents, sanitized and de-duplicated.
They are **evidence records, not a narrative**: they retain the measured numbers,
the root-cause derivations, the reproduction commands and the failed/unattempted
items. Where a later document corrects an earlier one, both are kept and the
correction is cross-referenced — history is not rewritten.

## Reading order

| # | Document | Why read it |
|---|---|---|
| 1 | [`BASE-DECISION.md`](BASE-DECISION.md) | The R0 deliverable: chosen base route, exact dependency pins, the hybrid-offloadability **kill-gate verdict** (PASS-WITH-CONSTRAINTS) and its four launch constraints, the 3,272-commit drift analysis of the persistence port, and the overlay port record. Start here for *why this stack*. |
| 2 | [`FINDINGS-FIRST-BOOT-20260910.md`](FINDINGS-FIRST-BOOT-20260910.md) | The first real boot. The five printed-recipe defects that had to be fixed and the one image rebuild; the measured first-boot numbers; and the disclosed MISS (observer geometry receipt) plus the S2-3/S2-4 boundary. |
| 3 | [`BENCH-20260910.md`](BENCH-20260910.md) | Ladder / needle / concurrency measurements, with the correction appended at the top: the decode column is undercounted ~5–6× because the harness counted SSE chunks. |
| 4 | [`CONCURRENCY-TESTS-20260910.md`](CONCURRENCY-TESTS-20260910.md) | Engine-free concurrency testing of the persistence interfaces: harness, 4-way fan-out, the 8 shipped fixes, the 15 open defects, and the disk-tier performance floor. |
| 5 | [`FIX-PLAN-20260911.md`](FIX-PLAN-20260911.md) | The full defect register with an **exact reproduction for every item**, confidence labels, ranked proposed fixes, and the engine-level analysis (request cap, decode). |
| 6 | [`REQUEST-CAP-LOCATION-20260911.md`](REQUEST-CAP-LOCATION-20260911.md) | The request-cap root cause: one named 512 MiB profile-time reservation in the sparse-indexer profiling branch, why lowering `MAX_MODEL_LEN` would not have helped, and the confirmation experiment. |
| 7 | [`FIXES-CAP-AND-PERSISTENCE-20260911.md`](FIXES-CAP-AND-PERSISTENCE-20260911.md) | Both fixes with reproduction: the UMA over-commitment (util 0.85 → 0.75) and the per-lease fsync (**8.697 ms → 0.024 ms**). Carries the withdrawn 4.8× decode claim inline, marked as confounded. |
| 8 | [`CORRECTION-LOCKFIX-AND-DECODE-20260911.md`](CORRECTION-LOCKFIX-AND-DECODE-20260911.md) | **Correction of record.** Why the "lock fix regression" was not real, the three independent measurement artifacts behind it, the controlled A/B (ms/step 83 both ways), the conclusion that there is no decode bug, and the benchmark rules that follow. |
| 9 | [`TEST-ITERATION-AND-PROFILING.md`](TEST-ITERATION-AND-PROFILING.md) | How to iterate and instrument faster: loop costs, crash-resilient experiment design, and the instrument list (1 Hz UMA sampler, `/metrics`, torch profiler, Nsight) with the concrete commands. |
| 10 | [`REFILL-HANG-ROOT-CAUSE-20260911.md`](REFILL-HANG-ROOT-CAUSE-20260911.md) | The disk tier's read path: **the store and the lookup are both correct** (112 of 112 chunks found in 2 s) and the transfer is never issued. Includes a hypothesis that a stub supported and the real source **refuted**, kept deliberately. |
| 11 | [`FIX-LOOKUP-BUDGET-20260911.md`](FIX-LOOKUP-BUDGET-20260911.md) | `PERSIST_LOOKUP_KEYS_PER_STEP=8` throttled nearly every real prompt into RETRY; vLLM discards the whole hit count on any RETRY, so requests hung. Controlled A/B: hung past 240 s at 8, completed in 7.6 s at 2048, with lookup churn down ten-fold. Includes the sizing rule. |
| 12 | [`FIX-LOOKUP-GATE-DEADLOCK-20260911.md`](FIX-LOOKUP-GATE-DEADLOCK-20260911.md) | The lookup gate's only rotation hook does not run for a request the gate itself deferred — a deadlock, observed live as `Running: 0` with the request parked. Fixed, with a test that fails on the reverted tree and passes on the fixed one. |

## The one-paragraph result, with its boundary

The deployment serves GLM-5.3-Flash EXL3 TR3 4bpw at TP4 across four GB10 nodes
with a rank-local disk KV tier active from first boot, at a cost of ≈0.4 % of the
GPU KV pool. The tier's decode penalty was one durable fsync per lease grant and
is fixed (isolated 8.697 ms → 0.024 ms per grant). **There is no decode bug**:
controlled measurement puts the engine at ~83 ms/step, better than the 118–125
ms/step implied by the upstream recipe's own published throughput figures.
Large-context requests were capped by two independent memory defects, both fixed
and both measured; the safe prefill bound above 255k tokens was not bisected, so
`MAX_MODEL_LEN` was left deliberately over-large and an over-large request can
still kill the engine.

**The disk tier's read path does not serve, and this is now characterised rather
than suspected.** The tier stores correctly and durably (7 GB, 2,302 objects,
surviving restarts), and a controlled test shows it locating an entire prefix —
112 of 112 chunks, within two seconds. The transfer that should follow is never
issued: zero load bytes over 240 s, the request deferred the whole time, and
36,137 lookup retries against a prefix that had already been fully found. The
store, the keying, the geometry, the eviction policy and the lease-validation path
are each excluded by measurement. No oversubscription test was run.

Disk hits **are** counted as cache hits at the accounting surface
(`vllm:kv_offload_tiering_chunk_{queries,hits}_total{tier="disk"}`, plus
`prompt_tokens_details.cached_tokens`); what stays at zero is vLLM's *served*
external-prefix-hit metric, because the transfer never lands.
