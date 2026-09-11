# GLM-5.3-Flash EXL3 4bpw at TP4 with rank-local disk KV persistence

## Executive summary

Four DGX Spark GB10 nodes serving **GLM-5.3-Flash EXL3 TR3 4bpw** at tensor parallelism 4, with a
node-local disk KV tier intended to let a long agent session be parked and resumed instead of
recomputed. Getting there produced two genuine defects, one wrong conclusion of our own, and one
finding that matters more than either fix.

**The request cap was a single named allocation, not a vague memory problem.** Prefills above roughly
83k tokens killed the engine on the original configuration. The cause was unified-memory
over-commitment, and the binding term was a 512 MiB reservation in the sparse-indexer profiling branch
governed by an environment variable, *not* by `max_model_len`. Lowering the GPU memory fraction from
0.85 to 0.75 and bounding that reservation restored host headroom from **884 MB to 16.4–18.2 GB** and
raised the largest surviving prompt from **~83k to 255k tokens** — **404,749 tokens** with the full
persistence stack enabled. Every one of those deaths exited **0** with `OOMKilled=false`.

**The persistence decode penalty was a durable commit on a cache.** The store committed a lease with
`synchronous=EXTRA`, on a path that runs every scheduler step. Because the store is a reconstructible
cache whose configured failure policy is `recompute`, that bought nothing and cost a full `fsync`:
**8.697 ms per lease grant, reduced to 0.024 ms** by making durability an explicit, validated policy
(defaults unchanged). WAL additionally cut the concurrency suite's wall time from 50 s to 12 s.

**We published a wrong number and corrected it.** An earlier note claimed this fix recovered decode
4.8×. It did not. The comparison arms differed in speculative-decoding acceptance, in a running
page-cache flusher, and in leftover benchmark containers. Controlled, the arms are **42.1 vs 41.0 tok/s
at ~83 ms/step — identical**. The rule this produced: **report ms/step and acceptance length, never
bare tok/s across differing prompts.**

**The finding that matters most: the tier writes but does not read, and the failure mode is a hang.**
7 GB of persisted objects survived several engine restarts, so the write path and durability are real.
But a controlled test that stored one ~5.3k-token prefix and immediately re-requested it saw the repeat
**never return**. The counters added in this work then showed exactly why: the tier **found** the
prefix (**76 chunk hits**) and transferred **zero bytes**, while the scheduler retried the lookup
**32,096 times** for that one request. The engine correctly reported
`External prefix cache hit rate: 0.0%` throughout — complete and accurate accounting, because the tier
was consulted and did not serve. Clients hang while the metric says only "low hit rate". This is
**open**; the two counters exist to separate "consulted and missed" from "found but deferred", and they
identified the latter.

**Measured throughput, Local Inference Lab standard C1–C6 bench, 63 requests, zero failures:** aggregate
output throughput saturates at **73–75 tok/s from concurrency 2 onward**; per-stream falls from 39.6 to
21.1 tok/s across C1–C6; `MAX_NUM_SEQS=4` caps real concurrency at 4, so C5 and C6 measure queueing, not
parallelism. Acceptance ranged 0.331–0.478 accepted-per-drafted at k=7, which is what actually moves the
headline number.

## What this release is

Four-node tensor-parallel (TP4) serving of **GLM-5.3-Flash EXL3 TR3 4bpw** across four
DGX Spark GB10 nodes, with a **rank-local disk KV persistence tier active from first
boot**. Engine lineage: the upstream vLLM native offloading ABI plus this campaign's
own `recipe_persistence` package and a four-patch companion series; EXL3 kernel
overlay and DFlash2 k=7 speculative decoding ported from the upstream EXL3 recipe.

**Status: research source release.** Nothing here is a prebuilt image, a one-command
installer or a production qualification. Every number below was measured on this
deployment and carries its source document; the boundaries of what was *not* measured
are stated in [Provenance and limits](#provenance-and-limits).

Sources for the campaign documents referenced throughout are under [`docs/`](docs/).

## Contents

| Path | What it is |
|---|---|
| [`persistence/`](persistence/) | `recipe_persistence` — rank-local disk KV persistence for the native vLLM offloading ABI (CPU-testable, no torch/vLLM import at module load). Includes its packaging, `LICENSE` and `NOTICE`. |
| [`tests/`](tests/) | The package's unit suite, the concurrency stub harness (`tests/conc/`, stdlib-only runner + instant dummy data), and `validate_kv_transfer_config.py`. |
| [`patches/`](patches/) | The vLLM overlay patch scripts, the three anchored build patches, the startup-observer call-site port, the four-patch persistence series (`patches/persistence-flash/`) and the offline test harness for all of it. See [patches/README.md](patches/README.md). |
| [`configs/`](configs/) | The launch env file and launcher, host-side memory-ritual/flusher scripts, the chat template, and the image build recipe. See [configs/build/README.md](configs/build/README.md). |
| [`benchmark/`](benchmark/) | Engine-side harnesses: C1..C6 decode sweep, statistically robust mixed-prompt sweep, eviction/refill probe, and the Local Inference Lab C1..C6 wrapper. See [benchmark/README.md](benchmark/README.md). |
| [`docs/`](docs/) | The campaign's findings and corrections, in the order they were established. |

## Measured results

### Request cap and its root cause

The deployment originally died on large single prefills: the GB10 **unified** memory
was over-committed and the NVIDIA driver failed the allocation
(`NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal`), after which
the engine shut down through vLLM's SIGTERM handler, **exited 0 with
`OOMKilled=false`**, and hung draining the client socket. Every death in this campaign
looked like a clean exit; the real signals are the NVRM lines in `dmesg` and `/health`
going dead while PID 1 stays alive. (`docs/FIXES-CAP-AND-PERSISTENCE-20260911.md`,
`docs/FIX-PLAN-20260911.md` §C1.)

Two independent causes were separated:

| Cause | Evidence | Fix |
|---|---|---|
| UMA over-commitment at `GPU_MEM_UTIL=0.85` | host `MemAvailable` fell to **884 MB** while the engine's own planner believed it had 18.26 GiB spare | `GPU_MEM_UTIL=0.75` in the launch env — a configuration value, no code change |
| A **512 MiB** profile-time reservation that binds *below* `max_model_len` | `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (default **512**) × 1 MiB vs a computed 103.8 MiB decode-logits term | set `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=96` |

The second item is the more interesting finding and is derived in
[`docs/REQUEST-CAP-LOCATION-20260911.md`](docs/REQUEST-CAP-LOCATION-20260911.md): the
reservation is `max(decode_logits_elems, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1 MiB)`
bytes taken on **first indexer use** (i.e. the first prefill), assigned to `_` and
therefore apparently free — but PyTorch's caching allocator keeps the block
*reserved*. Corollary: lowering `MAX_MODEL_LEN` would **not** have shrunk it; the
buffer only becomes the binding term above `max_model_len ≈ 4.19 M`.

| Measurement | Before | After |
|---|---:|---:|
| `GPU_MEM_UTIL` | 0.85 | 0.75 |
| KV pool | 55.77 GiB | 43.2–43.9 GiB |
| KV tokens | 4,333,333 | 3,398,717–3,441,025 |
| host `MemAvailable` | 884 MB | 16,400–18,200 MB |
| Largest surviving single prompt | ~83k tokens (died at 90k) | **255,393 tokens survived** |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | 512 (default) | 96 |
| host `MemAvailable` at 0.85 with cap 96 | — | 2,866 MB (vs 884 MB) |
| KV pool at 0.85 with cap 96 | — | 56.74 GiB (vs 55.77 GiB) |

Prefill ladder with `PERSISTENCE=off` at util 0.75: **101,324 / 153,527 / 255,393
tokens all survived** (52 s, 80 s, 68 s). The crash is **not** caused by persistence:
an equivalent ladder with persistence off also ended in an NVRM OOM.

**Still open at the time of writing:** `MAX_MODEL_LEN` remained 850,000 while the
measured safe prefill bound was ≥255k but un-bisected above it, so a request between
the measured bound and 850k can still kill the engine. The intended end state is
`MAX_MODEL_LEN` set to the measured safe bound so an over-large request is refused
with a clean 400.

### Persistence cost and the durability fix

Measured at first real boot (`docs/FINDINGS-FIRST-BOOT-20260910.md` §4):

| Quantity | Value |
|---|---:|
| GPU KV cache, `PERSISTENCE=on` | **4,388,461 tokens** (56.18 GiB) |
| GPU KV cache, `PERSISTENCE=off` | 4,406,410 tokens |
| Persistence cost to the GPU pool | **≈0.4 %** |
| KV on disk, per rank (symmetric across all four) | 2,250,340,608 B (2.10 GiB) |
| `kv_offload_store_bytes_total` over 392 store operations | 9,029,116,928 B |
| `/health = 200` after boot | 390 s |

The decode penalty that the tier initially imposed was one durable fsync per lease
grant: `DiskStore.reserve_read` committed the lease `INSERT` under
`PRAGMA synchronous=EXTRA`, and `_Manager.on_schedule_end()` runs every scheduler
step. Isolated on the engine-free unit harness (dummy data):

| SQLite policy | Lease-grant latency |
|---|---:|
| `DELETE` + `EXTRA` (original) | 8.697 ms |
| `DELETE` + `NORMAL` | 4.498 ms |
| `DELETE` + `OFF` | 0.054 ms |
| **`WAL` + `NORMAL` (chosen)** | **0.024 ms** |

**≈360× faster**, and WAL additionally removes reader/writer serialisation (the
concurrency suite went from 50 s to 12 s wall). Durability became an explicit,
validated policy — `DiskStore(..., journal_mode=..., synchronous=...)` with
allowlists, **defaults unchanged** (`delete` + `extra`) — rendered from
`PERSIST_SQLITE_JOURNAL_MODE` / `PERSIST_SQLITE_SYNCHRONOUS`. The store is a
reconstructible cache (`kv_load_failure_policy=recompute`), so `EXTRA` bought no real
guarantee while costing an fsync per commit. Deliberate fail-closed latches in
`invalidate()`/`collect()`/`evict()` were **kept**; `clear_failure()` was added as the
operator recovery path.

### Decode throughput — and the correction of record

**Read [`docs/CORRECTION-LOCKFIX-AND-DECODE-20260911.md`](docs/CORRECTION-LOCKFIX-AND-DECODE-20260911.md)
before quoting any decode number from this campaign.** An earlier report of a 4.8×
end-to-end decode win from the durability fix was **confounded in three independent
ways** and is withdrawn:

1. a leftover benchmark container was still issuing requests (`num_requests_running=1`
   while nothing of ours should have been running);
2. the page-cache flusher was still running (13–17 tok/s with it up, **41–44 tok/s**
   with it stopped);
3. acceptance-length variance — throughput is `(acceptance × 7 + 1) / ms_per_step`, so
   comparing raw tok/s across *different prompts* compares prompt difficulty, not
   engine speed.

The controlled A/B (same protocol: engine healthy and idle, flusher stopped, 5 warmups
discarded, 8 reps of 200 new tokens at ~8k context):

| Build | Median tok/s | p25–p75 | **ms/step** | accept | store ms/op |
|---|---:|---|---:|---|---:|
| lock fix reverted | 42.1 | 41.7–44.2 | **~83** | 0.35–0.61 | — |
| lock fix applied | 41.0 | 35.3–58.2 | **~83** | 0.19–0.75 | 247 |

**ms/step — the actual engine speed — is identical.** There is no decode bug, and the
lock fix neither helps nor hurts decode. The durability fix is kept because the store
path is better (`alloc_fail` 0, 247 ms/op) and because it removes the fsync from the
per-step path, not because of an end-to-end throughput claim.

Decomposed the same way against the upstream recipe's own published figures
(`docs/CORRECTION-LOCKFIX-AND-DECODE-20260911.md` §3):

| Source | tok/s | acceptance | tokens/step | implied ms/step |
|---|---:|---:|---:|---:|
| upstream "structured" regime | 65.1 | 0.959 | 7.71 | **118** |
| upstream "prose" regime | 27.1 | 0.341 | 3.39 | **125** |
| **this deployment** | ~42 | ~0.42 | 3.94 | **83** |

Single-request decode is 7.2–12.0 tok/s **as counted by streamed SSE chunks**, which
undercounts by ~5–6× because DFlash2 emits several accepted tokens per chunk. Measured
directly against `usage.completion_tokens`:

| context | `completion_tokens` (usage) | SSE chunks | tokens/chunk | decode by usage | as first reported |
|---|---:|---:|---:|---:|---:|
| 7.5k | 200 | 31 | 6.45 | **27.5 tok/s** | 4.3 |
| 30.8k | 200 | 39 | 5.13 | **13.3 tok/s** | 2.6 |

**Always measure decode from `usage.completion_tokens`** (`stream_options:
{"include_usage": true}`) or from `vllm:spec_decode_num_accepted_tokens_total` ÷ steps,
never from the count of streamed chunks.

### Ladder, prefill and needle (C=1)

`docs/BENCH-20260910.md` §1. Decode column as originally reported — see the ÷5–6
correction above.

| test type | ctx target | ctx tokens | C | TTFT s | prefill tok/s | decode tok/s (as reported) | needle |
|---|---:|---:|---:|---:|---:|---:|---|
| speed | 4,096 | 4,029 | 1 | 2.30 | 1,754 | 11.6 | — |
| needle @92 % | 4,096 | 4,088 | 1 | 2.26 | 1,809 | 12.0 | **HIT** |
| needle @50 % | 4,096 | 4,093 | 1 | 2.30 | 1,781 | 10.4 | **HIT** |
| needle @50 % | 16,384 | 16,383 | 1 | 12.46 | 1,315 | 10.1 | **HIT** |
| needle @92 % | 16,384 | 16,394 | 1 | 8.22 | 1,994 | 9.9 | **HIT** |
| speed | 16,384 | 16,416 | 1 | 8.17 | 2,009 | 8.7 | — |
| needle @50 % | 32,768 | 32,731 | 1 | 17.26 | 1,896 | 8.7 | MISS |
| needle @92 % | 32,768 | 32,755 | 1 | 16.12 | 2,032 | 7.2 | **HIT** |
| speed | 32,768 | 32,825 | 1 | 21.68 | 1,514 | 9.8 | — |
| speed | 65,536 | 65,711 | 1 | 53.04 | 1,239 | 9.7 | — |

TTFT scales close to linearly with context (2.3 s @4k → 8.2 s @16k → ~16–22 s @32k →
53 s @65k). Prefill holds 1,750–2,050 tok/s to 32k, then falls to 1,239 tok/s at 65k.
Needle: **5 of 6 probes HIT**; the single MISS was 32k at mid-depth.

### Concurrency — Local Inference Lab standard C1–C6 bench

This supersedes the earlier concurrency sweep in `docs/BENCH-20260910.md` §2. Those
numbers counted streamed SSE chunks rather than `usage.completion_tokens` and read
roughly 5–6× low; they are retained in `docs/` as the record of the error, not as a
result. The bench below is the standard one: three waves of `c` parallel requests per
level, 700 output tokens each, code/reasoning prompts salted uniquely per request so
prefix caching cannot flatter the numbers, acceptance scraped from `/metrics`.

Verbatim output, 63 requests, **zero failures**:

```
 c reqs  agg_tok/s per_stream  mean_s fails  accept_len
 1    3       39.4       39.6    17.7     0  ratio=0.331
 2    6       73.3       38.2    19.1     0  ratio=0.478
 3    9       73.8       30.9    28.5     0  ratio=0.411
 4   12       73.0       22.9    38.3     0  ratio=0.379
 5   15       72.0       22.5    48.6     0  ratio=0.435
 6   18       75.1       21.1    55.9     0  ratio=0.395
```

| C | aggregate tok/s | per-stream tok/s | mean wall s | acceptance | derived ms/step |
|---:|---:|---:|---:|---:|---:|
| 1 | 39.4 | **39.6** | 17.7 | 0.331 | ~84 |
| 2 | 73.3 | 38.2 | 19.1 | 0.478 | ~119 |
| 3 | 73.8 | 30.9 | 28.5 | 0.411 | ~158 |
| 4 | 73.0 | 22.9 | 38.3 | 0.379 | ~200 |
| 5 | 72.0 | 22.5 | 48.6 | 0.435 | queue-limited |
| 6 | **75.1** | 21.1 | 55.9 | 0.395 | queue-limited |

Three readings:

- **Aggregate saturates at ~73–75 tok/s from C=2 onward.** A batch of two already reaches
  the ceiling; C=6 adds nothing.
- **Per-stream halves** from 39.6 to 21.1 tok/s across C1–C6. Concurrency buys latency
  headroom for other clients, not throughput for any one of them — the same conclusion as
  the earlier sweep, now with correct magnitudes.
- **C5 and C6 are not real parallelism.** An occupancy sampler recorded a hard maximum of
  4 concurrently running requests throughout, because `MAX_NUM_SEQS=4` caps the batch.
  The flat aggregate confirms it. ms/step is therefore only derived where the running
  batch equals the requested concurrency.

Measurement conditions: the run took 10 min 27 s. The endpoint is gateway-fronted, so
both an occupancy sampler and the gateway's request log were recorded. Over the
60-minute window ending with the run the gateway logged 23 requests to this model and
**zero fell inside the measurement window**, so these numbers are uncontended. Of 208
occupancy samples, 100% had at least one request running and 77% had more than one, mean
2.64 — all of it this bench.

### Eviction and refill: the tier writes but does not read

The disk tier's own quota is 1 TB (watermarks 900/800 GB), so its eviction path is not
reachable at test scale. What is reachable, and far more important, is the GPU side:
force the GPU prefix cache to release a session and then ask for it again.

`benchmark/probe_evict_refill.py` sends one unique ~5.3k-token prompt cold, then
immediately re-sends it. That repeat is the simplest possible expression of the tier's
whole purpose.

```
=== phase 0: cold prefill of the target prompt ===
  cold: 2.84 s  prompt_tokens=5279  prefix_hits=0  store_B=182949888
=== phase 1: immediate repeat (should be served from a cache) ===
```

**Phase 1 never returned.** The request entered the deferred state and stayed there.

| Signal at capture | Value |
|---|---|
| bytes accepted by the store for the phase-0 prompt | **182,949,888 (~183 MB) — the write succeeded** |
| `num_requests_running` | 0 |
| `num_requests_waiting_by_reason{reason="capacity"}` | 0 |
| `num_requests_waiting_by_reason{reason="deferred"}` | **2** |
| `vllm:kv_offload_allocation_failure_total` | **4,034 and climbing** |
| `External prefix cache hit rate` | **0.0%** |

The deferred state is the one to understand: the request is not waiting for a GPU slot
(capacity-waiting is 0, nothing is running). It waits for a KV transfer that is never
issued. No error is logged, nothing is returned to the client, and nothing times out. It
simply never completes. The probe's later flood phase was abandoned rather than run,
because each deferred request holds its worker for the full client timeout.

**Verdict, from the counters described below.** After deploying the two disk-tier counters to all four
ranks and restarting, the experiment was run against a fresh engine with one unique 8,171-token prompt,
sampling the engine's counters every 2 s for 240 s:

| pass | wall time | disk chunk queries | disk chunk hits | store | load bytes | external hits | deferred |
|---|---|---:|---:|---|---:|---:|---:|
| 1 — cold | 4.6 s | 1 | 0 | 252,797,952 B in **112 stores** | 0 | 0 | 0 |
| 2 — immediate repeat | **hung** (240 s client timeout) | **36,137** | **112** | 0 | **0** | **0** | **1** |

The counts 112 and 112 are the whole finding. Pass 1 stored the prompt as **112 chunks**; pass 2 found
**112 chunks**. The prefix is not partially cached, not mis-keyed and not slow — **the tier holds the
entire prefix and returns all of it, within two seconds.** Sampling shows the hit count rise 76 → 112
and then stop, because there is nothing left to find:

```
t=  0.0s  queries=  34916  hits= 76  load_B=0  deferred=0  running=0
t=  2.0s  queries=  35708  hits=112  load_B=0  deferred=1  running=0
t= 74.4s  queries=  66820  hits=112  load_B=0  deferred=1  running=0
t= 76.4s  queries=  67580  hits=112  load_B=0  deferred=1  running=0   <- query rate collapses
t=239.0s  queries= 71028  hits=112  load_B=0  deferred=1  running=0
RESULT: 240.1s -> TimeoutError: timed out
```

Five conclusions, and they are the whole diagnosis:

1. **The store and the lookup are both correct.** 112 of 112 chunks matched. This excludes the store,
   the keying, the geometry and the eviction policy, and it excludes the "consulted and missed" reading
   of the 0.0% hit rate.
2. **The transfer is never issued.** Zero load bytes for the full 240 seconds. The prefix is located and
   then discarded.
3. **The request is deferred, not failing.** `deferred = 1` continuously from t=2 s to t=240 s. Nothing
   errors server-side, nothing returns, nothing times out server-side. Only the client times out.
4. **The engine burns CPU retrying.** Lookups climb at roughly 880/s for the first 76 s, then collapse to
   about 4/s — a retry loop eventually throttled but never resolved. 36,137 lookups for a prefix fully
   located at t=2 s. `external_prefix_cache_queries` advanced by 11,280 while
   `external_prefix_cache_hits` stayed at 0 — correct accounting, because vLLM counts a hit only when a
   transfer actually supplies tokens.
5. **The defect is downstream of the lookup**: between a resolved hit and the transfer that should
   follow it. That is a far narrower statement than "the restore path is broken", and it came from
   adding the counters rather than from reading more code.

### A hypothesis tested and rejected

Worth recording, because a rejection is worth as much as a conclusion. vLLM's scan discards the entire
hit count if any chunk returns RETRY, and our lease-validity check returns `False` — without queueing a
renewal — when the coordinator reports no deadline. If a freshly granted lease had no deadline, every
reserved chunk would RETRY until a renewal landed, and the scan could never converge.

A stub modelling that coordinator reproduced the observed shape exactly: **673 scheduler steps to
converge instead of 10**, one renewal per chunk; with the deadline seeded at reservation, back to 10
steps with zero renewals.

**The hypothesis is wrong**, and reading the real coordinator rather than the model of it is what showed
that: `coordinator_http.py` already sets the lease deadline at grant time, so a fresh reservation is
immediately valid for its full window. **No change was made to the lease-validation, deadline or renewal
code on the strength of that stub.** This is the third hypothesis in this campaign that a stub could
support and the real source refuted, which is the argument for keeping both.

### Cache-hit accounting: already correct, but invisible and genuinely missing

An initial reading of this deployment concluded that a disk hit was not being counted as
a cache hit. **That was wrong.** vLLM already separates the two sources and already routes
offload-tier hits into their own accounting:

```
Engine 000: ... Prefix cache hit rate: 86.6%, External prefix cache hit rate: 0.0%
```

`Prefix cache hit rate` is the GPU-resident block cache; `External prefix cache hit rate`
is the offload tier, fed by a chain that already exists (tier lookup hit →
`get_num_new_matched_tokens` → `num_external_computed_tokens` →
`vllm:external_prefix_cache_hits_total`). A disk block is not, and should not be, part of
`vllm:prefix_cache_hits_total`, which is GPU-only by design. So a printed `0.0%` is a
**measurement**, not a gap: it proves the tier was consulted and that no query resolved as
a hit — independent confirmation of the defect above.

Two real gaps remain, and both are addressed here:

1. **The client could not see cache reuse at all.** API responses carried no
   `usage.prompt_tokens_details`, because `--enable-prompt-tokens-details` defaults to
   `false`. The underlying `num_cached_tokens` is already local *plus* external, so a
   disk-served prefix would have appeared in `cached_tokens`; it was simply never
   serialised. `configs/launch/start-tp4.sh` now enables the flag, and the field is confirmed on the
   wire: `"prompt_tokens_details": {"cached_tokens": 0, "created_cache_tokens": 0, ...}`.
2. **An operator could not distinguish "consulted and missed" from "found but deferred".**
   With the rate pinned at zero the two are indistinguishable, and they call for opposite
   responses. `persistence/recipe_persistence/native.py` now emits
   `vllm:kv_offload_tiering_chunk_queries_total{tier="disk"}` and
   `vllm:kv_offload_tiering_chunk_hits_total{tier="disk"}`, registered through vLLM's sanctioned
   `build_metric_definitions()` extension point, **confirmed live at `/metrics`** after a fleet
   redeploy:

   | queries | hits | reading |
   |---:|---:|---|
   | 0 | 0 | the connector never reaches the tier — investigate the lookup path |
   | >0 | 0 | the tier is consulted and does not hold/validate the prefix — investigate keying, geometry, eviction |
   | >0 | >0, external hits still 0 | the tier holds it and the lookup returns retry — the stuck deferral; investigate staging/admission |

   These are increments only. They change no hit, miss, admission or eviction decision.

### Concurrency unit tests (engine-free)

`docs/CONCURRENCY-TESTS-20260910.md`. The persistence interfaces were exercised at
every major boundary with stubbed instant data, zero engine and zero inference, then
driven down through a fix/re-run cycle. Resource allocation was 4 CPU / 8 GB per
tester on each of the four nodes.

| Run | Result |
|---|---|
| Concurrency suite, before fixes | 184 tests, 164 pass, 20 fail, 0 hang |
| Concurrency suite, **after 8 fixes** | **184 tests, 167 pass, 17 fail, 0 hang** (189 s) |
| Package's own 99 `unittest` tests, after fixes | **OK — 99/99** (210 s, no regressions) |

Load cost: **0.07 s** for all seven package modules, **0.70 s** including `docker run`.
Of the 17 remaining failures, **15 are genuine open defects** and 2 are stale pins of a
harness-stub bug that was fixed. Eight fixes shipped, each verified by re-running the
reporting tester's own test: `_CloseOnce` re-entrant hang, `wait`/`_poll` iteration
snapshot, distinguishable non-latching storage-failure record, idempotent
`native.shutdown()`, `lease_valid`/`read_into` consistency, `usage()` after close,
post-close maintenance latch, `clear_failure()`, plus the stub-shape fix.

The highest-value open defects (full register in
[`docs/CONCURRENCY-TESTS-20260910.md`](docs/CONCURRENCY-TESTS-20260910.md) §3 and
[`docs/FIX-PLAN-20260911.md`](docs/FIX-PLAN-20260911.md) Part B) are the RPC admission
bound rejecting healthy callers instead of queueing (O1), that rejection becoming
spurious all-rank cache misses that silently lose cache writes (O2), and
`RemoteCoordinator._reserve` holding the global lock across network I/O (O3).

Disk-tier performance, stated plainly: `DiskStore` does **not** scale — 16–27 write
ops/s and 19–37 read ops/s, flat from C=1 to C=8, with a fixed ~64–68 ms per-op cost
(fsync-dominated) and a global lock that makes C=8 *slower* than C=1 at 1 MB.
`collect(force=True)` costs 19.7 ms per tombstone under that lock.
**The disk tier's real throughput is tens of ops/s per rank.** One object and one fsync
per logical block favours reviewable correctness, not a throughput optimum.

### First boot — defects found and fixed

`docs/FINDINGS-FIRST-BOOT-20260910.md`. The printed recipe could not execute as
written; five defects were fixed and the image rebuilt once.

| # | Defect | Resolution |
|---|---|---|
| 1.1 | Image Id does not survive `docker save` → `docker load` (containerd snapshotter store vs classic store): 34.4 GB vs 23.2 GB reported, different config digest, **identical 55 `RootFS.Layers` digests** | pin the **loaded** Id after `docker load`; all four ranks agreed bit-for-bit |
| 1.2 | `$HOME` mounts single-quoted into a remote shell: docker received a literal `$HOME` and refused; a junk tree was created | removed six single-quote pairs so the remote shell expands the path |
| 1.3 | `--limit-mm-per-prompt` must be a JSON dict on this base (`LIMIT_MM=100` → pydantic `dict_type` error) | `LIMIT_MM='{"image":100}'` |
| 1.4 | `preflight` could not `stat` the persistence token as the unprivileged user (directory is `0700 root:root`) | the four metadata commands now run under `sudo`; **the token value is still never read, echoed or logged** |
| 1.5 | Persistence spec construction had no ambient vLLM config (`ValueError: persistence requires an active vLLM cache configuration`) — the scheduler path constructs the spec inside `EngineCoreProc.__init__`, which never calls `set_current_vllm_config` | new anchored, hash-pinned build patch [`patches/vllm/patch_offloading_ambient_config.py`](patches/vllm/patch_offloading_ambient_config.py) wraps spec construction in `with set_current_vllm_config(vllm_config)`; fail-closed on BEFORE/AFTER hashes and anchors |

Also recorded as **MISS, not backfilled**: the S2-1 observer geometry receipt criterion
fails — only rank 0 writes a receipt, and the manifest itself declares
`engine_projection`, `worker_preallocation` and `worker_postbind` as
`known_incomplete_stages`. This is a diagnostic-subsystem gap, not a serving or
persistence fault.

**S2-3 is PARTIAL and S2-4 was not run.** The storage direction is proven; the
*load-back* direction is **not** demonstrated. This build exposes no disk-tier load/hit
counter and the head log's `Prefix cache hit rate` read 0.0 % across the round-trip. The
second identical 8k prompt returned coherent output, but that is equally consistent with
the prefix still being resident in the GPU pool. The test that would settle it (restart
retention: `down`, `up`, repeat the seed, look for the post-restart request hitting
stored KV) requires taking the handed-over service down and was not run.

## Configuration reference

`configs/launch/env.tp4.fleet` is the single source of truth for the launch; the
launcher renders per-rank argv from it. Addresses and the coordinator port in the
published copy are placeholders — see [Sanitization](#sanitization-of-this-release).

### Serving contract

| Setting | Value |
|---|---|
| `MODEL_REPO` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (EXL3 / TR3 4bpw) |
| `DFLASH_REPO` | `incoai/GLM-5.3-Flash-DFlash2` |
| `SERVED_MODEL_NAME` | `glm-5.3-flash-exl3` |
| `TP` / `NNODES` | 4 / 4 |
| `QUANTIZATION` | `exl3` |
| `KV_CACHE_DTYPE` | `fp8` |
| `MAX_MODEL_LEN` | 850000 |
| `GPU_MEM_UTIL` | **0.75** (post cap fix; 0.85 was the failing value) |
| `MAX_NUM_SEQS` | 4 |
| `MAX_NUM_BATCHED_TOKENS` | 7168 |
| `ENFORCE_EAGER` | 1 (CUDA graphs disabled; the upstream recipe's published numbers are with graphs) |
| `LIMIT_MM` | `{"image":100}` |
| `PREFIX_MATCH_UNIT` | **4 (mandatory)** |
| `PORT` / `MASTER_PORT` | 8000 / 29521 |
| `SPEC_METHOD` / `DFLASH_TOKENS` | `dflash` / 7 |
| EXL3 knobs | `EXL3_FAT_GROUPED=1`, `EXL3_TEMP_ROWS_FUSED=32`, `GLM53_ADAPTIVE_K=off`, `GLM53_DENSE_FP8=off` |
| `GLM53_INDEXER_WORKSPACE` / `GLM53_SPINWAIT_MS` | `rightsize` / 16 |

`PREFIX_MATCH_UNIT=4` is mandatory, not tuning: with the sparse kpool indexer active
`KpoolTailSpec.block_size == index_kpool` (4), and `offloading/config.py` asserts every
group's `tokens_per_block % tokens_per_hash == 0`; the default `tokens_per_hash` is the
gcd of prefix-cacheable group sizes (the MLA block, ≥128), so `4 % 128` fails and the
engine dies at init.

### Persistence tier

| Setting | Value |
|---|---|
| `PERSISTENCE` | `on` (renders `--kv-transfer-config` for `recipe_persistence.native`); `off` runs exactly as the upstream recipe ships |
| `PERSIST_COORDINATOR_PORT` | placeholder in the published copy |
| `PERSIST_TENANT` | `glm53-flash-exl3-tp4` |
| `PERSIST_TOKEN_FILE` / `_HOST` | `/run/secrets/persistence-token` / `/run/glm53/persistence-token` |
| `PERSIST_STAGING_BYTES` / `_ROWS` | 4294967296 / 64 |
| `PERSIST_DISK_IO_THREADS` | 4 |
| `PERSIST_MAX_PENDING_KEYS` / `_BYTES` | 32768 / 64000000000 |
| `PERSIST_LOOKUP_KEYS_PER_STEP` | 8 |
| `PERSIST_SQLITE_JOURNAL_MODE` / `_SYNCHRONOUS` | `wal` / `normal` |
| `PERSIST_LIMIT_QUOTA` / `HIGH` / `LOW` | 1000000000000 / 900000000000 / 800000000000 |
| `PERSIST_LIMIT_INDEX_BYTES` / `MAX_OBJECTS` / `MAX_OBJECT_BYTES` | 67108864 / 1000000 / 67108864 |
| `PERSIST_LIMIT_FREE_BYTES` / `FREE_INODES` | 1073741824 / 128 |
| `PERSIST_LIMIT_LEASE_SECONDS` / `GRACE_SECONDS` / `IO_CHUNK_BYTES` | 300 / 60 / 1048576 |
| `PERSIST_MIN_FREE_BYTES` | 1288490188800 (1,200 GiB) |
| `PERSIST_COORDINATOR_STARTUP_TIMEOUT` | 1800 s (the **only** value that differs from the package default; the code default is 60 s, which would fail closed on a healthy fleet because the census runs after a ~175 GiB weight load) |
| `PERSIST_COORDINATOR_RPC_TIMEOUT` / `SERVER_THREADS` / `CLIENT_POOL` | 10 / 16 / 4 |
| `PERSIST_COORDINATOR_RENEW_MARGIN` | 30 (derived: `max(1, min(30, 0.1*lease_seconds))`) |

The persistence roots are per-rank, on that node's local NVMe-backed filesystem, never
NFS and never the weights share: `DiskStore` takes an **exclusive** lock on
`disk_root/rank-<N>` and requires `0 < low < high <= quota` with `quota >
2*index_bytes`. `preflight` creates a missing root `0700`, verifies the backing
filesystem is a local block filesystem via `findmnt -T` (only
`ext4|xfs|btrfs|f2fs|ext3` accepted), and requires free space ≥ `PERSIST_MIN_FREE_BYTES`.

### Host-side scripts

`configs/host/` holds the host-side scripts that run **outside** any container, before
the launcher: `mem-ritual.sh` and the **unconditional** page-cache flusher. The flusher
must stay unconditional — a threshold-triggered flusher can sit below its threshold and
still leave the NVRM allocator short, which shows up as the same command booting or
OOMing depending on the moment.

Order on each node: `mem-ritual.sh` → flusher (`nohup`, whole boot window) → on rank 0
only, `start-tp4.sh up` → once `/health` is 200, stop the flusher. Neither script is a
floor guard: neither watches a threshold, decides anything, or kills a process. **The
campaign deliberately has no floor guards or kill machinery of its own** — only
recipe-native protections. The launchers refuse an existing container with their own
name but cannot detect arbitrary other GPU owners.

### Build and dependency pins

Full detail and the honest "what was never verified" list is in
[`configs/build/README.md`](configs/build/README.md) and
[`docs/BASE-DECISION.md`](docs/BASE-DECISION.md).

| Component | Pin |
|---|---|
| vLLM | upstream main `83252ea899c6538eaa0c1fb31f28a92c661bbffc`; wheel `vllm-0.28.1rc1.dev617+g83252ea89-cp38-abi3-manylinux_2_28_aarch64.whl` |
| torch | `2.13.0+cu130`, cp312 aarch64 |
| FlashInfer | `0.6.18.post1` (python/cubin/jit-cache+cu130) |
| exllamav3 | `0.0.43` @ `c5d9c657966ffeeaa9353f0cc899f18629da4a13` |
| CUDA base | `nvidia/cuda:13.0.3-devel-ubuntu24.04` @ `sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266` |
| Persistence patch series pinned upstream | `vllm@83252ea899c6538eaa0c1fb31f28a92c661bbffc` |
| Native offloading ABI ported from | `ab666069935c1f23e8ef56038b4659ac9e8f19f8` |
| EXL3 overlay port | 15/15 vLLM-targeting scripts applied at the pin; 16 files touched; 3,847-line patch round-trips byte-exact |

Known drift at the pin: base `ab6660699` is **3,272 commits** behind `83252ea89`. Of
the original four-patch series (50 hunks / 8 files), **4 hunks were absorbed, 44 were
still needed, 2 were obsolete**. The overlay port re-anchored 7 scripts and needed 2
real reworks (`patch_adaptive_k.py`, `patch_dense_fp8.py`).

## Provenance and limits

**This is a source-and-configuration research release**, not a verified deployment.
Read [`docs/BASE-DECISION.md`](docs/BASE-DECISION.md) §3 for the kill-gate verdict
(**PASS-WITH-CONSTRAINTS**) and its constraints, and
[`docs/FINDINGS-FIRST-BOOT-20260910.md`](docs/FINDINGS-FIRST-BOOT-20260910.md) §3–4 for
what passed and what is a disclosed MISS.

Not established by anything in this release:

- **No load-back proof.** The disk tier's store direction is proven; restore/hit
  accounting is not (see above). No restart-retention test was run.
- **No fault-path qualification.** Terminal 401 / sticky 410, lease expiry,
  partial-admission finished-store frontier and drain-before-close were not exercised
  on the engine. All four nodes were healthy throughout.
- **No oversubscription test across the measured KV boundary** — the stated point of
  the exercise. Nor was the S2-7 A/B matrix run.
- **No CUDA graphs.** `ENFORCE_EAGER=1`; the upstream recipe's published decode figures
  are with CUDA graphs. An `ENFORCE_EAGER=0` arm was proposed and not run.
- **No population statistics.** Two repetitions per cell in the benchmark tables; these
  are point measurements.
- **No encryption at rest.** The disk tier's file modes are access control only.
- **No concurrency ceiling was found**, and throughput does not scale with concurrency
  (see above). Do not read the persistence tier as a throughput feature at short
  context.

Every engine death in this campaign exited **0** with `OOMKilled=false`. Only `dmesg`'s
NVRM lines and `/health` distinguish "died" from "fine".

## Footnote: benchmark rules this campaign produced

Apply these to any future engine measurement on this class of hardware
(`docs/CORRECTION-LOCKFIX-AND-DECODE-20260911.md` §5, `docs/TEST-ITERATION-AND-PROFILING.md`):

1. **Stop the flusher** after `/health == 200`, before measuring anything.
2. **Assert the engine is idle** — `num_requests_running == 0` and
   `num_requests_waiting == 0` — and check for leftover bench containers; a killed
   `ssh` does not stop the `docker run` it started.
3. **Self-limit every bench runner** (`timeout N docker run --rm --name ...`).
4. **Never compare raw tok/s across different prompts.** Report **ms/step and
   acceptance** and derive tok/s; hold the prompt set fixed when comparing builds.
5. **Warm up properly** — discard ~5 requests; the first request after a boot also pays
   JIT.
6. **One 4-node TP4 engine arm at a time**; fan out over the nodes only while it is down.
7. **Never trust a clean exit code as "no crash".**

## Sanitization of this release

This tree is a **sanitized** copy of a private campaign directory. Host names were
replaced with role names (`node0`…`node3`, `head`, `build-host`), internal addresses
with RFC 5737 documentation addresses (`192.0.2.0/24`, `198.51.100.0/24`), internal
ports with neutral ports, absolute home paths with `/home/user`, secret-store
references with placeholders, and private receipt/report paths with generic
placeholders (`$STAGE`, `$WORK`, `<RECEIPTS>`). **No credential values were ever
present in the published files** — only environment-variable names and file paths.

Two things to know when reading the configs:

- **You must supply your own addresses, ports and secret store.** The published
  `env.tp4.fleet` uses documentation addresses and a neutral coordinator port; the
  token file must be provisioned by you (the private provisioning utility is not
  distributed).
- **The observer manifest digest was re-pinned.** The published
  `patches/observer/observer-callsites-83252ea89.manifest.json` differs from the private
  one by one sanitized metadata field, so `OBSERVER_MANIFEST_SHA256` in the launch env
  was recomputed to match. The image bakes the same file, so a rebuild stays
  self-consistent.

The four numbered patches in `patches/persistence-flash/` are published **byte-for-byte
verbatim** (their pinned upstream is `vllm@83252ea89`); the overlay scripts under
`patches/overlay/` are likewise verbatim. Every launcher and environment-file change in
this release is a functional change, not a cosmetic one: the comments were genericised
but no argument, default or path expression was altered, and this is checkable by
diffing the non-comment lines.

Three functional changes were made **after** the first snapshot of this release and are
included here, so a reader diffing against an earlier copy should expect them:

1. `benchmark/bench_c1c6.py` gained a `--model` flag (default unchanged) and now prints
   the first request-failure reason. The second change matters: an early run reported
   0.0 tok/s for all 63 requests and still exited 0, because a wrong model id produced
   HTTP 404 into a handler that counted failures silently.
2. `persistence/recipe_persistence/native.py` gained the two disk-tier counters described
   under *Cache-hit accounting* above, plus `tests/test_disk_tier_metrics.py`.
3. `configs/launch/start-tp4.sh` gained `--enable-prompt-tokens-details`, so that
   `cached_tokens` is actually serialised in API responses.

## Licence

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Original
`recipe_persistence` source carries its own SPDX notices; retain them. Third-party
patches, the vLLM base and model/draft weights remain subject to their respective
upstream terms. **No model weights, images, credentials or private operations scripts
are distributed here.**
