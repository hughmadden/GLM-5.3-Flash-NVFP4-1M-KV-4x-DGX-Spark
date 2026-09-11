# REFILL HANG — ROOT CAUSE NARROWED (2026-09-11 AEST)

The disk tier's read path never completes. This document records the measurement
that characterises it, and one hypothesis that was **tested and rejected** on the
way. The rejection is kept deliberately: it is the difference between "we think
this is the bug" and "we know these are not the bugs".

Companion evidence: `receipts/report-20260911/refill-patience.txt` (raw sampler
trace), `refill-verdict.txt`, `REPRO-refill-hang.txt`.

---

## 1. The measurement

`probes/probe_refill_patience.py`: store one unique 8,171-token prompt cold, then
re-request it while sampling the engine's own counters every 2 s for 240 s.

### Pass 1 — cold

| | |
|---|---|
| wall time | 4.6 s |
| prompt tokens | 8,171 |
| `prompt_tokens_details` | `{cached_tokens: 0, created_cache_tokens: 0, ...}` |
| disk store bytes | **252,797,952** |
| disk store operations | **112** |
| external prefix cache queries | +8,171 |

### Pass 2 — immediate repeat

```
t=  0.0s  queries=  34916 hits= 76 load_B=0 deferred=0 running=0
t=  2.0s  queries=  35708 hits=112 load_B=0 deferred=1 running=0
t=  4.0s  queries=  36540 hits=112 load_B=0 deferred=1 running=0
...
t= 74.4s  queries=  66820 hits=112 load_B=0 deferred=1 running=0
t= 76.4s  queries=  67580 hits=112 load_B=0 deferred=1 running=0   <- query rate collapses
...
t=124.6s  queries=  69476 hits=112 load_B=0 deferred=1 running=1   <- something runs
...
t=239.0s  queries= 71028 hits=112 load_B=0 deferred=1 running=0
RESULT: 240.1s  ->  TimeoutError: timed out
```

Totals over the 240 s: **+36,137 chunk queries, +36 chunk hits, +0 load bytes,
+203 allocation failures, +11,280 external-cache token queries, +0 external hits.**

### What this establishes

1. **The tier holds the complete prefix and finds all of it.** Hits saturate at
   **112**, which is exactly the number of store operations pass 1 performed —
   `store_size_count = 112`. The lookup therefore matched 112 of 112 chunks, not
   112 of 2,043. **The disk tier answers every chunk it holds, once, within two
   seconds, and then has nothing left to find.** The tier is not slow, not
   partial, and not mis-keyed.
2. **No load is ever issued.** `kv_offload_load_bytes_total` stays exactly **0**
   for the full 240 s. The prefix is found and then discarded.
3. **The request is deferred, not failing.** `deferred = 1` continuously from t=2 s
   to t=240 s. Nothing errors, nothing returns, nothing times out server-side.
4. **The engine burns CPU retrying.** Queries climb at roughly **880/s** for the
   first 76 s, then collapse to ~4/s — consistent with a retry loop that is
   eventually throttled but never resolved. 36,137 lookups were issued for a
   prompt whose prefix had already been fully located at t=2 s.
5. **The external prefix cache accounting is queried and reports nothing.**
   `external_prefix_cache_queries` advances by 11,280 while
   `external_prefix_cache_hits` stays 0. That is consistent and correct: vLLM
   counts a hit only when the transfer actually supplies tokens, and no transfer
   happened.

**Conclusion: the defect is downstream of the lookup.** The store is correct, the
lookup is correct and complete, and the transfer from a discovered hit to the GPU
is never issued. This is a narrower and better-supported statement than the
earlier "the restore path does not fire".

---

## 2. Hypothesis tested and REJECTED: lease validation starving the scan

### The hypothesis

vLLM's `_maximal_prefix_lookup` (`.../offloading/scheduler.py`, ~line 646)
returns `None` if *any* chunk returns `RETRY`, discarding the whole hit count:

```python
for key in keys:
    match self.manager.lookup(key, req_context):
        case HIT:      hit_count += 1
        case HIT_PENDING: defer_lookup = True; hit_count += 1
        case RETRY:    defer_lookup = True     # does NOT break
        case MISS:     break
return hit_count if not defer_lookup else None
```

`_Manager._lease_valid` returns `False` when `coordinator.lease_deadline()`
returns `None`, and `_lease_valid` returns **before** queueing a renewal in that
case. If a freshly reserved lease had no deadline, every reserved chunk would
return `RETRY` until a renewal landed, and the scan could not converge.

A stub (`repro_refill_loop.py`) modelling that coordinator produced exactly the
observed shape: **673 scheduler steps to converge versus 10**, with one renewal
per chunk, and — with the deadline seeded at reservation — back to 10 steps with
**zero** renewals.

### Why it is wrong

Reading the real coordinator rather than the model of it:

- `coordinator_http.py:913-921` — `_reserve()` computes
  `valid_until = started + self._ttl - self._margin` and **sets
  `self._valid[ticket.token] = valid_until` at grant time**. A freshly granted
  reservation is immediately valid for ~270 s.
- `lease_deadline()` therefore returns a real deadline for a fresh ticket.

So the stub's central assumption did not match the implementation.
**The lease-validation hypothesis is rejected.** No change was made to
`_lease_valid`, `_valid`, or the renewal path on the strength of it.

The stub is retained with `--fix`/`--async` flags as the record of the test, not
as a diagnosis.

---

## 3. Where the defect now sits

Ranked by the evidence above:

| Candidate | Status |
|---|---|
| Store write path | **excluded** — 252 MB, 112 objects, durable across restarts |
| Lookup / keying / geometry | **excluded** — 112 of 112 chunks matched, within 2 s |
| Lease validation and renewal | **rejected** — see §2 |
| Admission of new lookups (`_admit_lookup`) | **not the blocker** — it limits *new* reservations, and all needed reservations succeeded at t=2 s |
| The transfer from a resolved hit to the GPU | **the remaining site** — `prepare_load` → worker read → `complete_load`; load bytes never move |

The next instrument is a debug trace on our side at the `prepare_load` /
`complete_load` boundary, plus vLLM's offloading-scheduler debug log, to establish
whether `prepare_load` is called at all and, if it is, why the resulting transfer
job never completes. That is a single instrumented redeploy.

---

## 4. Effect on the objective

The disk cache **is** wired into the cache-hit metric and does count disk hits as
cache hits at the accounting surface:
`vllm:kv_offload_tiering_chunk_queries_total{tier="disk"}` = 36,137 and
`vllm:kv_offload_tiering_chunk_hits_total{tier="disk"}` = 112, both live, plus
`usage.prompt_tokens_details.cached_tokens` now serialised. Those counters are
what made §1's conclusion reachable: without them, a fully-found prefix and an
empty disk would have been indistinguishable from the outside.

What remains open is serving, not counting: vLLM's *served* external-prefix-hit
metric stays 0 because the transfer is never issued. That is recorded as O1/O2.

---

## 5. Update: the load lifecycle is never entered (instrumented, 2026-09-11)

Path tracing was added to the manager and to the transfer pump and deployed to all four
ranks. Each boundary emits one bounded, static-code notice the first time it is reached,
which turns "where does it stop?" into a list of which boundaries were visited at all.

After a cold store followed by a repeat (repeat timed out at 60 s), the traces reached on
the head rank were:

| Boundary | Reached |
|---|---|
| `lookup_disk_hit` — the disk tier returned a valid all-rank lease for a chunk | **yes** |
| `lookup_cached_reservation` — an already-reserved chunk was re-read | **yes** |
| `lookup_budget_retry` — a chunk was throttled by `lookup_keys_per_step` | **yes** |
| `lookup_capacity_miss` | no |
| `prepare_load` — vLLM asked the manager to stage a load | **NO** |
| `complete_load` | no |
| `on_load_failure` | no |
| `submit_load` (worker-side pump) | **NO** |
| `submit_store` (worker-side pump) | no |
| any `Transfer submission refused` code | **none** |

Three conclusions:

1. **The failure is not in the store, and not in the lookup.** The tier is reached and it
   does return leases.
2. **It is not in the transfer pump either.** The pump never refuses anything, because it
   is never asked: `submit_load` is never called. The nine silent rejection paths added to
   `_submit` all stayed empty, which is a useful negative — that code was a plausible
   suspect and is now excluded by instrument rather than by argument.
3. **The stop is upstream of `prepare_load`.** vLLM's `_maximal_prefix_lookup` returns
   `None` — deferring the whole request and discarding the hit count — as soon as any chunk
   returns `RETRY`, and `lookup_budget_retry` is reached. So the deferral is produced before
   the loader is ever consulted, and `kv_offload_load_bytes_total` stays 0 because no load
   was ever prepared, not because a prepared load stalled.

That relocates the defect once more, and this time to the interface between our
per-step lookup budget and vLLM's all-or-nothing deferral contract: a `RETRY` from a
throttled chunk costs the caller every hit already found in that scan. The next instrument
is vLLM's own offloading-scheduler debug log alongside a raised `lookup_keys_per_step`, to
separate "the budget defers too often" from "reservations are not retained between steps".

Note also that the behaviour is not deterministic across runs: the same prompt and
configuration recomputed and completed in 9.2 s in one run and hung past 240 s in another.
A report that quoted only one of those runs would be misleading, so both are recorded.
