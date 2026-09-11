# FIX: `PERSIST_LOOKUP_KEYS_PER_STEP` was hanging requests (2026-09-11 AEST)

A configuration value, not a code change, was causing requests to hang
indefinitely. It is fixed, and the fix is verified by a controlled A/B.

---

## 1. The defect

`PERSIST_LOOKUP_KEYS_PER_STEP` bounds how many **new** disk-tier reservations the
manager may charge in one scheduler step. `_Manager._admit_lookup` charges a chunk
while the budget lasts and returns `None` — vLLM's `RETRY` — for every chunk after
that.

vLLM's `_maximal_prefix_lookup` (`.../offloading/scheduler.py`) treats a single
`RETRY` as "defer the whole lookup":

```python
case RETRY:
    defer_lookup = True       # does NOT break
...
return hit_count if not defer_lookup else None
```

So one throttled chunk **discards every hit already found in that scan** and
returns `None`, which makes the scheduler defer the request. Our shipped value was
**8**. A real prompt needs far more than eight fresh reservations, so the scan
returned `None` on essentially every pass and the request was deferred forever.

This is the mechanism behind the "restore path hangs" defect. It is not the store
and not the lookup: the tier answers correctly, and then the request is deferred
anyway.

## 2. Controlled A/B

Same engine build, same 8,171-token prompt, cold then immediately repeated, client
timeout 240 s. Only `PERSIST_LOOKUP_KEYS_PER_STEP` differs.

| | budget = 8 (shipped) | budget = 2048 |
|---|---|---|
| cold pass | 4.6 s | 9.3 s |
| **repeat pass** | **hung, client timed out at 240 s** | **completed in 7.6 s** |
| `deferred` during the repeat | 1, continuously from t=2 s to t=240 s | 0 |
| disk chunk queries attributed to the repeat | **34,400** | **3,528** |
| disk chunk hits found | 72–112 | 74 |
| `kv_offload_load_bytes_total` | 0 | 0 |

Two things changed and one did not.

- **The hang is gone.** The repeat completes normally instead of being deferred
  indefinitely. That is the headline.
- **Lookup churn fell roughly ten-fold.** 34,400 queries for one request became
  3,528 — the retry storm was a direct consequence of the throttling.
- **The disk still does not serve.** Load bytes remain exactly 0, because
  `prepare_load` is still never reached. That is a *separate* defect and this fix
  does not address it.

## 3. Sizing rule

Measured on this stack: an 8,171-token prompt resolved to **74 chunks**, i.e. about
**110 tokens per chunk**. The budget must therefore be at least the chunk count of
the longest prefix you intend to restore:

```
PERSIST_LOOKUP_KEYS_PER_STEP >= ceil(max_expected_context / tokens_per_chunk)
```

| Target context | Chunks at ~110 tok/chunk | Budget |
|---|---:|---:|
| 8k | 74 | 128 |
| 64k | 580 | 1024 |
| 128k | 1,160 | 2048 |
| 256k | 2,330 | 4096 |
| 405k (largest prompt measured safe here) | 3,700 | 4096 |

The value shipped in this release is **2048**, which covers every context size in
the session matrix. A deployment configured for the largest measured safe prompt
should use 4096. The old value of 8 was not merely suboptimal: it made restores
hang for any prompt longer than roughly 900 tokens.

## 4. The residual defect, stated precisely

Raising the budget works around the problem rather than fixing its shape. The
underlying issue is a contract mismatch:

- our manager uses `RETRY` to mean "I am throttling this step, ask again later";
- vLLM uses `RETRY` to mean "I cannot answer yet, and you should discard what I
  have told you so far".

Those are not the same statement, and the second is destructive to progress. A
budget-exhausted chunk should almost certainly report `MISS` — terminating the scan
cleanly and letting the caller act on the prefix it did find — with the budget then
bounding hit-rate growth rather than liveness. That change is **not** made here: it
alters hit/miss behaviour and needs its own controlled measurement, particularly
because the `RETRY` path is also used for genuine lease-renewal uncertainty, where
`MISS` would be wrong.

Recorded as defect O13. Two further points belong with it:

- **Still open:** `prepare_load` is never reached, so a resolved hit is never turned
  into a transfer. Load bytes stay 0 with the budget fixed, which isolates this
  cleanly from the hang.
- **Not deterministic:** before the fix the same prompt sometimes recomputed and
  completed in 9.2 s rather than hanging. Any report quoting a single run of that
  path would be misleading, so both outcomes are on record.
