# IMPLEMENTATION — patch 0010: the in-RAM durable-key index (2026-09-12)

Kills the lookup-scan tax measured in the write-behind report addendum
(~0.7s at 8k, ~1.2s at 64k on the scheduler's critical path) without
touching the engine's scan or the group convergence logic.

## Root cause (source-verified 2026-09-12)

The manager's per-chunk `lookup()` fans a `reserve_read` RPC to ALL
FOUR rank coordinators (RemoteCoordinator._reserve → client.fanout).
Each rank's `_MetadataRPC._rpc_reserve_read` answers with a DiskStore
`reserve_read` — a **per-key sqlite index read plus a file stat**, hit
or miss. Near-miss prompts (long shared prefix, unique tail — the
agentic shape) walk deep: ~100 keys × 4 ranks of index reads +
localhost HTTP ≈ the measured seconds. There is no negative
short-circuit anywhere.

## Design

One owner of truth: the **DiskStore grows an in-RAM set of durable
`(namespace, key)` pairs**, because every durability transition already
funnels through it:

- Built at the end of `_recover()` (after file-validity tombstoning and
  `collect`) from `state='C'` rows. Build failure leaves the set `None`
  = permissive fallback (old slow path, correctness preserved).
- `durable(ns, key) -> bool`: O(1) set lookup; `None` set → True.
- `reserve_read` short-circuits absent keys **before** `_expire()`,
  lease-count, index SELECT, and file stat.
- The set is maintained on every transition that moves a row into or
  out of `state='C'`: write completion (add), invalidate (discard),
  the reserve_write supersede-tombstone (discard), the 'C' expiry
  sweep, and reclaim (discard) — each site already holds the row's
  identity or selects it.

Coordinator side: `_rpc_reserve_read` checks `store.durable(ns, key)`
immediately after envelope validation and returns
`{"ok": True, "lease": None}` with a static `absent_index_skips`
counter — **no store I/O, no why-refusal probe** (which would itself
re-read the index). Idempotent-replay safety: a cached lease for a key
that became absent is dead anyway (invalidate semantics), so
short-circuiting before the replay cache is correct.

Explicitly NOT in 0010 (staged): batching the per-key RPCs into one
`reserve_read_multi` call (needs engine-scan cooperation), and
in-process dispatch for the local rank leg (RemoteCoordinator local
shortcut). The durable set removes the I/O from every leg; those two
remove the per-leg HTTP overhead when profiling says it still matters.

## Correctness invariants

1. The set is a conservative superset for positives: a key is in the
   set iff a 'C' row exists. All add/discard sites are the same sites
   that execute the corresponding SQL.
2. `durable() is True` (unbuilt) preserves today's behaviour exactly.
3. Absent short-circuit returns the same shape as a store refusal
   (`lease: None`), so the client's all-rank completeness logic is
   untouched.
4. Concurrency: all mutations hold `self._lock` (the store's existing
   discipline); the set is only read under the same lock or via
   `durable()` which is safe for membership tests (CPython set).

## Tests

- Rebuild: pre-populated store (C + T + W rows) → set has exactly the
  C pairs after open; tombstones/absent excluded.
- Fast negative: absent reserve_read performs **zero** db executes
  (counting wrapper) and returns None.
- Add on completion: write → reserve_read succeeds; set contains pair.
- Discard on invalidate / supersede-tombstone / 'C' expiry sweep /
  reclaim → reserve_read refuses afterwards.
- Fallback: set forced None → reserve_read uses the SQL path.
- Reload: pairs persist across close/reopen.
- Concurrency pins: interleaved complete/invalidate/lookup visibility.
- Full existing suites stay green (storage 24, native 114,
  failure-injection 12, harness, conc runner).
