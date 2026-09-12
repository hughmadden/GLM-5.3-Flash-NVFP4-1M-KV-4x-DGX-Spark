# FIX ROUND 23 — 0010b: the lookup tax is dead at 64k (2026-09-12 ~20:15 AEST)

In-process dispatch for the rank-local coordinator leg
(`RpcClient.bind_local_dispatch`): the worker role builds its
MetadataServer in-process, so 1 of every 4 reserve RPCs per lookup key
(now fast negatives via 0010a) skipped the localhost HTTP round trip.
Error mapping mirrors the HTTP status semantics exactly (400/410 ->
_Permanent fail-closed, 503/500 -> _Transient); unbound clients are
mechanically inert.

Tests: 7 new (wire-vs-local proof against unroutable peer stubs, fanout
rank-order merge, absent fast-negative through the local leg, all three
error mappings, bind validation, unbound fallback). 114 pytest + full
unittest suites green; conc failures unchanged in class (the
expired-lease pin is a documented 0.4s-lease timing flake, mechanically
unreachable here — unbound clients take the old path).

Live (same ladder instrument throughout):
- 64k cold: **31.96s (2,026 tok/s) vs OFF 2,028 — tax ~0%**
  (was −12% pre-0010, −5% after 0010a).
- 256k cold: 139.4s (1,919 tok/s) with 16,128 external hits integrated
  mid-prefill (deep cross-session restore working); warm-probe 2.4s
  with 264,960/262,144 tokens served from cache.
- Decode parity holds; background eviction copies draining; health 200.

The staged 0010 follow-up list is now one item deep (true batching of
the per-key RPCs needs engine-scan cooperation = engine patch 0011);
profiling no longer justifies it at current latencies.
Commit `9bd3703d`.
