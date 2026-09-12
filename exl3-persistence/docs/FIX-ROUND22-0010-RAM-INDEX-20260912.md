# FIX ROUND 22 — 0010 live: the scan tax roughly halved, all ranks verified clean (2026-09-12 ~19:45 AEST)

The two slow workers during the restart were weight-load skew; verified
post-boot: all four containers Up together, all four rank stores
`PRAGMA integrity_check = ok` with byte-identical census (21,396 'C'
objects on every rank — the all-rank write contract intact), overlay
byte-identical on all four ranks, and the only boot "errors" are the
pre-existing benign STARTUP_OBSERVER_INCOMPLETE notices.

## 0010 verification summary

- Implementation: DiskStore in-RAM durable-key set (rebuilt at the end
  of `_recover`, maintained on every 'C' transition: write completion,
  invalidate, read-failure retire, evict reclaim); fast negatives in
  `reserve_read`/`exists`; coordinator `reserve_read` short-circuits
  absent keys with a static `absent_index_skips` counter, no store I/O.
- Tests: 10 new unit tests; 114 pytest + full unittest suites green;
  the six conc failures are byte-identical with 0010 stashed (pre-
  existing environment pins, zero regressions).
- Live (same ladder instrument as rounds 15/18): 64k cold **33.5s
  (1,931 tok/s) vs OFF 32.3s (2,028) — tax now ~5%**, down from −12%
  pre-0010 at the same size; decode 44.5–52.5 at parity; background
  eviction copies draining (159 MB windows); engine healthy.
- The remaining ~5% is per-leg HTTP overhead (hits still cost 4
  localhost reserve RPCs each) — the staged 0010 follow-ups:
  `reserve_read_multi` batching and in-process dispatch for the local
  rank leg.

Fleet: evict-only, gate 32768, 0010 via overlay, health 200. Commit
`f26b2b38`.
