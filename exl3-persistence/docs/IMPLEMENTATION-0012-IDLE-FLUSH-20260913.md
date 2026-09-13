# IMPLEMENTATION — patch 0012: the idle-time staleness flusher (2026-09-13)

Hugh's policy, stated 2026-09-12: "only start writing to disk when there is
concurrency and memory pressure; if contents are quite stale — over a
configurable number, one hour default — flush them to disk in idle time, so
normal non-pressurized decode has truly zero tax."

0011 delivered the first clause (pressure gate: idle engines write nothing,
measured 0 bytes / −4.6%-under-flood). 0012 delivers the second: durability
for stale content without any memory pressure, at decode-zero cost.

## Design

Population: APC-resident blocks enumerated from the block pool's own hash
index (`cached_block_hash_to_block`) — exactly the cache, no side map for
enumeration. Ages come from an in-connector `block_id -> last_seen` map,
updated for every block of every active request each step (bounded by
max_num_seqs); blocks absent from the map (never seen this process life)
are treated as stale — correct, because all APC content originated from
a request.

When to run: **decode-free steps only** — a step whose scheduled batch has
no decode tokens (prefill chunks, finish-drain steps). The engine makes no
steps at all on a fully idle node (measured, round 15), so "idle time" is
operationalized as "steps that carry no decode": the flusher lands stale
bytes during the next prefill-heavy window, never competing with a decode
step (R1 by construction). A fully idle engine therefore flushes nothing
until activity resumes; a one-request driver forces progress when needed.

Rate: one bounded scan per eligible step, at most
`idle_flush_per_scan` (default 32) blocks, at most one scan per
`idle_flush_scan_seconds` (default 5). Copies are **in place**: the blocks
stay in the APC; reuse is fenced through the existing
`_block_id_to_pending_jobs` flush machinery (the 0002-era fence: a block
pending a store is flushed-and-awaited before reallocation — allocation is
never blocked indefinitely, and the copy reads stable bytes).

Jobs are synthetic `req_id="idle-flush"` stores through the normal manager
admission (durable probe dedups against the sink's copies), completed by
the existing eviction-completion branch (extended to accept both synthetic
owners; held-block release is a no-op for in-place copies).

Config (spec extra_config): `idle_flush_stale_seconds` (default 3600),
`idle_flush_per_scan` (32), `idle_flush_scan_seconds` (5),
`idle_flush_enabled` (default true in evict-only mode, ignored otherwise).
Counters: `idle_flush_candidates`, `idle_flush_enqueued`,
`idle_flush_skipped_fresh` — static, observable.

## Tests (AST)

- decode-free gate: a step WITH decode tokens never scans.
- staleness: fresh blocks skipped; stale blocks (clock advanced) become
  candidates; never-seen blocks are stale.
- rate bound: per-scan enqueue capped at idle_flush_per_scan.
- job shape: fenced ids registered in _block_id_to_pending_jobs, no
  _held_evictions growth (in-place, not held).
- completion: idle-flush jobs take the synthetic-owner release branch.
- config: defaults; disabled when store_mode != evict_only.
