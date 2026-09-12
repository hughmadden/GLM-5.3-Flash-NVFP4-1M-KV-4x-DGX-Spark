# IMPLEMENTATION — patch 0009: the write-behind disk tier (2026-09-12)

Implements `DESIGN-EVICT-ONLY-SPIKE-20260912.md` (binding requirements
R1–R3). Eager mode stays as a config mode (the design: "a mode, not a
deletion"); the tidy-up removes what is defunct in BOTH modes. The
endgame is the §3.4 dirty-ratio dial; 0009 lands its evict-only endpoint
and the R3 read gate, with the dial's seams already in place.

---

## 0. Surfaces established (pinned 83252ea89 + 0001–0008)

- `OffloadPolicy(Enum)` in `vllm/v1/kv_offload/base.py` — today
  `CHUNK_LEVEL | REQUEST_LEVEL`; `RequestOffloadingContext.policy`.
  0009 adds `EVICT_ONLY = "evict_only"`.
- Block-pool eviction events: `block_pool._emit_block_removed_events`
  emits `BlockRemoved(block_hashes, medium, group_idx)` into
  `kv_event_queue` — POST-HOC: emitted when the block is freed, after
  `cached_block_hash_to_block.pop` and `block.reset_hash()`. The block
  is then free for immediate reallocation. **A post-hoc copy races
  reallocation**, so the deferred free must intercept BEFORE the pool
  retakes the block.
- Core event fan-out (`v1/core/sched/scheduler.py:2246`): block-pool
  events are merged with connector events and PUBLISHED to external
  subscribers. Nothing feeds them back into the connector today.
- `make_offload_key(block_hash, group_idx)` — the offload key embeds the
  prefix-cache block hash, so a `BlockRemoved` maps to its offload key
  directly (no request-state lookup needed).
- The deferred-free MODEL already exists for stores:
  `_block_id_to_pending_jobs` + jobs-to-flush fence a block's REUSE while
  a store job reads it (reallocation triggers a flush-wait). 0009
  reuses this fencing for eviction copies.

## 1. The deferred-free interception (the one deep hunk)

`KVCacheCoordinator.free(...)` / the coordinator's free paths return
blocks to the pool. 0009 adds an optional connector hook:

```python
# in the coordinator's free path, before blocks join the pool's free list:
if connector is not None:
    held = connector.defer_evicted_blocks([(block, hash) for ...])
    # held blocks move to the connector's deferred-free registry;
    # the pool does not see them until released.
```

`defer_evicted_blocks` returns the subset the connector elects to hold
(under its credit bound); the rest free normally. The connector's
registry: `block_id -> (offload_key, group_idx, refcount)`. Copy path:
the existing pump (`submit_store`) reads GPU → CPU → disk; on
`complete_store` drain the block is released back to the pool (a new
connector→coordinator release callback, delivered through the existing
`update_connector_output`/worker-meta round trip — the same channel
finished_sending uses). Skip-on-timeout: if the pump's admission
(credits/ring) cannot start the copy within a bounded window, the block
is released unstored and the static skip counter increments. The
allocation path is NEVER blocked: held blocks are outside the pool, so
worst case is bounded memory pressure from the registry itself (credit-
bounded by `max_pending_keys`), never a stall.

Guard: only blocks whose (hash, group) maps to a complete stored-shape
chunk are held; partial/volatile tail chunks (the eagle exclusion) are
freed immediately — the existing `storable_chunks` arithmetic decides.

## 2. Store path under EVICT_ONLY (scheduler)

- `_build_store_jobs` offers nothing: at mode entry it returns {}.
  No `finish_pending` hold, no `finished_sending` gate for stores, no
  frontier advance (0002/0004/0005/0007 store-mode hunks are inert).
  The response path is untouched (R1: zero BAU tax by construction).
- Eviction-store queue: credit-based, fed by `defer_evicted_blocks`;
  jobs are built per evicted chunk key with the SAME manager
  `prepare_store` admission path (durable probe + skip), the SAME pump
  transfer, the SAME `complete_store`/`_advance_store_frontiers`
  accounting (frontiers become eviction bookkeeping, not request state).
- `has_pending_push_work` includes the registry (drain before reset).
- `reset_cache` releases the registry (fail-closed: un-copied blocks
  return to the pool; copied ones are durable).

## 3. Read gate (R3)

- Config: `PERSIST_MIN_DISK_LOOKUP_TOKENS` (spec extra_config
  `min_disk_lookup_tokens`, default 4096, validated positive).
- Engine hunk in `get_num_new_matched_tokens`:
  `if request.num_prompt_tokens < self.min_disk_lookup_tokens: count a
  static skip counter; return (0, False)` — before `_lookup`, RAM hit or
  not. The gate keys on prompt size (per R3), not miss size.
- The counter is exported with the tier stats (observable, never
  silent). The WRITE path is size-agnostic (R3).

## 4. Tests

Stub-first (the package harness), then the series AST suite:

- `test_evict_only_offers_nothing`: a full store-then-finish cycle under
  EVICT_ONLY produces zero store batches (the batch-shape probe sees
  nothing).
- `test_block_removed_maps_to_offload_key`: a BlockRemoved(hash, g)
  yields exactly `make_offload_key(hash, g)`; duplicates coalesce.
- `test_deferred_free_lifecycle`: held block is not pool-visible; freed
  after complete_store drains; skip-on-timeout frees unstored and counts.
- `test_read_gate`: prompt below the threshold never consults the
  manager (lookup counter zero, skip counter +1); at/above it the
  manager is consulted from the RAM boundary (the two RAM-first pins).
- Series AST: mode A/B invariants (frontier no-ops, response path
  untouched) for BOTH modes.

## 5. Tidy-up (with 0009, both modes)

- Remove the `lookup_keys_per_step` default of 8 (now a required config;
  the pathological default hangs restores — FIX-LOOKUP-BUDGET).
- The store/lookup concurrency pins that encoded the OLD eager-default
  behavior stay only where both modes share the contract.
- Eager mode machinery remains, clearly marked as the `eager` dial
  endpoint (§3.4), until the A/B retires it.

## 6. Sequencing

1. Package config + engine-free gate tests (this round).
2. Patch 0009 hunks (policy enum, deferred-free hook, scheduler queue,
   read gate) + AST tests; chain verify.
3. Image bake; live A/B: BAU leak audit → concurrency burst (2× pool)
   → old-session restore; skip-rate reporting.
4. PO the A/B matrix (prefill/decode × concurrency × sizes incl. 100k+
   agentic contexts), then the full benchmarked report.
