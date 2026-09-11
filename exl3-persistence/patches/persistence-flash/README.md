# Native offloading correctness patches — re-derived onto `83252ea89`

**Candidate, CPU/source-tested; not GPU or distributed runtime qualification.**
Pinned upstream: [`vllm@83252ea899c6538eaa0c1fb31f28a92c661bbffc`](https://github.com/vllm-project/vllm/tree/83252ea899c6538eaa0c1fb31f28a92c661bbffc)
(main, 2026-09-09; contains GLM-5.3-Flash `glm5next`).

This is the port of `patches/persistence/` (pinned to `ab666069`) onto the new
`OffloadingSpec` / `OffloadingWorker` API. It keeps the same four contract
groupings and the same eight-file blast radius minus one deleted file: the four
patches now modify **seven** native files, because
`vllm/v1/kv_offload/worker/worker.py` no longer exists upstream. They are
explicit build-time changes, not an import-time monkeypatch or a replacement
connector.

## Apply and test

Use Python 3.10+ and Git with a quiescent source tree. `VLLM_SOURCE` is either
a source root containing `vllm/` **or** the `vllm` package directory itself (the
bare overlay layout shipped in some runtime images) — `apply.py` and
`test_native.py` accept both and map the always-`vllm/…` manifest keys onto
whichever it was given:

```sh
python3 patches/persistence-flash/apply.py check "$VLLM_SOURCE"
python3 patches/persistence-flash/test_native.py --source-tree "$VLLM_SOURCE" -v
python3 patches/persistence-flash/apply.py apply "$VLLM_SOURCE"
python3 patches/persistence-flash/apply.py verify "$VLLM_SOURCE"
# Roll back only an unchanged patched source tree:
python3 patches/persistence-flash/apply.py reverse "$VLLM_SOURCE"
```

Tests copy only the manifest-listed files into a temporary directory and execute
AST-extracted classes/methods against fakes. They do not import vLLM or torch.
A **pristine** `--source-tree` has the real patches applied in the temporary
copy first; an **already-ported** `--source-tree` is detected by its AFTER
hashes and used as-is. `--no-apply` runs the suite against the tree exactly as
given, which is how the pre-fix reproduction is demonstrated.

`manifest.json` records exact before/after SHA256 values, unique source anchors,
patch ordering and patch digests. All inputs and staged results are validated
before replacing any source file. An unexpected file or patch fails closed.
Application is a build operation, not crash-atomic deployment across files;
never patch a running installation. Build/runtime rollout and rollback must
use separate immutable artifacts.

Apply against pristine pinned source **before any other patches modifying
these files**. Later baseline overlays require composition review and their own
final build receipt. Do not weaken the hashes to accept an arbitrary fork.

## Semantics and package contract

1. **Drained outcomes.** Failed submissions call `wait({job_id})` before a
   synthetic failure. Handler `wait` must tolerate unknown IDs and drain all
   partially issued GPU, CPU and media work; every `TransferResult`, including
   failure, certifies no later buffer access. Persistent-store success means
   durable publication, not merely completed D2H. An inability to drain must
   raise: it cannot safely become a recoverable cache miss. Shutdown drains
   outstanding and deferred transfers before clearing ownership.
2. **All-rank boundary.** Worker metadata aggregates completion counts,
   failure vetoes and failed-load destination block IDs. Workers do not emit
   load `finished_recving` independently. The connector scheduler waits for
   every rank, invokes existing `complete_store(..., success=False)` on any
   failed shard, and publishes load invalid IDs and finished-receive together.
   Core callback ordering now reduces connector outcomes before invalid-block
   recovery. A vanished rank cannot be treated as drained; runtime failure
   handling must stop/restart safely rather than inventing its completion.
3. **Failed loads.** After all-rank drain, call `complete_load(keys, context)`,
   then the new default-no-op `on_load_failure(keys, context)` hook on failure.
   Persistent managers must invalidate/quarantine those objects across their
   coordinator. The affected request bypasses further external reads to avoid
   infinite retry even with legacy managers; GPU APC remains enabled. Its
   optimistic hit-store frontiers reset so recomputed blocks can be stored.
4. **Group-aware recovery.** For multiple groups, identify affected requests
   by physical block membership, never by flattening unequal token geometries.
   Conservatively reset the affected computed prefix while retaining accepted
   request/output tokens. Async loads retain existing waiting/free ordering;
   synchronous grouped recovery evicts dependent cache entries and uses the
   existing preemption/free fence for fresh all-group allocation. Single-group
   valid-prefix behavior is retained. No HMA disablement or allocator changes.
5. **Retryable store frontiers.** Each group's frontier stops before its
   EAGLE-family volatile tail and its first unacknowledged offered key.
   `PrepareStoreOutput.skipped_keys=()` is a new optional field: list only
   intentional policy skips or confirmed durable existing keys, **not capacity
   declines or other pending writes**. `None` and empty output without skips
   remain retryable; skip-only output advances without a zero-key job.
   Preserve offered group/key ordering in `keys_to_store` and the paired spec.
   Managers must not admit duplicate writes for keys already owned by a job.
   The native CPU producer acknowledges policy skips and ready keys without
   calling `lookup` (which would mutate reuse counters).
6. **Ownership on retry/reset.** Failed stores rewind full-attention frontiers.
   Failed SWA positions behind the previous frontier are conservatively nulled:
   those rows may have been recycled while the write ran. They must not be
   reread through stale IDs. Existing preemption/reallocation flush fences are
   retained. `reset_cache()` returns `False` while native jobs or manager work
   remain; retry only after normal idle draining.
7. **Finished-request frontiers.** A normal stop/length finish retains the core
   request, its current GPU block table and connector context until all eligible
   stable keys have been admitted/explicitly skipped and all admitted jobs have
   drained on every rank. Partial and zero admission do not release ownership.
   Held finished requests are offered first on later steps, including idle
   no-forward steps. The finish hook snapshots the current table supplied by
   core after removal of skipped SWA rows, not historical recycled IDs.
   Only the terminal scheduler `finished_sending` signal releases the held
   request; aborted loads retain their existing receive-release path.
8. **Bounded failure, not false persistence.** Abort/error/ignored finishes stop
   new store admissions but drain existing jobs. A finished full-attention key
   gets at most one additional retry after its first all-rank failed transfer;
   a second failure of that key abandons further admissions for that finished
   frontier, logs an explicit incomplete-store warning, and still drains every
   already-issued job. Failure-key tracking is cleared on release. Temporary
   pressure/quantum declines consume no failure budget. Failed SWA rows are
   nulled, never retried through stale IDs. The new default-true manager hook
   `can_store()` must return false on permanent storage disablement, **not**
   temporary pressure; this also abandons unsaved frontiers without advancing
   them or fabricating skips/success. Permanent refusal mislabeled as temporary
   (`can_store()` stays true forever) can still hold ownership indefinitely.
   This is storage recovery only, not an inference retry or admission timeout.

These changes do not add durable cross-process metadata consensus, checksums,
quotas or a disk handler. Those belong to the external persistence package and
must fulfill the contracts above. No factory, GCD/hash geometry, model, kernel,
DFlash lookahead, cache sizing, context limit, APC or scheduling configuration
is changed. EAGLE-family flags still come from native runtime group metadata.

These changes do not add durable cross-process metadata consensus, checksums,
quotas or a disk handler. Those belong to the external persistence package and
must fulfill the contracts above. No factory, GCD/hash geometry, model, kernel,
DFlash lookahead, cache sizing, context limit, APC or scheduling configuration
is changed. EAGLE-family flags still come from native runtime group metadata.

## Port provenance — old hunk → new location

Source of the classification: `notes/port-study.md` (§B, 50 hunks:
4 ABSORBED / 44 STILL-NEEDED / 2 OBSOLETE). Line numbers below are the
**83252ea89** tree; the study's were `4ebf61ebda`, so a few shifted.

### 0001 — drained-outcome API (11 old hunks → 9 carried, 2 dropped)

| Old hunk | Old location | New location on `83252ea89` | Status |
|---|---|---|---|
| 1 | `offloading/common.py` `OffloadingWorkerMetadata` fields | same file, `:85-99` (`failed_jobs`, `failed_load_blocks`) | carried |
| 2 | `offloading/common.py` `mark_completed` + `aggregate` | same file, `:88-113` | carried |
| 3 | `kv_offload/base.py` `PrepareStoreOutput.skipped_keys` | `base.py:150-154` | carried |
| 4 | `kv_offload/base.py` `OffloadingManager.on_load_failure` | `base.py:371-381`, inserted after `on_request_finished` | carried |
| 5-7 | `worker/worker.py` `OffloadingHandler.{transfer_async,get_finished,wait}` docstrings | `base.py:579-623` on **`OffloadingWorker`**; the single `transfer_async` doc is split across `submit_store` and `submit_load`, and the two `...`-bodied abstract methods gained real docstring bodies | carried, re-expressed |
| 8-9 | `worker/worker.py` `OffloadingWorker` (dispatcher) docstrings | — | **OBSOLETE**: the `(src_medium, dst_medium)` dispatcher class was deleted by `f237e16b41` (#45053). There is no second copy of the contract to annotate; the single-medium `OffloadingWorker` ABC carries it once (hunks 5-7) |
| 10-11 | `cpu/manager.py` `prepare_store` skips | `cpu/manager.py:184-207`, `:256-261` | carried, **merged** with upstream's new `_record_accesses(keys)` call in the same block (added after our pin) rather than overwriting it |

### 0002 — offloading failure frontiers (20 old hunks → 19 carried, 1 absorbed)

| Old hunk | Old location | New location on `83252ea89` | Status |
|---|---|---|---|
| 1 | `offloading/worker.py` import `TransferResult` | same file, imported from `vllm.v1.kv_offload.base` (the `worker/` package is gone) | carried |
| 2 | `worker.py` `__init__` ownership state | `offloading/worker.py:57-62` | carried |
| 3 | `worker.py` `_submit` + `handle_preemptions` | **split** into `_drain_failed_submission` + `_submit_store` + `_submit_load` (`offloading/worker.py:212-236`); `handle_preemptions` at `:255-257` | carried, split |
| 4 | `worker.py` `start_kv_transfers` | `offloading/worker.py:263-278`; `entry.transfer_spec[1]` → `entry.dst_spec` | carried |
| 5-6 | `worker.py` `get_finished` | `offloading/worker.py:291-341`; the `assert transfer_result.success` is removed | carried |
| 7 | `worker.py` `shutdown` drain | `offloading/worker.py:343-355`; guarded on `self.worker is not None`, which upstream made Optional | carried |
| 8 | `scheduler.py` `TransferJobStatus.{failed,failed_load_blocks}` | `offloading/scheduler.py:85-86` | carried |
| 9 | `scheduler.py` `RequestOffloadState.load_failed` | `offloading/scheduler.py:371-372` | carried |
| 10 | `scheduler.py` `get_num_new_matched_tokens` lookup suppression | `offloading/scheduler.py:1038` | carried |
| 11 | `scheduler.py` new `_advance_store_frontiers` | `offloading/scheduler.py:1430-1454`; the per-group upper bound now comes from upstream's `RequestOffloadState.storable_chunks()` instead of a hand-rolled `num_tokens // size` + EAGLE decrement | carried, re-based |
| 12-13 | `scheduler.py` `_build_store_jobs` × 2 `advance_stored_idx` sites | `offloading/scheduler.py:1660`, `:1676` | carried |
| 14 | `scheduler.py` EAGLE volatile-tail `num_blocks - 1` | — | **ABSORBED**: upstream `storable_chunks()` (`offloading/scheduler.py:425-446`) applies the EAGLE exclusion, and `_build_store_jobs` clamps finished requests to `max(num_prompt_tokens, num_tokens - 1)` (`:1518-1520`). Both are now reached through `_advance_store_frontiers` |
| 15 | `scheduler.py` frontier advance at job creation | `offloading/scheduler.py:1734-1739`, replacing upstream's `next_stored_chunk_idx = max(..., num_chunks)` | carried |
| 16 | `scheduler.py` `update_connector_output` retired-job tolerance | `offloading/scheduler.py:1911-1918` | carried |
| 17 | `scheduler.py` `update_connector_output` drained outcomes (largest hunk) | `offloading/scheduler.py:1925-1975`; re-derived on `fenced_block_ids`/`deferred_fence_block_ids`/`_chunks_being_loaded` and on `self.config.blocks_per_chunk` in place of `block_size_factor` | carried, re-based |
| 18-19 | `scheduler.py` `reset_cache -> bool` | `offloading/scheduler.py:2114-2117`, `:2157` | carried |
| 20 | `offloading_connector.py` `reset_cache` verdict | `offloading_connector.py:190-192` | carried |

### 0003 — grouped recovery ordering (4 old hunks → 4 carried)

Lowest-risk patch: all four land near-verbatim, and the
`TODO (davidb): add support for hybrid memory allocator` anchor survives
byte-identical.

| Old hunk | New location on `83252ea89` | Status |
|---|---|---|
| 1 | `v1/core/sched/scheduler.py:1895-1899` (above `failed_kv_load_req_ids = None`) | carried |
| 2 | `v1/core/sched/scheduler.py:3016` | carried |
| 3 | `v1/core/sched/scheduler.py:3074-3094` | carried |
| 4 | `v1/core/sched/scheduler.py:3200-3214` | carried |

### 0004 — finished-store frontier (15 old hunks → 12 carried, 3 absorbed)

| Old hunk | Old location | New location on `83252ea89` | Status |
|---|---|---|---|
| 1 | `kv_offload/base.py` `can_store()` | `base.py:287-295`, before `prepare_store` | carried |
| 2 | `scheduler.py` import `RequestStatus` | — | **ABSORBED**: already imported at `offloading/scheduler.py:60` |
| 3 | `scheduler.py` `TransferJobStatus` ownership comment | `offloading/scheduler.py:78-80`; the field is now named `deferred_fence_block_ids` | carried |
| 4 | `scheduler.py` `RequestOffloadState` finish fields | `offloading/scheduler.py:373-377`; coexists with upstream's `finished_signaled` (`:369`) | carried |
| 5 | `scheduler.py` `_block_id_to_pending_jobs` ownership note | `offloading/scheduler.py:598-601` | carried |
| 6 | `scheduler.py` `_num_offloadable_tokens` + `_has_finished_store_frontier` + `_build_store_jobs` head | `_num_offloadable_tokens` is **ABSORBED** as `_calc_num_offloadable_tokens` (`offloading/scheduler.py:635`) and is called directly; iterating finished requests is **ABSORBED** via upstream's `chain(..., scheduler_output.finished_req_ids)` (`:1490-1497`). `_has_finished_store_frontier` (`:1455-1480`) and the multi-step held-request set (`:1485-1489`, `:1498-1506`) are carried | part absorbed |
| 7 | `scheduler.py` `_build_store_jobs` tail | `offloading/scheduler.py:1757-1758`, `:1780-1784`: upstream's new `if req.is_finished(): register deferred_fence_block_ids` block is **removed**, because a held request's rows cannot be re-allocated (see PORT-NOTES) | carried |
| 8 | `scheduler.py` `has_pending_push_work` | `offloading/scheduler.py:1854-1858` | carried |
| 9 | `scheduler.py` one retry per failed full-attention key | `offloading/scheduler.py:1951-1968`, composed into 0002 hunk 17's `elif` chain | carried |
| 10 | `scheduler.py` release held reqs + `finished_sending` | `offloading/scheduler.py:1994-2023`; upstream's `if finished_signaled and not transfer_jobs: del _req_status` is now additionally gated on `not finish_pending`, and the post-loop release scan owns the terminal signal | carried, re-based |
| 11-12 | `scheduler.py` `request_finished(request, block_ids)` | `offloading/scheduler.py:2040-2100`; upstream's deferred-fence registration loop is replaced by the hold | carried |
| 13 | `scheduler.py` `reset_cache` gate | `offloading/scheduler.py:2116` (`has_pending_push_work()`) | carried |
| 14-15 | `offloading_connector.py` `request_finished` / `request_finished_all_groups` | `offloading_connector.py:172`, `:180` | carried |

## Semantics that changed because of upstream renames

* `block` → `chunk` throughout (`138d137b5b`, #52615): `next_stored_block_idx`
  → `next_stored_chunk_idx`, `offloaded_block_size` → `tokens_per_chunk`,
  `block_size_factor` → `blocks_per_chunk`, `sliding_window_size_in_blocks` →
  `sliding_window_size_in_chunks`, `_blocks_being_loaded` →
  `_chunks_being_loaded`, `OffloadPolicy.BLOCK_LEVEL` → `CHUNK_LEVEL`. Only
  names: the frontier is still a per-group index into `offload_keys`.
* `OffloadingHandler.transfer_async(job_id, spec)` → `OffloadingWorker`'s
  `submit_store(job_id, src_spec, dst_spec)` / `submit_load(...)`
  (`f237e16b41`, #45053). Our failed-submission drain therefore exists twice
  (`_submit_store` / `_submit_load`) around one shared
  `_drain_failed_submission`. Direction is now explicit rather than derived
  from `LoadStoreSpec.medium()`, which was deleted (`c46ced1ee3`, #46544).
* `TransferResult` moved to `kv_offload/base.py` and lost `transfer_type`.
  Nothing in our patches consumed it; `is_load` is still decided by
  `job_id in self._load_jobs`.
* `OffloadingManager.lookup()` returns `LookupResult` (MISS / HIT /
  HIT_PENDING / RETRY) instead of `bool | None` (`bb61177e49`, #46363). Our
  patches do not change `lookup`; the CPU manager's pending-write state is now
  `HIT_PENDING` rather than `None`, which is strictly more explicit and does
  not weaken contract 5.
* `on_schedule_end()` takes a `ScheduleEndContext` (`0fc2512094`, #46450).
  Our patches do not call it; the change only affects test fakes.
* `OffloadingSpec.__init__` now takes a single `OffloadingConfig`
  (`a9531edfa6`, #48150) and `get_handlers` became `get_worker`. This is a
  package-side change (`recipe_persistence.native`), not a patch-side one.
* **EAGLE/MTP tail, behavioural change (inherited, not introduced).** Our old
  `_advance_store_frontiers` and `_has_finished_store_frontier` unconditionally
  subtracted one chunk for an EAGLE group. Upstream's `storable_chunks()` only
  excludes the tail while *decoding* and *not finished* (issue #52735: during
  prefill the draft input for a chunk's last position is the next prompt token,
  and a finished request can no longer suffer spec-token rejection). The port
  defers to `storable_chunks()` so both paths agree, so a finished request now
  stores its final chunk. This is more data stored, never less, and it removes
  the "permanent hole breaks prefix-reuse lookup" failure upstream documents.
  `test_finished_frontier_retains_prompt_cap_and_eagle_tail` asserts the new
  bound (12 chunks, not 11) and carries the reasoning inline.
* **`on_request_finished` timing.** Upstream now fires it in
  `build_connector_meta` for every id in `finished_req_ids` (`:1830-1842`).
  Contract 7 requires more `prepare_store` calls after that point for a held
  request, which would violate upstream's own documented "no more submit-side
  calls" guarantee. The port therefore *defers* the signal for a
  `finish_pending` request to the terminal release in
  `update_connector_output`, matching the pre-port ordering.

## Verification boundary

The CPU suite covers failed/partial submission, mixed-rank completion, delayed
load invalidation, cancellation and shutdown drain, reset/stale/duplicate
outcomes, unequal and empty groups, grouped/single-group recovery, preemption
ordering, store partial/nonprefix decline, explicit skips, EAGLE tails and stale
SWA rows. The 40-test suite includes 14 finished-frontier integration tests:
actual core `update_from_output` stop handling, `_free_request`, HMA finish
handoff, idle connector steps and final core free are executed together. A
2,056-key request with an eight-key admission quantum and one generated token
retains its unsaved suffix through every batch. Zero admission, pressure retry,
all-rank/out-of-order drain, skip-only completion, current SWA table, failed
store retry/abandonment and abort-during-load are covered. These checks execute
the resulting methods, not string assertions; dependency fakes do not establish
actual GPU allocator or stream behavior.

Two of the 40 pass on **pristine** source by design and are not regressions:
`PackagingTests.test_pristine_reproduction_and_apply_verify_reverse` exercises
apply/verify/reverse/refusal on its own temporary copy, and
`RecoveryTests.test_single_group_retains_prefix_recovery` is the guard that
single-group valid-prefix behaviour is *unchanged* by patch 0003. The other 38
fail or error without the patches.

One test from the pre-port suite was deleted and one added.
`test_inner_worker_submission_exception_is_failure` extracted
`OffloadingWorker.transfer_async` from the deleted dispatcher class (0001 hunks
8-9, OBSOLETE). Its replacement,
`test_failed_store_submission_drains_before_synthetic_outcome`, covers the same
contract on the store leg of the new split API — the leg the port introduces,
and the one the surviving load-side test did not reach.

Still required before rollout: full module/import integration, real worker
serialization/aggregation, byte-exact multi-group/rank roundtrip, CUDA stream
and allocator reuse ordering under async scheduling/speculation, actual
canonical cache geometry, coordinator failure behavior, crash durability and
unchanged serving capacity. No benchmark or deployment is triggered here.
See `PORT-NOTES.md` at the branch root for the GPU-smoke risk list.

Source modifications retain upstream SPDX/copyright headers and Apache-2.0
provenance. Patch preambles also identify the upstream contributor copyright.
