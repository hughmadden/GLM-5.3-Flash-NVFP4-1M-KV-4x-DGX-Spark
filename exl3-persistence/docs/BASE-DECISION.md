# BASE-DECISION — R0 deliverable, GLM-5.3-Flash EXL3 TP4 + persistence

Written 10 September 2026 AEST. Sources: campaign `the campaign state log` (authoritative — its later corrections win over any report below), `the campaign plan`, `the campaign handover note`; build-receipts/port artifacts under `$WORK/{reports,port,image,receipts}/`. Every claim carries its source path. Inferences labelled **INFERENCE**; nothing has run on a GPU — every runtime claim is **UNVERIFIED**.

## 1. Decision

**Route B1.** Newer vLLM prebuilt aarch64 wheel + Mia's prebuilt torch-2.13.0 `exllamav3_ext` + Python-level overlay/persistence patches re-derived at the pin.

| Component | Pin | Source |
|---|---|---|
| vLLM | upstream main `83252ea899c6538eaa0c1fb31f28a92c661bbffc` (09-09), one commit behind HEAD, has GLM merge `98ed085` | `the campaign state log` |
| vLLM wheel | `vllm-0.28.1rc1.dev617+g83252ea89-cp38-abi3-manylinux_2_28_aarch64.whl` (`wheels.vllm.ai/nightly/cu130`) | `receipts/build-host/wheels/MANIFEST.md` |
| torch | `2.13.0+cu130`, cp312 aarch64 | `MANIFEST.md`; matches Mia's image pin |
| FlashInfer | `0.6.18.post1` (python/cubin/jit-cache+cu130) — nightly METADATA pins post1, not bare 0.6.18 | `the campaign state log`; `MANIFEST.md` |
| exllamav3 | `0.0.43` @ `c5d9c657966ffeeaa9353f0cc899f18629da4a13`, Mia's prebuilt torch-2.13.0 `.so` (`EXT_MODE=prebuilt`) | `build-receipts/build-host-setup.md §3` |
| CUDA base | `nvidia/cuda:13.0.3-devel-ubuntu24.04` | `the private image build notes` |
| Mia overlay | HEAD `1de5c41a` re-anchored onto `83252ea89` | `notes/OVERLAY-PORT-NOTES.md` |
| Persistence fork | branch `persistence/flash-83252ea89`, `$WORK/` | `the campaign state log` |

**Kill-gate verdict:** PASS-WITH-CONSTRAINTS (§3). **Port shape:** the operator directed re-deriving `NodeLocalDiskOffloadingSpec` onto the new `OffloadingSpec`/`OffloadingWorker` API, keeping direct GPU→disk — rejecting the tiering-connector route because `SecondaryTierManager` runs once in the scheduler process over a `/dev/shm` `SharedOffloadRegion`, single-node only; our ranks are on four separate Sparks (`the campaign state log` "port shape decided"; `notes/port-study.md §D`).

## 2. Why not the alternatives

- **Route A** (Mia's image as-is, day-0 `g487ecf187` ≈ 2026-08-26 PR-head build, Python-only persistence port): viable fallback. Her extracted tree (`trees/extracted-vllm/`) already carries the modern offload subsystem (`offloading/{config,canonical_mapping,events}.py`, `kv_offload/tiering/`), so the persistence port cost would be of the same order as B1 (**INFERENCE** — no anchor audit was run against her tree). Not chosen because B1 adds the post-08-26 offload/glm5next fix wave at near-zero compile cost (`kill-gate-study.md` Q3: draft-group handling `4a806d08ee`, packaging floor `78300cdabf`, hybrid prefix-cache pair), her base is an unreproducible PR-head build with no wheel, and her 4bpw-serving `.so` carries over unchanged either way (§6).
- **Route C** (fork + full ARM64 source build): unneeded — a prebuilt aarch64 nightly wheel exists at the exact pin, and the one genuinely-compiled artifact (`exllamav3_ext`) has a torch-2.13.0-matched prebuilt `.so` already on build-host (`build-receipts/build-host-setup.md §3`; import unverified, §8). Residual use: ext-rebuild fallback only (`EXT_MODE=inimage`, §6).
- **v0.29.0 (PyPI stable):** rejected — does not contain `glm5next` despite postdating the merge (`the campaign state log`; corrects `notes/wheels.md`, §9).
- **Pin `138d137b5b^`** (kill-gate study's own suggestion, pre block→chunk rename): overruled for `83252ea89` because a prebuilt nightly wheel exists only near HEAD (short retention), `138d137b5b^` is at/before the packaging floor `78300cdabf` (glm5next's `__init__.py` missing from the wheel without it), and the rename was already priced as mechanical (`the campaign state log`).
- **Option-2 tier port** (`SecondaryTierManager`/`TieringOffloadingSpec`, the port study's own recommendation): rejected structurally. It runs in the scheduler process over `SharedOffloadRegion` = `/dev/shm/vllm_offload_<engine_id>.mmap` with per-rank offsets (`cpu/shared_offload_region.py:74-105`); ranks on node1–4 cannot map the head's `/dev/shm`, and it double-copies inside GB10 UMA regardless. Option 1 (direct GPU↔disk on `OffloadingSpec`, per-rank NVMe, HTTP coordinator) is the only multi-node-capable shape (`the campaign state log`).

## 3. Kill-gate verdict and launch constraints

**PASS-WITH-CONSTRAINTS** (`notes/kill-gate-study.md`). The `83252ea89` offloading connector admits the glm5_next hybrid: MLA, KDA state, and the kpool tail are all stored/loaded. KDA groups classify as sliding-window (`window = 1 chunk`, `offloading/scheduler.py:136-138`), inside `_lookup_groups`; KDA state is page-padded onto and aliased into the MLA slot tensor (`kv_cache_utils.py:1228-1230`); the offload worker moves only unpadded bytes (~1.09 MB/KDA layer/state at TP4, `worker.py:115-119`).

1. **`--prefix-match-unit = index_kpool` (=4) is mandatory** — `KpoolTailSpec.block_size = index_kpool`, and `offloading/config.py:57-63` asserts every group's `tokens_per_block % tokens_per_hash == 0` (default `tokens_per_hash` ≥128, so `4 % 128` fails init otherwise). (C1)
2. **KDA checkpoints offload only at chunk-aligned boundaries** (`_build_aligned_boundary_store_jobs`, `scheduler.py:1216-1275`); `supports_partial_tail=False` from mixed block sizes; hit windows round down conservatively, not incorrectly. (C2/C3)
3. **Draft-group annotation — fixed in the overlay.** Upstream has DFlash but no MLA slot-sharing (`mla_attention.py:3193-3194`); glm5_next+MTP annotates no eagle group by default, so the connector's post-`4a806d08ee` fallback treats every group as non-draft. The overlay port adds `is_eagle_group=True` on the DFlash2 group, closing this (§5; `notes/OVERLAY-PORT-NOTES.md §4`).
4. **NoPE/`pe_dim==64`** — `fp8_ds_mla` hard-requires `pe_dim==64` (`csrc/libtorch_stable/cache_kernels.cu:925-934`); glm5_next is true NoPE. Satisfied via Mia's Python-side zero-pad (`k_pe.new_zeros((N,1,64))`) — **no csrc rebuild** (`OVERLAY-PORT-NOTES.md §5`). sm120/121 offers only `TRITON_MLA`/`FLASHINFER_MLA_SPARSE_SM120`; the latter needs `index_topk==2048` and forces `fp8_ds_mla`.

**Disk-budgeting hazard:** `platforms/interface.py:813-820`'s pre-alignment MLA page probe omits `state_content_bytes`, computing 576 B/token under `fp8_ds_mla` instead of the real 656 B/token (~12% under). Use the measured table instead (656 B/token/MLA layer; ~1.09 MB/KDA layer/state at TP4).

## 4. Persistence port plan

**Drift** (`port-study.md`): base `ab6660699` (06-19) is **3,272 commits** behind `83252ea89`. Three PRs cause ~90%: `a9531edfa6` (#48150, `OffloadingSpec` ctor signature), `f237e16b41` (#45053, `OffloadingHandler`/`get_handlers`→`OffloadingWorker`/`get_worker`), `138d137b5b` (#52615, `block`→`chunk` rename). `kv_offload/worker/worker.py` is deleted; `TransferResult` moved to `base.py:541`, lost `transfer_type`; `lookup` now returns `LookupResult`.

**Our 4 patches, 50 hunks/8 files: 4 ABSORBED, 44 STILL-NEEDED, 2 OBSOLETE** (`port-study.md §B`). Nothing upstream implements our contracts — paths moved/renamed. Per-patch: 0001 (11 hunks, S), 0002 (20 hunks, L — `transfer_async`→`submit_store`/`submit_load` split), 0003 (4 hunks, S, near-verbatim — `TODO (davidb)` anchor at `sched/scheduler.py:3071` byte-identical), 0004 (15 hunks, M, composes with upstream's new `finished_signaled`).

**Re-derived:** the 4-patch series onto `OffloadingSpec`/`OffloadingWorker`, plus `native.py` (subclasses the old API, `native.py:619-625,757-760`) and `handlers.py` (two-yield `get_handlers` collapses to one `OffloadingWorker`). `geometry.py`/`storage.py`/`coordinator*.py`/`http_rpc.py` are unaffected and carry over as-is (`port-study.md §D`).

**Acceptance bar:** the re-pointed 40-test native contract suite (`test_native.py`, AST-extraction, no vLLM import needed) must pass against the ported tree and fail against pristine, proving the patches load-bearing; plus the 237-test package suite via the `fake_vllm` stub, repointed for the same renames.

**Status: DONE.** (Updated from "in progress" — the campaign state log advanced into R1 while this was written; later corrections win.) Fork branch `persistence/flash-83252ea89`, worktree `fork-worktree`: commit `3355aa1` (4 patches re-derived: 7 files, 44 hunks carried/4 absorbed/2 obsolete; `apply.py` refuses on hash or anchor mismatch) + `d031761` (`recipe_persistence` on the new API). **Tests:** native contract 40/40 on ported tree; same suite `--no-apply` against pristine = 4 fail + 34 error (proves patches load-bearing) + 2 pass by design; package suite 240/240. Top runtime risks (`PORT-NOTES.md`): `CanonicalPageMapping` writer semantics fenced by refusal; `replicated_layout` non-writer ack refused; chunk/block sizing mismatch vs `prefix_match_unit` → key-identity miss, not corruption; non-attention groups mirrored from `offloading/worker.py`, NOT skipped; private coupling to `CPUOffloadingWorker._store_handler`/`_load_handler` fails closed on rename. One inherited behavioural delta (upstream `storable_chunks()` #52735): finished requests now store their final chunk (more stored, never less).

## 5. Overlay port

**15/15** vLLM-targeting `patch_*.py` APPLIED at `83252ea89`, exit 0; 16 files touched (14 modified + 2 added), `py_compile`-clean; patch (3,847 lines) round-trips byte-exact (`notes/OVERLAY-PORT-NOTES.md`). 8 unchanged from Mia's source; **7 re-anchored** (cosmetic: `patch_glm_eagle3.py`, `patch_kpool_tail_slotmap.py`, `patch_indexer_workspace.py`, 3 of `patch_glm5_drafter_group.py`'s edits, `patch_dflash2.py`'s registry/dispatch edits downgraded to fail-closed asserts); **2 real reworks**: `patch_adaptive_k.py` folds in upstream's new `strip_speculative_padding` call, `patch_dense_fp8.py`'s injected conditional moves inside upstream's new `try/finally`.

**Drafter-group fix:** `_get_kv_cache_groups_glm5_next` early-returns before eagle annotation runs and neither upstream detection rule can fire for this group — added `is_eagle_group=True` on the DFlash2 group, closing constraint 3 (§3.3).

**Heredoc findings** (NoPE/SM120 zero-pad lives in Mia's Dockerfile heredocs, not `patch_*.py`): of 17 anchors at the pin, **4 drifted, 2 dropped** — corrects the port notes' original "3 drifted, 2 dropped" (§9). One dropped anchor (`platforms/cuda.py` sm120 page-size fix) is upstreamed (`cuda.py:431-437`) and must NOT be re-applied. The other three re-anchor cleanly (2× `expand_pools_and_append_tail` calls now under `kpool_ops.`; a 4th found during head validation — `supports_dense_mha_prefill = False` duplicate-attribute risk — downgraded to a fail-closed presence assert).

**Capability gating:** `patch_dflash2.py` no longer blind-overwrites — upstream at this pin ships a newer DFlash2 with a fixed Gumbel draw and `draft_logits_spec` hook Mia's vendored copies predate. The script checks both required APIs and DFlash2's presence: if both present, keeps upstream's files and refuses loudly only in the ambiguous case (`OVERLAY-PORT-NOTES.md §2,§6`).

## 6. Build budget & assets

**build-host:** x86_64 build host, `/srv` 546 GB free (10 Sep), binfmt/qemu-aarch64 verified (`docker run --rm --platform linux/arm64 ubuntu:24.04 uname -m` → `aarch64`), task root `$STAGE/{images,trees,receipts,build,wheels}` (`build-receipts/build-host-setup.md §1`).

**Wheels** (`receipts/build-host/wheels/MANIFEST.md`):

| File | sha256 (prefix) | Used in image |
|---|---|---|
| `vllm-0.28.1rc1.dev617+g83252ea89-cp38-abi3-manylinux_2_28_aarch64.whl` (304,399,241 B) | `8ccc389b80e4abac…` | yes |
| `torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl` (427,082,459 B) | `5ebd552c887e707c…` | yes |
| `flashinfer_python-0.6.18.post1-py3-none-any.whl` (18,348,022 B) | `adb6d2952b471311…` | yes (vLLM nightly pin) |
| `flashinfer_cubin-0.6.18.post1-py3-none-any.whl` (1,565,107,285 B) | `bbacb5b8bbf429e4…` | yes |
| `flashinfer_jit_cache-0.6.18.post1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl` (1,130,243,484 B) | `860ec5ce11a1d686…` | yes |
| `flashinfer_{python,cubin,jit_cache}-0.6.18` (bare) | in `SHA256SUMS` | **no** — wrong pin, retained only |

Sources: `receipts/build-host/wheels/MANIFEST.md` (first two + bare 0.6.18), build-host `$STAGE/wheels/SHA256SUMS` + `post1-fetch.log` (post1 set; jit-cache/cubin hashes match the flashinfer.ai index fragments).

**Ext decision:** `EXT_MODE=prebuilt` default — Mia's `.so` (105,811,592 B, `sm_121a` confirmed via `strings`, built against torch 2.13.0+cu130, the exact pin here) needs no rebuild (`build-receipts/build-host-setup.md §3`). Fallback `EXT_MODE=inimage`: emulated build, 68 TUs, est. **~15–30 min** under qemu (Probe A: 18–71 s/TU, `build-receipts/ext-build-probe.md`). **Cross-compile rejected**: 9–14× faster per-TU (Probe B, `cuda-cross-sbsa-13-0`) but `torch.utils.cpp_extension` assumes host-arch build == target-arch and needs a custom CC/CXX/path-override driver — un-built engineering, out of R0 scope.

**Image layer order:** persistence patches apply **first** on the pristine tree — `apply.py` hash-pins seven files incl. `sched/scheduler.py`, which two overlay scripts also edit; persistence-first is the only order that works (validated on head: both apply cleanly on the persistence-patched file, `the private image build notes`). Then: NoPE/SM120 heredoc → exl3 registration + ext → Python overlay (14 scripts) → observer (last, lands on final tree) → gates → receipts/launcher.

**Tag/stamp:** `glm53-flash-exl3-tp4-persist:20260910-83252ea89`, recipe stamp `78ab74cee920d1eb`. R1 build STARTED on build-host 2026-09-10 (`build.sh`, log `receipts/build-20260910-101500.log`) — R1 in progress per `the campaign state log`, not an R0 finding, not re-litigated here.

**Receipts:** `receipts/build-host/` (binfmt, image inspect, filelist, versions, sha256, qemu probe); `receipts/build-host/wheels/` (MANIFEST.md, SHA256SUMS); build-host `$STAGE/receipts/`.

## 7. Fleet readiness (S1 prerequisites already met)

**Spark staging** (`notes/staging-audit.md`): EXL3 4bpw (175,715,854,754 B, 120 shards) and DFlash2 (2,342,175,855 B) byte-verified complete on all four nodes. node0: local at `/srv/node0-share/huggingface/hub/`. node1: local at `~/.cache/huggingface/hub/`. **node2/node3:** the real target path is a live read-only NFS mountpoint from node0 — genuine local copies were rsynced to `~/.cache/huggingface/hub-local-staging/<model>/` instead (byte-exact). The report's "operator must unmount + `mv`" was corrected: `the campaign state log` says no unmount is needed — the S1 launcher simply points S3/S4 at `hub-local-staging` (or sets `HF_HOME`); the staged copies are the goal state as-is (§9).

**Tony playbook items adopted** (`build-receipts/tony-playbook.md`): unconditional page-cache flusher for the whole boot (a threshold-triggered flusher silently starves the NVRM allocator); the memory ritual (swap present, `swappiness=0`, does not survive reboot); worker-first launch (3→2→1→0 — a fresh rank rendezvousing with a dying one hangs); image-ID (not tag) verification on all four nodes; the gate suite (28–32K-token prompt AND ≥100 decoded tokens AND 3× concurrent prefills AND `/health` only, never `/v1/models`).

**Fleet fabric:** node0–4 LAN `192.0.2.{162,157,216,234}`, fabric `198.51.100.{1,2,3,4}`, HCA `rocep1s0f1`, NIC `enp1s0f1np1` all four, RoCEv2 GID index **5/5/3/3**.

**Two Mia TP4 gaps fixed** (`notes/upstream-repo.md §6`): (1) her `start-tp4.sh` env-forward loop omits `EXL3_FAT_GROUPED`/`GLM53_ADAPTIVE_K*`/`GLM53_DENSE_FP8` — ours forwards them to all four ranks; (2) her per-worker CX7 IF/HCA/GID default to one shared value — ours are genuinely per-rank, since our measured GIDs differ (5/5/3/3) and a wrong/unset GID hangs NCCL silently ~20 min at 96% GPU/22 W.

## 8. Open items / UNVERIFIED until GPU

- The wheel's `.py` sources vs the git checkout the persistence manifest was hashed against — not yet checked against the actual nightly wheel contents.
- The prebuilt `exllamav3_ext.so` against *our* built torch wheel — version match confirmed, import/symbol load not yet exercised on our image.
- The entire NoPE/SM120 heredoc at runtime — never applied to the overlaid tree; correctness only ever shown on Mia's own base image.
- The whole build under qemu beyond a trivial single-file compile (0.47 s data point only).
- Combined offset correctness of the observer patch after all 15 overlay scripts run for real (both touch `kv_cache_utils.py`; proven individually, not together).
- `--prefix-match-unit 4` interacting with the persistence connector's group assertions at engine init — no engine has started with it.
- Every serving number in `env.tp4.fleet` (`MAX_MODEL_LEN=850000`, `GPU_MEM_UTIL=0.85`, `EXL3_TEMP_ROWS_FUSED=32`, `GLM53_SPINWAIT_MS=16`) — Mia's TP2 defaults, a starting point at TP4, not a qualified config.
- Persistence staging bounds (`staging_bytes`/`staging_rows`/`max_pending_keys`) — conservative initial values pending geometry/headroom qualification.
- The four GID indices (5/5/3/3) — not stable across link bounces/reboots; re-verify with `./start-tp4.sh preflight` before every launch.
- The 1 TB/rank persistence roots — named, not created; local-NVMe backing unconfirmed on any node.
- Drafter exact-fit vs padded-slot-share branch selection, derived `num_blocks`, whether `_reshape_kv_cache` accepts the new offset-based standalone drafter tensors.
- Offloading-connector behaviour with the drafter group now annotated: volatile-tail pop, finish-time lift, `supports_partial_tail`/`store_horizon_chunks` under the glm5 hybrid's mixed block sizes.
- Upstream's newer DFlash2 vs Mia's vendored DFlash2 on GB10 hardware (deliberate semantic choice, §5, never run).

## 9. Corrections of record

- **`notes/wheels.md`'s "0.29.0 suffices" verdict was WRONG.** v0.29.0 does not contain `glm5next` despite postdating the merge. Corrected in `the campaign state log`.
- **Mia repo stale-snapshot HEAD is `3021f24c88a0904c768c46ff22a508407e31360a`, not `6599585`** — `6599585` doesn't exist in the stale snapshot's own history; it's a later commit only reachable in the current upstream clone (`notes/upstream-repo.md §2`).
- **"No `exl3_fat_moe` ext in image" was wrong.** The prebuilt Mia `.so` DOES contain the fat-MoE kernels — corrected in `the campaign state log`.
- **`port-study.md`'s option-2 (`SecondaryTierManager`) recommendation is REJECTED**, structurally: it runs over a `/dev/shm`-backed region, single-node only; our four Spark ranks cannot share that (`the campaign state log`).
- **FlashInfer version mismatch, wheels agent vs actual pin.** The wheels worker grabbed plain `0.6.18`; the vLLM nightly's own METADATA pins `0.6.18.post1`. Post1 wheels fetched separately and are what the image installs.
- **`notes/staging-audit.md`'s "operator must unmount + mv" was unnecessary.** `the campaign state log`: the S1 launcher can point S3/S4 at `hub-local-staging` directly, no unmount required.
- **`PORT-NOTES.md` (persistence fork branch) claim "GLM-5.3-Flash has no Mamba groups" is WRONG**, per the kill-gate study: KDA groups ARE `MambaSpec`-classified and aliased into the MLA slot tensor (§3); the partial-tail/CoW path WILL be exercised by Flash and must be covered by the S2 ladder. Corrected on the branch in commit `f163c39`.
