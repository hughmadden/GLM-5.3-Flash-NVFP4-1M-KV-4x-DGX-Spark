# Self-tests ported to vLLM `83252ea89` — changed-assertion log

Companion to `../../../notes/OVERLAY-PORT-NOTES.md` (referred to below as **the
port notes**, by section). Mia's self-tests in this directory assert the
*source text* of her day-0 vLLM tree (`0.1.dev20051+g487ecf187`). The overlay
was re-anchored onto upstream `83252ea89`, so some of those assertions no
longer matched and failed the image build.

| | |
|---|---|
| Ported against | `$WORK/port/vllm-83252ea89-overlay` (pristine pin + all 15 overlay scripts applied) |
| Negative control | `$WORK/port/vllm-83252ea89` (pristine pin, no overlay) |
| Host | head, no GPU, **no `torch`/`vllm` importable** — every check below is text/AST-level |
| Date | 2026-09-10 AEST |

Rules followed: keep each test's intent; cite the port-note section in a
comment next to any assertion whose *contract* changed; never weaken a check
into `pass`/skip. Every new or re-anchored snippet was confirmed present in the
overlaid tree **and absent from the pristine pin** (`grep -c`, table at the
bottom) so the checks still discriminate.

---

## 1. `test_exl3_overlay.py`

### 1.1 Structural changes (no assertion weakened)

| Change | Why |
|---|---|
| Added module-level `SITE = Path(os.environ.get("GLM53_SITE", "/usr/local/lib/python3.12/dist-packages"))` and `VLLM = SITE / "vllm"`; every `Path("/usr/local/.../vllm/…")` in `_check_dflash2` now reads `VLLM / "…"`. | The file had no path override at all. `GLM53_SITE` is the **same variable name and meaning** already used by `verify/verify_overlay_static.py`, and its default is byte-identical to the old hardcoded path, so the Dockerfile step is unchanged. Lets the source-text half run on a host with no vLLM installed. |
| Split `_check_dflash2()` into `_check_dflash2_imports()` (class attrs + registry, needs `import vllm`) and `_check_dflash2_sources()` (all `read_text` assertions). | Only the import half needs a live vLLM. Nothing was dropped: the union of the two functions is the original set of assertions plus the additions in §1.3. |
| `main()` gained `GLM53_SELFCHECK_IMPORTS` (default `"1"`, i.e. ON). At `0` it skips the five import-level checks, prints a loud line naming the tree it *is* checking, and still runs `_check_dflash2_sources()`. `EXL3_SELFCHECK_GPU=1` combined with `GLM53_SELFCHECK_IMPORTS=0` now raises `SystemExit` instead of failing obscurely inside the GPU path. | Same fail-closed-by-default shape as the pre-existing `EXL3_SELFCHECK_GPU` knob. The Dockerfile does not set it, so the image gate is exactly as strict as before. |
| Removed one duplicate `read_text()` of `qwen3_dflash.py` (the trailing `src = Path(...)` re-read used only for the `is_causal` assert); that assert now uses the `qwen` text already read at the top of the function. | Dead I/O, same assertion. |

### 1.2 Stale assertions re-anchored

Both are the cosmetic loop/comprehension-variable rename recorded in **port
notes §3(a)** (`k, v, s` → `name, spec` in the glm5 helpers). Both are the
assertions that actually broke the build (`test_exl3_overlay.py:1308`).

| Old expectation | New expectation | Reason |
|---|---|---|
| `assert "type(v) is SlidingWindowSpec" in kv` | Exact match on the whole `draft_specs = { name: spec for name, spec in kv_cache_spec.items() if type(spec) is SlidingWindowSpec }` comprehension | §3(a) rename. Re-anchored on the **whole comprehension**, not just the predicate, so a future rename cannot quietly widen what counts as a drafter layer — the `type(...) is` exactness (which keeps `KpoolTailSpec` subclasses out) is what the assertion was protecting. |
| `assert "s.block_size != 64 or s.page_size_padded != mla_page" in kv` | `assert "                spec.block_size != 64 or spec.page_size_padded != mla_page\n" in kv` | §3(a) rename. Indentation pinned so it matches the padded-slot-share validation site only. |

### 1.3 New assertions — the changed contracts

| Contract | Assertions added | Note ref |
|---|---|---|
| `patch_dflash2.py` is **capability-gated** and deliberately does *not* install Mia's two vendored DFlash2 sources on a tree that ships newer upstream ones. The old test could not tell upstream's copy from the vendored shim. | Gate preconditions: `def get_top_k_tokens(` in `model_executor/layers/logits_processor.py`; `def gumbel_noised_argmax(` **and** `IS_DRAFTING` in `v1/worker/gpu/sample/gumbel.py`. Provenance of the installed files: `get_top_k_tokens(` in `model_executor/models/qwen3_dflash2.py` (the vendored copy's own docstring says `LogitsProcessor` has no such API); `IS_DRAFTING=True` **and** `def draft_logits_spec(` in `v1/worker/gpu/spec_decode/dflash2/speculator.py` (the vendored speculator has neither). Plus `dflash2/__init__.py` exists. | **§2**, **§6.1** |
| The port added one edit not in Mia's overlay: the DFlash2 drafter group is annotated `is_eagle_group=True`, because `get_kv_cache_groups()` early-returns the glm5 path before `_annotate_eagle_groups()` runs, which left the offloading connector treating the drafter's volatile tail as stable. | Exact match on the three-line `draft_group = KVCacheGroupSpec(list(new_draft_specs), draft_uniform, is_eagle_group=True)` construction. | **§4** |

Verified deliberately **not** changed: `assert "self.decoder_layer_cls(" in qwen`
and `assert 'getattr(config, "is_causal", None)' in qwen` both pass on the
pristine pin too, because §2 records those two edits as having become
already-upstream no-ops. They are retained exactly as Mia wrote them — they are
now the test-side mirror of `patch_dflash2.py`'s `require_once()` fail-closed
presence assertions, and a future upstream revert must still break the build.

### 1.4 Checks that can only be verified in-image / on a GPU

| Check | Status |
|---|---|
| `_check_quant_registry`, `_check_tp_shard`, `_check_ext_arch`, `_check_e2_diag_static`, `_check_dflash2_imports` | Need `import torch`/`vllm` (and `cuobjdump` for the arch check). The first four **already printed OK in the failing build**, so they are not stale. `_check_dflash2_imports` is the unchanged import half of the old `_check_dflash2`; every class attribute and the registry tuple it asserts were confirmed present in the overlaid tree by AST/grep (`decoder_layer_cls`/`model_cls` on all four classes; `registry.py:629`). |
| `_check_gpu_gemm` and everything it calls (`_check_fat_kernel`, `_check_fused_vs_loop`, `_check_fused_fat_and_row_tile`, `_check_mixed_thin_fat`, `_check_e2_diag`, `_check_apply_expert_map`, `_check_fused_cudagraph`, `_check_grouped_tables`, `_check_grouped_fat`) | GPU-only, skipped in the build (`EXL3_SELFCHECK_GPU=0`). **Not touched, and no port drift is possible**: they drive `overlay/exl3.py`, which the Dockerfile `COPY`s verbatim and which the port did not modify. All 23 symbols they import from it, plus the one `exl3mod._FAT_MOE_EXT_CACHE` attribute access, were confirmed present by AST. Their numeric tolerances can only be exercised on a Spark. |

---

## 2. `test_ablit.py` — **no changes needed**

Audited assertion by assertion. It has **zero** coupling to the vLLM tree: it
loads `overlay/ablit_runtime.py` directly by path (an APPLIED, byte-identical
overlay script — port notes §1 row 12) and reads the `context/ablit/` assets.
Nothing it asserts moved in the port.

Verified on head without torch:

* the eleven `ablit_runtime` symbols it uses (`DIRECTION_FILES`,
  `load_direction`, `parse_layers`, `AblitError`, `apply_to_o_proj`,
  `apply_ablit`, `maybe_apply`, `load_transplant_tensors`, `apply_transplant`,
  `_tp_world`, `_tp_rank`) all exist, by AST;
* `check_recipe_integrity` against the shipped `context/ablit/LAYER_MAP.json`:
  `hidden_size=4096`, `num_hidden_layers=45`, `o_proj_dtype="bfloat16"`,
  `quant_ignore_includes_o_proj=True`, 46 entries, 30 `edit` roles,
  `layers[45].role == "mtp-edit"`, `published.method` and `(min,max)=(15,45)`,
  every `o_proj` suffix and every `shard` filename pattern — **all hold**;
* `check_direction_files` against the two shipped `.pt` files, read with a
  stubbed unpickler (no torch): `dealign.source ==
  "dealign-oproj-svd-L15-35-39-43-45"`, `bf_oproj.source ==
  "blackfrost-oproj-svd"`, `bf_oproj.alpha_ref == 3.0`, and both `directions`
  storages are 16384 bytes = 4096 × fp32 — **all hold**.

The remaining checks are pure-torch numerics (orthogonalization identities,
bf16 roundtrip, TP shard equivalence, module walk, transplant byte-copy). They
need `import torch`, are unaffected by the port, and are expected to pass
in-image exactly as they did on Mia's base.

---

## 3. `verify/verify_overlay_runtime.py` — **no changes needed**

Two independent reasons:

1. It runs **immediately before** `test_exl3_overlay.py` in the same
   `RUN_IMPORT_CHECKS` step (`configs/build/Dockerfile`), and the build reached
   `test_exl3_overlay.py:1308`. It passed.
2. Every source-text fact it asserts at import level is also asserted textually
   by `verify/verify_overlay_static.py`, which runs earlier still (Layer 8,
   ungated) and also passed: the SM120 NoPE strings (`supports_dense_mha_prefill
   = False`, `def do_kv_cache_update(`, `self.rope_pad = 64`, `pad(q, (0,
   self.rope_pad))`, `qk_rope_head_dim=self.kernel_qk_rope_head_dim`,
   `return_valid_counts=True`, `sparse_mla_top_k=sparse_topk_capacity`,
   `seq_lens=topk_lengths`, the `masked_fill_`, and the `attn_metadata.topk_tokens`
   prohibition), `get_supported_kernel_block_sizes() -> [64]`, the exl3
   registration, `startup_observer.py` at the site root, and
   `recipe_persistence/native.py`.

Note for future porters: those SM120 facts come from the **Dockerfile heredoc**,
not from the 15 overlay scripts (port notes §5), so they are **not** present in
`port/vllm-83252ea89-overlay` and this verifier cannot be exercised against that
stand-in even if torch were available.

---

## 4. Re-verification of the two previously-fixed tests

Both still pass on head, unchanged, at three different targets:

| Test | default (no env) | vs overlaid tree | vs pristine pin |
|---|---|---|---|
| `test_kpool_tail_slotmap.py` | exit 0 | exit 0 (`GLM53_BLOCK_TABLE_PY_SRC=…-overlay/v1/worker/block_table.py`) | n/a (fixture *is* the pristine file) |
| `test_indexer_workspace.py` | exit 0 (21 tests) | exit 0 (`GLM53_INDEXER_BACKEND_PY_SRC=…-overlay/v1/attention/backends/mla/indexer.py`) | — |

`test_indexer_workspace.py`'s `test_live_copy_if_present` silently returns when
the installed file is absent — that is a pre-existing, self-named `_if_present`
opt-in, and it *does* run in the image. Feeding it the overlaid tree above is
what exercises that path on head.

---

## 5. Incidental findings (no edits made)

* **The overlaid tree is not a byte-exact stand-in for the image in one place.**
  `port/portrun/run_port.sh:68` exports `GLM53_DENSE_FP8=on`, while the
  Dockerfile runs `patch_dense_fp8.py` with the variable unset (`off` — the
  script's default). So the overlaid tree's `models/glm5next/nvidia/{kda,model}.py`
  carry the `# [glm53-dense-fp8]` marker and the image's do not. Consequence:
  `test_dense_fp8_patch.py` **fails** if pointed at the overlaid tree ("off
  leaves constructors alone") and **passes** against the pristine pin, which is
  the correct stand-in for the image's pre-test state. This is a stand-in
  fidelity gap, not a stale test — the test is right and passes in-image.
* The other four host-only tests in the Layer 8 block already have
  `GLM53_*_PY_SRC` overrides but no default that exists on head. Confirmed passing
  against the overlaid tree with the override set:
  `test_scheduler_decode_floor.py` (`GLM53_SCHEDULER_PY_SRC`),
  `test_hybrid_prefix_hit.py` (`GLM53_KV_COORDINATOR_PY_SRC`),
  `test_adaptive_k_patch.py` (`GLM53_SCHEDULER_PY_SRC` +
  `GLM53_CUDAGRAPH_UTILS_PY_SRC`). No edits were needed.
* The startup-observer patch adds call sites in `v1/core/kv_cache_utils.py` at
  lines 2309 and 2597 — well clear of the drafter-group region (~1180–1550) that
  `_check_dflash2_sources` reads, and it only inserts lines. The `kv.split(...)`
  window assertions are unaffected by it.

---

## 6. Discrimination table (overlay vs pristine)

Every overlay-specific snippet asserted by `_check_dflash2_sources`, checked
with `grep`-equivalent substring counts:

| Snippet | overlaid tree | pristine pin |
|---|---|---|
| `DFLASH2-DRAFTER-GROUP` | present | **absent** |
| `draft_specs` comprehension (§3a form) | present | **absent** |
| `compact_block = 64` | present | **absent** |
| `padded slot-share block=%d` | present | **absent** |
| `spec.block_size != 64 or spec.page_size_padded != mla_page` | present | **absent** |
| `draft_group = KVCacheGroupSpec(... is_eagle_group=True)` | present | **absent** |
| `PADDED SLOT-SHARE:` | present | **absent** |
| `draft_kv = "auto"` | present | **absent** |
| `self.decoder_layer_cls(` | present | present *(already-upstream, §2 — by design)* |
| `getattr(config, "is_causal", None)` | present | present *(already-upstream, §2 — by design)* |

Whole-file negative control: with `GLM53_SITE` pointed at the pristine pin the
file exits **1** on `assert 'draft_kv = "auto"' in dflash_utils`. It is
fail-closed.

---

## 7. Local run results

```
$ cd image/context/tests

$ python3 test_kpool_tail_slotmap.py                      # exit 0
$ python3 test_indexer_workspace.py                       # exit 0  (21 tests)
$ python3 test_suppress_stops.py                          # exit 0
$ python3 test_xgrammar_termination.py                    # exit 0
$ python3 test_spinwait_patch.py                          # exit 0  (9 tests)

$ V=$WORK/port/vllm-83252ea89-overlay
$ P=$WORK/port/vllm-83252ea89
$ GLM53_SCHEDULER_PY_SRC=$V/v1/core/sched/scheduler.py \
    python3 test_scheduler_decode_floor.py                # exit 0
$ GLM53_KV_COORDINATOR_PY_SRC=$V/v1/core/kv_cache_coordinator.py \
    python3 test_hybrid_prefix_hit.py                     # exit 0
$ GLM53_SCHEDULER_PY_SRC=$V/v1/core/sched/scheduler.py \
  GLM53_CUDAGRAPH_UTILS_PY_SRC=$V/v1/worker/gpu/cudagraph_utils.py \
    python3 test_adaptive_k_patch.py                      # exit 0
$ GLM53_KDA_PY_SRC=$P/models/glm5next/nvidia/kda.py \
  GLM53_GLM5_MODEL_PY_SRC=$P/models/glm5next/nvidia/model.py \
    python3 test_dense_fp8_patch.py                       # exit 0  (see §5)

# site stand-in: <scratch>/site/vllm -> $V
$ GLM53_SITE=<scratch>/site GLM53_SELFCHECK_IMPORTS=0 EXL3_SELFCHECK_GPU=0 \
    python3 test_exl3_overlay.py                          # exit 0
      GLM53_SELFCHECK_IMPORTS=0 — skipped every torch/vLLM import check; ...
      dflash2 overlay OK
      EXL3_SELFCHECK_GPU=0 — skipped GPU GEMM
      glm53 EXL3 overlay verify OK

# negative control: <scratch>/site-pristine/vllm -> $P
$ GLM53_SITE=<scratch>/site-pristine GLM53_SELFCHECK_IMPORTS=0 EXL3_SELFCHECK_GPU=0 \
    python3 test_exl3_overlay.py                          # exit 1, as required
```

Not runnable on head (no torch/vllm): `test_ablit.py`, and the import half of
`test_exl3_overlay.py` (`GLM53_SELFCHECK_IMPORTS=1`), and
`verify/verify_overlay_runtime.py`. Reasoning for why each is expected to pass
in-image is in §1.4, §2 and §3.
