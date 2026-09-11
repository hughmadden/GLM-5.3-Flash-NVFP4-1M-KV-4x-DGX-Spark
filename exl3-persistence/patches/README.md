# patches/ — the image's source deltas, as reviewable text

Everything under this directory is applied **inside the image build**, in a
load-bearing order. Nothing here is a standalone installer; read this index and
[`../configs/build/README.md`](../configs/build/README.md) before applying
anything.

The whole chain targets one upstream pin:
`vllm@83252ea899c6538eaa0c1fb31f28a92c661bbffc`. The persistence series declares
that pin in its own `manifest.json`; the overlay scripts re-anchor to it.

## Order (enforced by the build)

1. **`persistence-flash/` first, on a pristine tree.** `apply.py` hash-pins all
   seven target files against `manifest.json`'s BEFORE sha256 and refuses on any
   mismatch. This ordering is **the single most load-bearing decision in the
   build**: `vllm/v1/core/sched/scheduler.py` is one of the seven, and two overlay
   scripts (`patch_scheduler_decode_floor`, `patch_adaptive_k`) also edit it.
   Persistence first, overlay second, is the only order that works.
2. **`vllm/`, `overlay/`, `ablit/`** — the anchored build patches and the
   quantization/kernel overlay.
3. **`observer/`** — the metadata-only startup observer call-site port.

## Index

| Path | Applies to | What it is |
|---|---|---|
| `persistence-flash/0001-drained-outcome-api.patch` | `vllm@83252ea89` | Persistence patch 1 of 4 — drained-outcome API. |
| `persistence-flash/0002-offloading-failure-frontiers.patch` | `vllm@83252ea89` | Patch 2 of 4 — offloading failure frontiers. |
| `persistence-flash/0003-grouped-recovery-ordering.patch` | `vllm@83252ea89` | Patch 3 of 4 — grouped recovery ordering. |
| `persistence-flash/0004-finished-store-frontier.patch` | `vllm@83252ea89` | Patch 4 of 4 — finished-store frontier. |
| `persistence-flash/apply.py` | — | Guarded applier: hash-pinned BEFORE/AFTER anchors, refuses on drift, `check`/`apply`/`verify` subcommands. |
| `persistence-flash/manifest.json` | — | Per-file BEFORE hashes and the pinned `upstream_commit`. |
| `persistence-flash/test_native.py` | — | The re-pointed native contract suite (AST-extraction; no vLLM import needed). Proves the patches are load-bearing: it passes against the ported tree and fails against pristine. |
| `persistence-flash/README.md` | — | The series' own description of the four contracts. |
| `vllm/patch_offloading_ambient_config.py` | `vllm@83252ea89` | **Required.** Wraps offloading-spec construction in `with set_current_vllm_config(vllm_config)`. Fail-closed: target file must hash to BEFORE, both anchors must appear exactly once, result must hash to AFTER, re-run is a no-op. Without it the scheduler path raises `ValueError: persistence requires an active vLLM cache configuration`. |
| `vllm/patch_nope_sm120.py` | `vllm@83252ea89` | NoPE-MLA handling for SM12x (zero-pad into the 576-wide `FLASHINFER_MLA_SPARSE_SM120` record; upstream still carries the `pe_dim==64` assert class of failure). |
| `vllm/patch_register_exl3_quant.py` | `vllm@83252ea89` | Registers the EXL3 quantization method. |
| `overlay/` | `vllm@83252ea89` | The 15 vLLM-targeting overlay scripts plus the EXL3 kernel sources (`exl3_fat_gemm.cu/.cuh`, `exl3_fat_moe.cu/.cuh`), the EXL3 quantization module, the DFlash2 speculator, and the abliteration runtime overlay. `patch_*.py` only; no fork is vendored. |
| `ablit/` | — | The o_proj transplant fetcher and its layer map. **The ablation tensor blobs (`.pt`) are NOT distributed** — regenerate them from the public donor/method repos named in the scripts. |
| `observer/` | `vllm@83252ea89` | The startup-observer call-site port: `.patch`, the hash-pinned `.manifest.json` (4 stages / 15 offsets), and `startup_observer.py`. The launcher verifies the manifest digest against the image at preflight. |
| `tests/` | — | Offline, CPU-only harness for the overlay: text/AST-level patch tests (`test_*.py`) and `PORT-TESTS-NOTES.md`, the port record. Some tests need torch/vLLM and are marked as such. |
| `verify/` | — | In-image overlay verification: `verify_overlay_static.py` (no CUDA) and `verify_overlay_runtime.py` (import-level check run inside the built image). |

## Do not double-apply

The persistence series and the overlay are **already baked into every image built
from `configs/build/Dockerfile`**. The overlay port applied 15/15 scripts at the
pin, touching 16 files (14 modified, 2 added), `py_compile`-clean, with the
resulting patch round-tripping byte-exact. Re-applying either chain to an
already-built tree will fail closed on a hash mismatch — that is the intended
behaviour, not a defect to work around.

## Verification without an engine

```sh
# persistence series: hash-pinned check only, changes nothing
python3 persistence-flash/apply.py check

# native contract suite against a ported tree (40 tests, AST-extraction)
python3 persistence-flash/test_native.py

# overlay static checks (no torch, no CUDA)
python3 verify/verify_overlay_static.py
```

Absolute paths inside the scripts (`$WORK`, `$STAGE`, `pinned-vllm`) are
placeholders from sanitization — see [`../SANITIZATION-AUDIT.md`](../SANITIZATION-AUDIT.md).
Set them to your own checkout and build tree.
