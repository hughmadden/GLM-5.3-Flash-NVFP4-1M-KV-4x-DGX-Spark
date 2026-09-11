#!/usr/bin/env python3
"""Static (import-free) verification of the whole baked overlay.

Mia's equivalent build-time check (``Dockerfile`` lines 320-367) does
``from vllm... import FlashInferMLASparseSM120Impl`` and inspects live source.
This image is cross-built for linux/arm64 on an x86_64 host, so every ``RUN``
executes under qemu-user emulation; importing ``torch``/``vllm`` there is slow
and has an independent failure mode that has nothing to do with the overlay.
So this verifier asserts the *same* facts by reading and ``compile()``-ing the
installed sources, with no imports at all.

DEVIATION, recorded deliberately: this is weaker than Mia's check in exactly one
way -- it proves the source text is right and syntactically valid, not that the
resulting classes import and expose the expected attributes. The import-level
check is run separately by ``verify_overlay_runtime.py`` (build-arg gated,
default on) and again on-host at first launch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SITE = Path(os.environ.get("GLM53_SITE", "/usr/local/lib/python3.12/dist-packages"))
VLLM = SITE / "vllm"

failures: list[str] = []


def read(rel: str) -> str:
    path = VLLM / rel
    text = path.read_text()
    try:
        compile(text, rel, "exec")
    except SyntaxError as error:  # pragma: no cover - build gate
        failures.append(f"{rel}: SyntaxError after patching: {error}")
    return text


def want(rel: str, snippet: str, count: int = 1) -> None:
    actual = read(rel).count(snippet)
    if actual != count:
        failures.append(f"{rel}: expected {count}x {snippet!r}, found {actual}")


def forbid(rel: str, snippet: str) -> None:
    if snippet in read(rel):
        failures.append(f"{rel}: unexpected leftover {snippet!r}")


SM120 = "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"

# --- NoPE zero-pad + sparse capacity (patch_nope_sm120.py §1) --------------
want(SM120, "    supports_dense_mha_prefill = False\n")
want(SM120, "    def do_kv_cache_update(\n")
want(SM120, "            self.rope_pad = 64\n")
want(SM120, "torch.nn.functional.pad(q, (0, self.rope_pad))")
want(SM120, "qk_rope_head_dim=self.kernel_qk_rope_head_dim")
want(SM120, "return_valid_counts=True")
want(SM120, "sparse_mla_top_k=sparse_topk_capacity")
want(SM120, "seq_lens=topk_lengths")
want(SM120, "out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)")
forbid(SM120, "attn_metadata.topk_tokens")
want(SM120, "k_pe = k_pe.new_zeros((k_pe.shape[0], 1, self.rope_pad))")

# --- candidate buffer width (§2) ------------------------------------------
for rel in ("models/glm5next/nvidia/model.py", "models/glm5next/nvidia/mtp.py"):
    want(rel, "buffer_width = topk_tokens\n")
    forbid(rel, "kpool - 1 if kpool > 1")

# --- SM120 kernel block sizes (§3) ----------------------------------------
want(
    "v1/attention/backends/mla/flashinfer_mla_sparse.py",
    "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
    "        return [64]\n",
)

# --- warmup / autotune skips (§4) -----------------------------------------
forbid("model_executor/warmup/kernel_warmup.py", "flashinfer_sparse_mla_decode_autotune_warmup(worker)")
want("model_executor/warmup/kernel_warmup.py", "Skipping FlashInfer autotune on SM121")

# --- cuda.py: upstream alignment retained, PDL gate patched (§5) ----------
want("platforms/cuda.py", "        if cls.is_device_capability_family(120):\n")
want("platforms/cuda.py", "        return index_kpool * page\n")
want("platforms/cuda.py", "        return major in (9, 10)\n")
forbid("platforms/cuda.py", "return index_kpool * min(PAGED_MQA_PAGE_SIZES)")

# --- indexer candidate expansion, kpool_ops namespace (§6) ---------------
want(
    "model_executor/layers/sparse_attn_indexer_kpool.py",
    "pool_ids[:, : select_k - 1]",
    count=2,
)

# --- exl3 quantization registration ---------------------------------------
want("model_executor/layers/quantization/__init__.py", '    "exl3",\n')
want("model_executor/layers/quantization/__init__.py", "    from .exl3 import Exl3Config\n")
if not (VLLM / "model_executor/layers/quantization/exl3.py").is_file():
    failures.append("model_executor/layers/quantization/exl3.py: missing (overlay COPY failed)")

# --- observer call sites ---------------------------------------------------
want("v1/core/kv_cache_utils.py", "# STARTUP_OBSERVER_BEGIN engine_final\n")
want("v1/core/kv_cache_utils.py", "# STARTUP_OBSERVER_BEGIN engine_projection\n")
want("v1/worker/gpu/attn_utils.py", "# STARTUP_OBSERVER_BEGIN worker_preallocation\n")
want("v1/worker/gpu/attn_utils.py", "# STARTUP_OBSERVER_BEGIN worker_postbind\n")
if not (SITE / "startup_observer.py").is_file():
    failures.append("startup_observer.py: not importable from site root")

# --- persistence: patch series applied + package importable ---------------
want(
    "distributed/kv_transfer/kv_connector/v1/offloading/common.py",
    "class DirectionalTransferStats:",
)
if not (SITE / "recipe_persistence/native.py").is_file():
    failures.append("recipe_persistence/native.py: package not installed")

if failures:
    print("OVERLAY STATIC VERIFY FAILED:", file=sys.stderr)
    for line in failures:
        print("  - " + line, file=sys.stderr)
    raise SystemExit(1)

print("glm53 overlay static verify OK (NoPE sparse-MLA + exl3 + observer + persistence)")
