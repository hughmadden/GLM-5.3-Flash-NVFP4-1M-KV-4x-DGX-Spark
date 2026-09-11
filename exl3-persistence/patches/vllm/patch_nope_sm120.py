#!/usr/bin/env python3
"""GLM-5.3-Flash NoPE sparse-MLA overlay for SM121 (GB10), ported to vLLM 83252ea89.

Source of record: Mia's ``Dockerfile`` inline ``RUN python3 - <<'PY'`` heredoc
(``mia-upstream/Dockerfile`` lines 91-318). That heredoc is reproduced here as a
COPY-ed script so it can be ``py_compile``-checked on the authoring host and so
its anchors are diffable; the edit semantics are unchanged.

Port deltas versus Mia's heredoc, per
``notes/OVERLAY-PORT-NOTES.md`` §5 (anchor survival at the pin: 14/17 intact):

  DROPPED  ``platforms/cuda.py``  ``return index_kpool * min(PAGED_MQA_PAGE_SIZES)``
           -> upstream now implements the same sm120 alignment natively
           (``cuda.py:431-437``: ``if cls.is_device_capability_family(120): page =
           max(PAGED_MQA_PAGE_SIZES)``). Re-applying Mia's edit would double-apply.
           This script asserts the upstream form is present instead (fail-closed),
           so a future upstream revert breaks the build loudly rather than
           silently serving a mis-aligned indexer block.

  RE-ANCHORED  both ``expand_pools_and_append_tail`` call sites in
           ``model_executor/layers/sparse_attn_indexer_kpool.py`` -- upstream moved
           them into the ``kpool_ops.`` namespace (:580 prefill, :865 decode).

Every other anchor is byte-identical to Mia's and was re-confirmed
``count() == 1`` against the pinned tree before this script was written.

Fail-closed by construction: ``replace_once``/``require_once`` raise on a missing
or non-unique anchor, and the Dockerfile runs this under ``set -e``.
"""

from __future__ import annotations

import os
from pathlib import Path

SITE = Path(os.environ.get("GLM53_SITE", "/usr/local/lib/python3.12/dist-packages"))
VLLM = SITE / "vllm"


def _edit(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in pairs:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(
                f"{path}: expected exactly one patch target, found {count}: {old!r}"
            )
        text = text.replace(old, new)
    path.write_text(text)


def _require(path: Path, snippet: str, why: str) -> None:
    count = path.read_text().count(snippet)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected exactly one occurrence of an already-upstream "
            f"construct ({why}), found {count}: {snippet!r}"
        )


# --------------------------------------------------------------------------
# 1. flashinfer_mla_sparse_sm120.py -- the NoPE 512 -> 576 zero-pad, the
#    sparse-capacity/valid-count fix, and the do_kv_cache_update override that
#    feeds concat_and_cache_mla a 64-wide zero k_pe (satisfies the compiled
#    STD_TORCH_CHECK(pe_dim == 64) with no csrc rebuild).
# --------------------------------------------------------------------------
sm120 = VLLM / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"

# SECOND DROPPED EDIT, found while validating this port on the pinned tree
# (2026-09-10) and NOT recorded in OVERLAY-PORT-NOTES.md §5: upstream now
# declares ``supports_dense_mha_prefill = False`` on
# ``FlashInferMLASparseSM120Impl`` itself (:36, immediately after
# ``is_sparse = True``). Mia's anchor ends at ``is_sparse = True\n`` so it still
# matches exactly once, and re-applying her edit inserts a DUPLICATE class
# attribute. Harmless to Python, but it means the overlay is silently no longer
# the thing that causes the behaviour. Downgraded to the same fail-closed
# presence assert the port uses for the other upstreamed edits, so an upstream
# revert breaks the build loudly instead of falling back to the missing
# forward_mha path.
_require(
    sm120,
    "    is_sparse = True\n    supports_dense_mha_prefill = False\n",
    "upstream now declares supports_dense_mha_prefill=False (replaces Mia's edit)",
)

_edit(
    sm120,
    [
        (
            '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
            "        from vllm.config import get_current_vllm_config\n",
            '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
            "        self.rope_pad = 0\n"
            "        if self.qk_rope_head_dim == 0:\n"
            "            if self.kv_lora_rank != 512:\n"
            "                raise NotImplementedError(\n"
            '                    "FLASHINFER_MLA_SPARSE_SM120 pads NoPE MLA into the "\n'
            '                    "576-wide GLM_NSA geometry, which requires "\n'
            '                    f"kv_lora_rank=512; got {self.kv_lora_rank}."\n'
            "                )\n"
            "            self.rope_pad = 64\n"
            "        self.kernel_qk_rope_head_dim = self.qk_rope_head_dim + self.rope_pad\n"
            "        from vllm.config import get_current_vllm_config\n",
        ),
        (
            "        if isinstance(q, tuple):\n"
            "            q = torch.cat(q, dim=-1)\n"
            "\n"
            "        num_actual_toks = q.shape[0]\n",
            "        if isinstance(q, tuple):\n"
            "            q = torch.cat(q, dim=-1)\n"
            "        if self.rope_pad:\n"
            "            q = torch.nn.functional.pad(q, (0, self.rope_pad))\n"
            "\n"
            "        num_actual_toks = q.shape[0]\n",
        ),
        (
            "            qk_rope_head_dim=self.qk_rope_head_dim,\n",
            "            qk_rope_head_dim=self.kernel_qk_rope_head_dim,\n",
        ),
        (
            "        topk_indices_physical = cast(\n"
            "            torch.Tensor,\n"
            "            triton_convert_req_index_to_global_index(\n"
            "                attn_metadata.req_id_per_token[:num_actual_toks],\n"
            "                attn_metadata.block_table,\n"
            "                topk_indices,\n"
            "                BLOCK_SIZE=attn_metadata.block_size,\n"
            "                NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
            "            ),\n"
            "        )\n",
            "        topk_indices_physical, topk_lengths = cast(\n"
            "            tuple[torch.Tensor, torch.Tensor],\n"
            "            triton_convert_req_index_to_global_index(\n"
            "                attn_metadata.req_id_per_token[:num_actual_toks],\n"
            "                attn_metadata.block_table,\n"
            "                topk_indices,\n"
            "                BLOCK_SIZE=attn_metadata.block_size,\n"
            "                NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
            "                return_valid_counts=True,\n"
            "            ),\n"
            "        )\n"
            "        sparse_topk_capacity = topk_indices_physical.shape[1]\n"
            "        empty_rows = topk_lengths == 0\n"
            "        topk_indices_physical[:, 0] = topk_indices_physical[:, 0].masked_fill(\n"
            "            empty_rows, 0\n"
            "        )\n"
            "        topk_lengths = topk_lengths.clamp(min=1)\n",
        ),
        (
            "            seq_lens=None,\n            max_seq_len=attn_metadata.topk_tokens,\n",
            "            seq_lens=topk_lengths,\n            max_seq_len=sparse_topk_capacity,\n",
        ),
        (
            "            sparse_mla_top_k=attn_metadata.topk_tokens,\n",
            "            sparse_mla_top_k=sparse_topk_capacity,\n",
        ),
        (
            "        return out.squeeze(1), None\n",
            "        out = out.squeeze(1)\n"
            "        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)\n"
            "        return out, None\n"
            "\n"
            "    def do_kv_cache_update(\n"
            "        self,\n"
            "        kv_c_normed: torch.Tensor,\n"
            "        k_pe: torch.Tensor,\n"
            "        kv_cache: torch.Tensor,\n"
            "        slot_mapping: torch.Tensor,\n"
            "        kv_cache_dtype: str,\n"
            "        k_scale: torch.Tensor,\n"
            "    ) -> None:\n"
            "        if self.rope_pad:\n"
            "            k_pe = k_pe.new_zeros((k_pe.shape[0], 1, self.rope_pad))\n"
            "        super().do_kv_cache_update(\n"
            "            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale\n"
            "        )\n",
        ),
    ],
)

# --------------------------------------------------------------------------
# 2. Candidate buffer width: the indexer must emit exactly 2048 candidates,
#    not topk + (kpool - 1). Drops the 4 lowest-ranked pool tokens, keeps the
#    recent tail.
# --------------------------------------------------------------------------
for rel in ("models/glm5next/nvidia/model.py", "models/glm5next/nvidia/mtp.py"):
    _edit(
        VLLM / rel,
        [
            (
                "buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)",
                "buffer_width = topk_tokens",
            )
        ],
    )

# --------------------------------------------------------------------------
# 3. Pin the SM120 sparse backend's kernel block sizes to [64]: every
#    GLM_NSA/DSV3_2 kernel is instantiated at PAGE_BLOCK_SIZE=64 only, and the
#    upstream indexer alignment (see §5 below) now lands block_size on 1792,
#    which 256 *does* divide -- so [64, 256] would mis-select 256.
# --------------------------------------------------------------------------
_edit(
    VLLM / "v1/attention/backends/mla/flashinfer_mla_sparse.py",
    [
        (
            "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
            "        return [64, 256]\n",
            "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
            "        return [64]\n",
        )
    ],
)

# --------------------------------------------------------------------------
# 4. FlashInfer sparse-MLA autotune and the fused_moe gemm autotune both wedge
#    rank 0 on SM121. Skip the sparse warmup and return before the autotuner.
# --------------------------------------------------------------------------
_edit(
    VLLM / "model_executor/warmup/kernel_warmup.py",
    [
        (
            "    flashinfer_sparse_mla_decode_autotune_warmup(worker)\n"
            "    deepseek_v4_sparse_mla_attention_warmup(worker)\n",
            "    # GLM53_SKIP_FI_SPARSE_WARMUP: SM120 autotune wedges rank 0 on GB10.\n"
            "    deepseek_v4_sparse_mla_attention_warmup(worker)\n",
        ),
        (
            "    from flashinfer.autotuner import AutoTuner, set_autotune_process_group\n",
            '    logger.info_once("Skipping FlashInfer autotune on SM121")\n'
            "    return\n"
            "    from flashinfer.autotuner import AutoTuner, set_autotune_process_group\n",
        ),
    ],
)

# --------------------------------------------------------------------------
# 5. platforms/cuda.py
#    (a) DROPPED edit -- the indexer block alignment is upstream now. Assert it.
#    (b) PDL lowering races the KDA state kernels on SM12x; gate to 9/10.
# --------------------------------------------------------------------------
cuda = VLLM / "platforms/cuda.py"
_require(
    cuda,
    "        if cls.is_device_capability_family(120):",
    "upstream sm120 paged-MQA indexer alignment (replaces Mia's edit)",
)
_require(
    cuda,
    "        return index_kpool * page\n",
    "upstream sm120 paged-MQA indexer alignment return",
)
_edit(
    cuda,
    [
        (
            "            return False\n        return major >= 9\n",
            "            return False\n"
            "        # PDL lowering races KDA state kernels on SM12x (GB10).\n"
            "        return major in (9, 10)\n",
        )
    ],
)

# --------------------------------------------------------------------------
# 6. Indexer candidate expansion: drop the lowest-ranked selected pool rather
#    than the recent tail. RE-ANCHORED onto the upstream ``kpool_ops.`` namespace.
# --------------------------------------------------------------------------
_edit(
    VLLM / "model_executor/layers/sparse_attn_indexer_kpool.py",
    [
        (
            "                    expanded = kpool_ops.expand_pools_and_append_tail(\n"
            "                        pool_ids, q_seq, index_kpool\n"
            "                    )\n",
            "                    expanded = kpool_ops.expand_pools_and_append_tail(\n"
            "                        pool_ids[:, : select_k - 1], q_seq, index_kpool\n"
            "                    )\n",
        ),
        (
            "            out = kpool_ops.expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)\n",
            "            out = kpool_ops.expand_pools_and_append_tail(\n"
            "                pool_ids[:, : select_k - 1], dec_seq, index_kpool\n"
            "            )\n",
        ),
    ],
)

print("glm53 NoPE sparse-MLA overlay applied (83252ea89 port; cuda.py align edit DROPPED)")
