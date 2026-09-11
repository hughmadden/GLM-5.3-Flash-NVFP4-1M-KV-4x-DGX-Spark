#!/usr/bin/env python3
"""Regression tests for the K-pool tail one-block circular slot-map clamp."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_kpool_tail_slotmap.py",
        ROOT / "overlay" / "patch_kpool_tail_slotmap.py",
    )
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
from patch_kpool_tail_slotmap import (  # noqa: E402
    ANCHOR,
    MARK,
    PATCHED,
    circular_slot_ids,
    count_overruns,
    prepare,
    verified_state,
)

INSTALLED = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/block_table.py"
)

# Exact vLLM 487ecf187 / glm53-flash image kernel fragment.
# Regenerated 10 Sep 2026 AEST from pristine vLLM 83252ea89 vllm/v1/worker/block_table.py (the patch's TARGET)
# (the original fixture was Mia's day-0 tree and no longer matches the re-anchored patch).
PINNED_FIXTURE = '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n\nimport math\nfrom dataclasses import dataclass\nfrom enum import Enum\nfrom typing import Any\n\nimport numpy as np\nimport torch\n\nfrom vllm.distributed import get_dcp_group, get_pcp_group\nfrom vllm.logger import init_logger\nfrom vllm.model_executor.warmup.jit_warmup_triton_helper import (\n    LaunchSpec,\n    TritonWarmupTensor,\n    VllmTritonJitKernel,\n    kernel_launcher,\n    triton_scalar_specialization_rep,\n)\nfrom vllm.triton_utils import tl, triton\nfrom vllm.utils.math_utils import cdiv\nfrom vllm.v1.attention.backends.utils import PAD_SLOT_ID\nfrom vllm.v1.utils import CpuGpuBuffer\n\nlogger = init_logger(__name__)\n\n\ndef get_block_table_width(\n    max_num_blocks: int,\n    block_size: int,\n    kernel_block_size: int | None = None,\n    *,\n    token_alignment: int | None = 128,\n) -> int:\n    """Return the width after optional alignment and virtual block splitting."""\n    if kernel_block_size is None:\n        kernel_block_size = block_size\n    if block_size % kernel_block_size != 0:\n        raise ValueError(\n            f"kernel_block_size {kernel_block_size} must divide "\n            f"block_size {block_size} evenly"\n        )\n    if token_alignment is not None:\n        if token_alignment <= 0:\n            raise ValueError("token_alignment must be positive")\n        block_alignment = token_alignment // math.gcd(token_alignment, block_size)\n        max_num_blocks = cdiv(max_num_blocks, block_alignment) * block_alignment\n    return max_num_blocks * block_size // kernel_block_size\n\n\nclass SlotMappingMode(Enum):\n    TOKEN_TO_KV_SLOT = "token_to_kv_slot"\n    NONE = "none"\n\n\nclass BlockTable:\n    def __init__(\n        self,\n        block_size: int,\n        max_num_reqs: int,\n        max_num_blocks_per_req: int,\n        max_num_batched_tokens: int,\n        pin_memory: bool,\n        device: torch.device,\n        kernel_block_size: int,\n        cp_kv_cache_interleave_size: int,\n        slot_mapping_mode: SlotMappingMode = SlotMappingMode.TOKEN_TO_KV_SLOT,\n    ):\n        """\n        Args:\n            block_size: Block size used for KV cache memory allocation\n            max_num_reqs: Maximum number of concurrent requests supported.\n            max_num_blocks_per_req: Maximum number of blocks per request.\n            max_num_batched_tokens: Maximum number of tokens in a batch.\n            pin_memory: Whether to pin memory for faster GPU transfers.\n            device: Target device for the block table.\n            kernel_block_size: The block_size of underlying attention kernel.\n                Will be the same as `block_size` if `block_size` is supported\n                by the attention kernel.\n            slot_mapping_mode: How this cache group maps scheduled tokens to\n                cache slots. Mamba-like state caches do not use token slot\n                mappings and should use SlotMappingMode.NONE.\n        """\n        self.max_num_reqs = max_num_reqs\n        self.max_num_batched_tokens = max_num_batched_tokens\n        self.pin_memory = pin_memory\n        self.device = device\n        self.kv_cache_block_size = block_size\n\n        if kernel_block_size == block_size:\n            # Standard case: allocation and computation use same block size\n            # No block splitting needed, direct mapping\n            self.block_size = block_size\n            self.blocks_per_kv_block = 1\n            self.use_hybrid_blocks = False\n        else:\n            # Hybrid case: allocation block size differs from kernel block size\n            # Memory blocks are subdivided to match kernel requirements\n            # Example: 32-token memory blocks with 16-token kernel blocks\n            # → Each memory block corresponds to 2 kernel blocks\n            if block_size % kernel_block_size != 0:\n                raise ValueError(\n                    f"kernel_block_size {kernel_block_size} must divide "\n                    f"kv_manager_block_size size {block_size} evenly"\n                )\n\n            self.block_size = kernel_block_size\n            self.blocks_per_kv_block = block_size // kernel_block_size\n            self.use_hybrid_blocks = True\n\n        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block\n\n        self.block_table = self._make_buffer(\n            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32\n        )\n        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)\n\n        self.slot_mapping = self._make_buffer(\n            self.max_num_batched_tokens, dtype=torch.int64\n        )\n\n        if self.use_hybrid_blocks:\n            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(\n                1, -1\n            )\n        else:\n            self._kernel_block_arange = None\n\n        try:\n            self.pcp_world_size = get_pcp_group().world_size\n            self.pcp_rank = get_pcp_group().rank_in_group\n        except AssertionError:\n            # PCP might not be initialized in testing\n            self.pcp_world_size = 1\n            self.pcp_rank = 0\n        try:\n            self.dcp_world_size = get_dcp_group().world_size\n            self.dcp_rank = get_dcp_group().rank_in_group\n        except AssertionError:\n            # DCP might not be initialized in testing\n            self.dcp_world_size = 1\n            self.dcp_rank = 0\n        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size\n        self.slot_mapping_mode = slot_mapping_mode\n        if self.slot_mapping_mode == SlotMappingMode.TOKEN_TO_KV_SLOT:\n            _COMPUTE_SLOT_MAPPING_KERNEL.register_warmup(\n                kv_cache_block_size=self.kv_cache_block_size,\n                blocks_per_kv_block=self.blocks_per_kv_block,\n                total_cp_world_size=self.dcp_world_size,\n                total_cp_rank=self.dcp_rank,\n                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,\n                block_table_stride=self.block_table.gpu.stride(0),\n                block_size=self.block_size,\n            )\n\n    def append_row(\n        self,\n        block_ids: list[int],\n        row_idx: int,\n    ) -> None:\n        if not block_ids:\n            return\n\n        if self.use_hybrid_blocks:\n            block_ids = self.map_to_kernel_blocks(\n                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange\n            )\n\n        num_blocks = len(block_ids)\n        start = self.num_blocks_per_row[row_idx]\n        self.num_blocks_per_row[row_idx] += num_blocks\n        self.block_table.np[row_idx, start : start + num_blocks] = block_ids\n\n    def add_row(self, block_ids: list[int], row_idx: int) -> None:\n        self.num_blocks_per_row[row_idx] = 0\n        self.append_row(block_ids, row_idx)\n\n    def clear_row(self, row_idx: int) -> None:\n        num_blocks = self.num_blocks_per_row[row_idx]\n        if num_blocks > 0:\n            self.block_table.np[row_idx, :num_blocks] = 0\n        self.num_blocks_per_row[row_idx] = 0\n\n    def move_row(self, src: int, tgt: int) -> None:\n        num_blocks = self.num_blocks_per_row[src]\n        block_table_np = self.block_table.np\n        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]\n        self.num_blocks_per_row[tgt] = num_blocks\n        # Clear the vacated source row: dummy-run batches dereference stale\n        # rows as mamba state slots and write state in place there, possibly\n        # after the blocks have been freed and reallocated.\n        block_table_np[src, :num_blocks] = 0\n        self.num_blocks_per_row[src] = 0\n\n    def swap_row(self, src: int, tgt: int) -> None:\n        src_tgt, tgt_src = [src, tgt], [tgt, src]\n        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]\n        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]\n\n    def compute_slot_mapping(\n        self,\n        num_reqs: int,\n        query_start_loc: torch.Tensor,\n        positions: torch.Tensor,\n    ) -> None:\n        num_tokens = positions.shape[0]\n        if self.slot_mapping_mode == SlotMappingMode.NONE:\n            # Mamba/GDN groups consume the block table as recurrent state\n            # indices and do not use per-token slot mappings.\n            return\n        assert self.slot_mapping_mode == SlotMappingMode.TOKEN_TO_KV_SLOT\n\n        _COMPUTE_SLOT_MAPPING_KERNEL(\n            num_reqs,\n            num_tokens,\n            self.max_num_batched_tokens,\n            query_start_loc,\n            positions,\n            self.block_table.gpu,\n            self.block_table.gpu.stride(0),\n            self.block_size,\n            self.slot_mapping.gpu,\n            self.kv_cache_block_size,\n            self.blocks_per_kv_block,\n            self.dcp_world_size,\n            self.dcp_rank,\n            self.cp_kv_cache_interleave_size,\n        )\n\n    def commit_block_table(self, num_reqs: int) -> None:\n        self.block_table.copy_to_gpu(num_reqs)\n\n    def clear(self) -> None:\n        self.block_table.gpu.fill_(0)\n        self.block_table.cpu.fill_(0)\n\n    @staticmethod\n    def map_to_kernel_blocks(\n        kv_manager_block_ids: np.ndarray,\n        blocks_per_kv_block: int,\n        kernel_block_arange: np.ndarray,\n    ) -> np.ndarray:\n        """Convert kv_manager_block_id IDs to kernel block IDs.\n\n        Example:\n            # kv_manager_block_ids: 32 tokens,\n            # Kernel block size: 16 tokens\n            # blocks_per_kv_block = 2\n            >>> kv_manager_block_ids = np.array([0, 1, 2])\n            >>> Result: [0, 1, 2, 3, 4, 5]\n\n            # Each kv_manager_block_id maps to 2 kernel block id:\n            # kv_manager_block_id 0 → kernel block id [0, 1]\n            # kv_manager_block_id 1 → kernel block id [2, 3]\n            # kv_manager_block_id 2 → kernel block id [4, 5]\n        """\n        if blocks_per_kv_block == 1:\n            return kv_manager_block_ids\n\n        kernel_block_ids = (\n            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block\n            + kernel_block_arange\n        )\n\n        return kernel_block_ids.reshape(-1)\n\n    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:\n        """Returns the device tensor of the block table."""\n        return self.block_table.gpu[:num_reqs]\n\n    def get_cpu_tensor(self) -> torch.Tensor:\n        """Returns the CPU tensor of the block table."""\n        return self.block_table.cpu\n\n    def get_numpy_array(self) -> np.ndarray:\n        """Returns the numpy array of the block table."""\n        return self.block_table.np\n\n    def _make_buffer(\n        self, *size: int | torch.SymInt, dtype: torch.dtype\n    ) -> CpuGpuBuffer:\n        return CpuGpuBuffer(\n            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory\n        )\n\n\nclass MultiGroupBlockTable:\n    """The BlockTables for each KV cache group."""\n\n    def __init__(\n        self,\n        max_num_reqs: int,\n        max_num_batched_tokens: int,\n        pin_memory: bool,\n        device: torch.device,\n        block_sizes: list[int],\n        kernel_block_sizes: list[int],\n        max_num_blocks: list[int],\n        cp_kv_cache_interleave_size: int = 1,\n        slot_mapping_modes: list[SlotMappingMode] | None = None,\n    ) -> None:\n        if len(kernel_block_sizes) != len(block_sizes):\n            raise ValueError(\n                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "\n                f"must match block_sizes length ({len(block_sizes)})"\n            )\n        if slot_mapping_modes is None:\n            slot_mapping_modes = [SlotMappingMode.TOKEN_TO_KV_SLOT] * len(block_sizes)\n        if len(slot_mapping_modes) != len(block_sizes):\n            raise ValueError(\n                f"slot_mapping_modes length ({len(slot_mapping_modes)}) "\n                f"must match block_sizes length ({len(block_sizes)})"\n            )\n\n        if len(max_num_blocks) != len(block_sizes):\n            raise ValueError(\n                f"max_num_blocks length ({len(max_num_blocks)}) "\n                f"must match block_sizes length ({len(block_sizes)})"\n            )\n\n        max_num_blocks = [\n            (\n                get_block_table_width(n, block_size, token_alignment=None)\n                if slot_mapping_mode == SlotMappingMode.NONE\n                else get_block_table_width(n, block_size)\n            )\n            for n, block_size, slot_mapping_mode in zip(\n                max_num_blocks, block_sizes, slot_mapping_modes\n            )\n        ]\n\n        self.block_tables = [\n            BlockTable(\n                block_size,\n                max_num_reqs,\n                max_num_blocks_per_req,\n                max_num_batched_tokens,\n                pin_memory,\n                device,\n                kernel_block_size,\n                cp_kv_cache_interleave_size,\n                slot_mapping_mode=slot_mapping_mode,\n            )\n            for (\n                block_size,\n                kernel_block_size,\n                max_num_blocks_per_req,\n                slot_mapping_mode,\n            ) in zip(\n                block_sizes, kernel_block_sizes, max_num_blocks, slot_mapping_modes\n            )\n        ]\n\n    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:\n        for i, block_table in enumerate(self.block_tables):\n            block_table.append_row(block_ids[i], row_idx)\n\n    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:\n        for i, block_table in enumerate(self.block_tables):\n            block_table.add_row(block_ids[i], row_idx)\n\n    def clear_row(self, row_idx: int) -> None:\n        for block_table in self.block_tables:\n            block_table.clear_row(row_idx)\n\n    def move_row(self, src: int, tgt: int) -> None:\n        for block_table in self.block_tables:\n            block_table.move_row(src, tgt)\n\n    def swap_row(self, src: int, tgt: int) -> None:\n        for block_table in self.block_tables:\n            block_table.swap_row(src, tgt)\n\n    def compute_slot_mapping(\n        self,\n        num_reqs: int,\n        query_start_loc: torch.Tensor,\n        positions: torch.Tensor,\n    ) -> None:\n        for block_table in self.block_tables:\n            block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)\n\n    def commit_block_table(self, num_reqs: int) -> None:\n        for block_table in self.block_tables:\n            block_table.commit_block_table(num_reqs)\n\n    def clear(self) -> None:\n        for block_table in self.block_tables:\n            block_table.clear()\n\n    def __getitem__(self, idx: int) -> "BlockTable":\n        """Returns the BlockTable for the i-th KV cache group."""\n        return self.block_tables[idx]\n\n\nclass ComputeSlotMappingKernel(\n    VllmTritonJitKernel["ComputeSlotMappingKernel.CompileKey"]\n):\n    triton_block_size = 1024\n\n    @dataclass(frozen=True)\n    class CompileKey:\n        kv_cache_block_size: int\n        blocks_per_kv_block: int\n        total_cp_world_size: int\n        total_cp_rank: int\n        cp_kv_cache_interleave_size: int\n        block_table_stride: int\n        block_size: int\n\n    @staticmethod\n    @triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])\n    def kernel(\n        num_tokens,\n        max_num_tokens,\n        query_start_loc_ptr,  # [num_reqs + 1], int32\n        positions_ptr,  # [num_tokens], int64\n        block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)\n        block_table_stride,  # max_num_blocks_per_req\n        block_size,\n        slot_mapping_ptr,  # [max_num_tokens], int64\n        KV_CACHE_BLOCK_SIZE: tl.constexpr,\n        BLOCKS_PER_KV_BLOCK: tl.constexpr,\n        TOTAL_CP_WORLD_SIZE: tl.constexpr,\n        TOTAL_CP_RANK: tl.constexpr,\n        CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,\n        PAD_ID: tl.constexpr,\n        BLOCK_SIZE: tl.constexpr,\n    ):\n        req_idx = tl.program_id(0)\n\n        if req_idx == tl.num_programs(0) - 1:\n            # Pad remaining slots for CUDA graph compatibility.\n            for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):\n                offsets = i + tl.arange(0, BLOCK_SIZE)\n                tl.store(\n                    slot_mapping_ptr + offsets,\n                    PAD_ID,\n                    mask=offsets < max_num_tokens,\n                )\n            return\n\n        start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)\n        end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)\n\n        virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE\n        row_offset = req_idx * block_table_stride\n        for i in range(start_idx, end_idx, BLOCK_SIZE):\n            offsets = i + tl.arange(0, BLOCK_SIZE)\n            mask = offsets < end_idx\n            pos = tl.load(positions_ptr + offsets, mask=mask, other=0)\n            virtual_block_indices = pos // virtual_block_size\n            virtual_block_offsets = pos - virtual_block_indices * virtual_block_size\n            is_local = (\n                virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE\n            ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK\n            local_block_offsets = (\n                virtual_block_offsets\n                // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)\n            ) * CP_KV_CACHE_INTERLEAVE_SIZE + (\n                virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE\n            )\n\n            block_indices = (\n                virtual_block_indices * BLOCKS_PER_KV_BLOCK\n                + local_block_offsets // block_size\n            )\n            block_numbers = tl.load(\n                block_table_ptr + row_offset + block_indices,\n                mask=mask & is_local,\n                other=0,\n            ).to(tl.int64)\n            slot_offsets = local_block_offsets % block_size\n            slot_ids = block_numbers * block_size + slot_offsets\n            slot_ids = tl.where(is_local, slot_ids, PAD_ID)\n            tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)\n\n    def dispatch(  # type: ignore[override]\n        self,\n        *,\n        block_table_stride: int,\n        block_size: int,\n        **compile_key_fields: int,\n    ) -> CompileKey:\n        return self.CompileKey(\n            **compile_key_fields,\n            block_table_stride=triton_scalar_specialization_rep(block_table_stride),\n            block_size=triton_scalar_specialization_rep(block_size),\n        )\n\n    def get_warmup_keys(self, **dispatch_kwargs: int) -> list[CompileKey]:\n        return self._trace_dispatch(self.dispatch)(**dispatch_kwargs)\n\n    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:\n        int32_ptr = TritonWarmupTensor(torch.int32)\n        int64_ptr = TritonWarmupTensor(torch.int64)\n        return dict(\n            num_reqs=1,\n            num_tokens=2,  # arbitrary, in do_not_specialize\n            max_num_tokens=2,  # arbitrary, in do_not_specialize\n            query_start_loc=int32_ptr,\n            positions=int64_ptr,\n            block_table=TritonWarmupTensor(\n                torch.int32,\n                shape=(1, compile_key.block_table_stride),\n            ),\n            block_table_stride=compile_key.block_table_stride,\n            block_size=compile_key.block_size,\n            slot_mapping=int64_ptr,\n            kv_cache_block_size=compile_key.kv_cache_block_size,\n            blocks_per_kv_block=compile_key.blocks_per_kv_block,\n            total_cp_world_size=compile_key.total_cp_world_size,\n            total_cp_rank=compile_key.total_cp_rank,\n            cp_kv_cache_interleave_size=compile_key.cp_kv_cache_interleave_size,\n        )\n\n    @kernel_launcher\n    def __call__(\n        self,\n        num_reqs: int,\n        num_tokens: int,\n        max_num_tokens: int,\n        query_start_loc: torch.Tensor,\n        positions: torch.Tensor,\n        block_table: torch.Tensor,\n        block_table_stride: int,\n        block_size: int,\n        slot_mapping: torch.Tensor,\n        kv_cache_block_size: int,\n        blocks_per_kv_block: int,\n        total_cp_world_size: int,\n        total_cp_rank: int,\n        cp_kv_cache_interleave_size: int,\n    ) -> LaunchSpec:\n        return (num_reqs + 1,), dict(\n            KV_CACHE_BLOCK_SIZE=kv_cache_block_size,\n            BLOCKS_PER_KV_BLOCK=blocks_per_kv_block,\n            TOTAL_CP_WORLD_SIZE=total_cp_world_size,\n            TOTAL_CP_RANK=total_cp_rank,\n            CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,\n            PAD_ID=PAD_SLOT_ID,\n            BLOCK_SIZE=self.triton_block_size,\n        )\n\n\n_COMPUTE_SLOT_MAPPING_KERNEL = ComputeSlotMappingKernel()\n'  # pristine file already contains ANCHOR


def _run_patch(target: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_BLOCK_TABLE_PY"] = str(target)
    return subprocess.run(
        [sys.executable, str(PATCH)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_circular_math() -> None:
    # Tail group: one entry, block_size == index_kpool == 4.
    row = [17]
    positions = list(range(0, 64))
    assert count_overruns(positions, block_size=4, stride=1) == 60
    patched = circular_slot_ids(positions, row, 4, clamp=True)
    assert patched[:4] == [68, 69, 70, 71]  # 17*4 + 0..3
    # Circular: pos 4, 8, 12 map back onto the same four slots.
    assert patched[4:8] == patched[:4]
    assert patched[60:64] == patched[:4]
    unpatched = circular_slot_ids(positions[:4], row, 4, clamp=False)
    assert unpatched == patched[:4]
    try:
        circular_slot_ids([4], row, 4, clamp=False)
    except IndexError:
        pass
    else:
        raise AssertionError("unpatched mapping must IndexError at pos >= block_size")

    # Full-attention group: clamp is identity inside the row.
    wide = list(range(10))
    pos = [0, 63, 64, 639]  # block_size 64, last index 9
    assert count_overruns(pos, block_size=64, stride=10) == 0
    assert circular_slot_ids(pos, wide, 64, clamp=True) == circular_slot_ids(
        pos, wide, 64, clamp=False
    )


def test_fixture() -> None:
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "block_table.py"
        target.write_text(PINNED_FIXTURE)
        first = _run_patch(target)
        assert first.returncode == 0, first.stderr
        text = target.read_text()
        assert verified_state(text)
        assert MARK in text
        assert "tl.minimum(block_indices, block_table_stride - 1)" in text
        assert "already present" not in first.stdout
        second = _run_patch(target)
        assert second.returncode == 0, second.stderr
        assert "already present" in second.stdout
        assert second.stdout.count("already present") == 1
        again, action = prepare(text)
        assert action == "already present"
        assert again == text


def test_fail_closed() -> None:
    drifted = PINNED_FIXTURE.replace(
        "local_block_offsets // block_size",
        "local_block_offsets // kernel_block_size",
        1,
    )
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "block_table.py"
        target.write_text(drifted)
        result = _run_patch(target)
        assert result.returncode != 0
        assert "preflight failed" in result.stderr
        assert target.read_text() == drifted

    partial = PINNED_FIXTURE.replace(ANCHOR, MARK + ANCHOR, 1)
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "block_table.py"
        target.write_text(partial)
        result = _run_patch(target)
        assert result.returncode != 0
        assert "partial/inconsistent" in result.stderr


def test_installed_copy_if_present() -> None:
    src = Path(os.environ.get("GLM53_BLOCK_TABLE_PY_SRC", INSTALLED))
    if not src.is_file():
        return
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "block_table.py"
        target.write_text(src.read_text())
        result = _run_patch(target)
        assert result.returncode == 0, result.stderr
        assert verified_state(target.read_text())


def test_recipe_wiring_if_present() -> None:
    start = ROOT / "start.sh"
    dockerfile = ROOT / "Dockerfile"
    if not start.is_file() or not dockerfile.is_file():
        return
    launcher = start.read_text()
    image = dockerfile.read_text()
    assert 'KPOOL_TAIL_PATCH_HOST="${KPOOL_TAIL_PATCH_HOST:-' in launcher
    assert launcher.count("python3 /opt/glm53/patch_kpool_tail_slotmap.py") == 2
    assert (
        "-v '/tmp/patch_kpool_tail_slotmap.py:"
        "/opt/glm53/patch_kpool_tail_slotmap.py:ro'" in launcher
    )
    assert (
        '-v "$KPOOL_TAIL_PATCH_HOST:'
        '/opt/glm53/patch_kpool_tail_slotmap.py:ro"' in launcher
    )
    assert 'scp -q -o BatchMode=yes "$KPOOL_TAIL_PATCH_HOST"' in launcher
    assert "COPY overlay/patch_kpool_tail_slotmap.py" in image
    assert "RUN python3 /opt/glm53/patch_kpool_tail_slotmap.py" in image
    assert "python3 /opt/glm53/test_kpool_tail_slotmap.py" in image


def main() -> int:
    test_circular_math()
    test_fixture()
    test_fail_closed()
    test_installed_copy_if_present()
    test_recipe_wiring_if_present()
    print("kpool tail slot-map patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
