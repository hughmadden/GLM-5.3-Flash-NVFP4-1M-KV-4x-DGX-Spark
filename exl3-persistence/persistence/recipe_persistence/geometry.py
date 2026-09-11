# SPDX-License-Identifier: Apache-2.0
"""CPU-only geometry from vLLM's registered canonical cache, not layer guesses."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    padded_pages: tuple[int, ...]
    group_refs: tuple[tuple[tuple[int, int], ...], ...]

    @classmethod
    def from_canonical(cls, caches):
        pages = tuple(int(t.page_size_bytes) for t in caches.tensors)
        refs = tuple(tuple((int(r.tensor_idx), int(r.page_size_bytes)) for r in g)
                     for g in caches.group_data_refs)
        if not pages or any(p <= 0 for p in pages) or not refs:
            raise ValueError("canonical cache must have positive pages and groups")
        for group in refs:
            if not group:
                raise ValueError("empty canonical cache group is unsupported")
            for index, size in group:
                if index < 0 or index >= len(pages) or not 0 < size <= pages[index]:
                    raise ValueError("invalid canonical reference")
        # The canonical API guarantees that tensors is the unique physical list.
        return cls(pages, refs)

    @property
    def group_bytes(self) -> tuple[int, ...]:
        """Actual unpadded bytes copied, including every registered reference."""
        return tuple(sum(size for _, size in refs) for refs in self.group_refs)

    @property
    def row_bytes(self) -> int:
        """Physical allocation per row, counting each canonical tensor once."""
        return sum(self.padded_pages)

    def rows_for_budget(self, budget: int, max_rows: int) -> int:
        if budget <= 0 or max_rows <= 0:
            raise ValueError("staging budget and row bound must be positive")
        rows = min(int(budget) // self.row_bytes, int(max_rows))
        if rows < 1:
            raise ValueError("staging budget cannot hold one physical canonical row")
        return rows

    def buffers(self, cpu_tensors, row: int, group: int):
        """Views only: padding never enters the persistent object."""
        return tuple(memoryview(cpu_tensors[index][row].numpy()).cast("B")[:size]
                     for index, size in self.group_refs[group])


def key_group(key: bytes, num_groups: int) -> int:
    if not isinstance(key, bytes) or len(key) <= 4:
        raise ValueError("offload key must contain a hash and four-byte group index")
    group = int.from_bytes(key[-4:], "big")
    if group >= num_groups:
        raise ValueError("offload key has an unknown cache group")
    return group
