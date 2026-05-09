"""Indexed tensor provenance structure.

Lean implementation of the paper's tree-like sparse tensor idea: non-zero tensor
coordinates are stored once, and each tensor dimension has an inverted index from
record id -> derivation ids. Set-oriented slicing/projecting is therefore
proportional to the number of relevant derivations, not to all tensor entries.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

RecordId = int
DerivationId = int
Coord = Tuple[RecordId, ...]


@dataclass
class IndexedTensor:
    """A sparse binary tensor optimized for provenance queries."""

    dims: Tuple[str, ...]
    coords: List[Coord] = field(default_factory=list)
    by_dim: List[Dict[RecordId, Set[DerivationId]]] = field(init=False)

    def __post_init__(self) -> None:
        if not self.dims:
            raise ValueError("IndexedTensor requires at least one dimension")
        self.by_dim = [defaultdict(set) for _ in self.dims]

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def add(self, coord: Sequence[RecordId]) -> DerivationId:
        if len(coord) != self.ndim:
            raise ValueError(f"coord has length {len(coord)} but tensor has {self.ndim} dims")
        deriv_id = len(self.coords)
        c = tuple(int(x) for x in coord)
        self.coords.append(c)
        for dim, value in enumerate(c):
            self.by_dim[dim][value].add(deriv_id)
        return deriv_id

    def add_many(self, coords: Iterable[Sequence[RecordId]]) -> None:
        for coord in coords:
            self.add(coord)

    def derivations_for(self, dim: int, ids: Iterable[RecordId]) -> Set[DerivationId]:
        result: Set[DerivationId] = set()
        idx = self.by_dim[dim]
        for rid in ids:
            result.update(idx.get(int(rid), ()))
        return result

    def project(self, source_dim: int, source_ids: Iterable[RecordId], target_dim: int) -> Set[RecordId]:
        derivs = self.derivations_for(source_dim, source_ids)
        return {self.coords[d][target_dim] for d in derivs}

    def project_many(self, source_dim: int, source_ids: Iterable[RecordId], target_dims: Iterable[int]) -> Dict[int, Set[RecordId]]:
        derivs = self.derivations_for(source_dim, source_ids)
        return {td: {self.coords[d][td] for d in derivs} for td in target_dims}

    def co_contributors(self, source_dim: int, source_ids: Iterable[RecordId], other_dim: int) -> Set[RecordId]:
        return self.project(source_dim, source_ids, other_dim)

    def approx_memory_bytes(self) -> int:
        ints = len(self.coords) * self.ndim
        ints += sum(len(derivs) for dim_index in self.by_dim for derivs in dim_index.values())
        return ints * 8
