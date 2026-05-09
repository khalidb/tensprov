"""In-memory graph baseline for provenance."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Iterable, Sequence, Set, Tuple

RecordNode = Tuple[int, int]
DerivationNode = Tuple[str, int]
Node = Tuple[object, int]


@dataclass
class ProvenanceGraph:
    dims: Tuple[str, ...]
    adjacency: DefaultDict[Node, Set[Node]] = field(default_factory=lambda: defaultdict(set))
    derivation_count: int = 0

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def add(self, coord: Sequence[int]) -> None:
        if len(coord) != self.ndim:
            raise ValueError(f"coord has length {len(coord)} but graph has {self.ndim} dims")
        did: DerivationNode = ("d", self.derivation_count)
        self.derivation_count += 1
        for dim, rid in enumerate(coord):
            rn: RecordNode = (dim, int(rid))
            self.adjacency[rn].add(did)
            self.adjacency[did].add(rn)

    def add_many(self, coords: Iterable[Sequence[int]]) -> None:
        for coord in coords:
            self.add(coord)

    def project(self, source_dim: int, source_ids: Iterable[int], target_dim: int) -> Set[int]:
        out: Set[int] = set()
        for sid in source_ids:
            source: RecordNode = (source_dim, int(sid))
            for deriv in self.adjacency.get(source, ()):
                for neighbor in self.adjacency.get(deriv, ()):
                    if isinstance(neighbor[0], int) and neighbor[0] == target_dim:
                        out.add(int(neighbor[1]))
        return out

    def co_contributors(self, source_dim: int, source_ids: Iterable[int], other_dim: int) -> Set[int]:
        return self.project(source_dim, source_ids, other_dim)

    def approx_memory_bytes(self) -> int:
        edge_entries = sum(len(v) for v in self.adjacency.values())
        return edge_entries * 16
