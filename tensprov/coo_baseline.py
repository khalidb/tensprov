"""COO sparse tensor baseline. Querying scans all non-zero coordinates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Set, Tuple

RecordId = int
Coord = Tuple[RecordId, ...]


@dataclass
class COOTensor:
    dims: Tuple[str, ...]
    coords: List[Coord] = field(default_factory=list)

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def add(self, coord: Sequence[RecordId]) -> None:
        if len(coord) != self.ndim:
            raise ValueError(f"coord has length {len(coord)} but tensor has {self.ndim} dims")
        self.coords.append(tuple(int(x) for x in coord))

    def add_many(self, coords: Iterable[Sequence[RecordId]]) -> None:
        for coord in coords:
            self.add(coord)

    def project(self, source_dim: int, source_ids: Iterable[RecordId], target_dim: int) -> Set[RecordId]:
        ids = {int(x) for x in source_ids}
        return {coord[target_dim] for coord in self.coords if coord[source_dim] in ids}

    def co_contributors(self, source_dim: int, source_ids: Iterable[RecordId], other_dim: int) -> Set[RecordId]:
        return self.project(source_dim, source_ids, other_dim)

    def approx_memory_bytes(self) -> int:
        return len(self.coords) * self.ndim * 8
