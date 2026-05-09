from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple
import numpy as np


RecordId = int
DerivationId = int


class NumpyTensorIndex:
    """
    NumPy-backed provenance index.

    Drop-in replacement for IndexedTensorArray.

    Stores coordinates as one NumPy int64 array per dimension:
      filter/dropna: out, in
      join: out, left, right

    Identity operations are handled by IdentityOperation and therefore
    do not instantiate this index.
    """

    def __init__(self, dims: Tuple[str, ...]):
        if not dims:
            raise ValueError("NumpyTensorIndex requires at least one dimension")

        self.dims = tuple(dims)
        self.columns: List[np.ndarray] = [
            np.empty(0, dtype=np.int64) for _ in self.dims
        ]
        self._ncoords = 0

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def ncoords(self) -> int:
        return self._ncoords

    def add(self, coord) -> int:
        """
        Compatibility method. Prefer add_many().
        """
        arr = np.asarray(coord, dtype=np.int64).reshape(1, self.ndim)
        start = self._ncoords
        self.add_many(arr)
        return start

    def add_many(self, coords) -> None:
        """
        Bulk construction from NumPy array or iterable.

        Expected shape: (ncoords, ndim).
        """
        arr = np.asarray(coords, dtype=np.int64)

        if arr.size == 0:
            return

        if arr.ndim != 2 or arr.shape[1] != self.ndim:
            raise ValueError(
                f"coords must have shape (n, {self.ndim}), got {arr.shape}"
            )

        # Store parallel arrays. Copy to detach from caller memory.
        self.columns = [
            np.ascontiguousarray(arr[:, dim], dtype=np.int64)
            for dim in range(self.ndim)
        ]
        self._ncoords = arr.shape[0]

    def derivations_for(self, dim: int, ids: Iterable[RecordId]) -> Set[DerivationId]:
        """
        Return derivation ids where columns[dim] is in ids.
        """
        ids_arr = np.fromiter((int(x) for x in ids), dtype=np.int64)

        if ids_arr.size == 0 or self._ncoords == 0:
            return set()

        mask = np.isin(self.columns[dim], ids_arr)
        return set(np.nonzero(mask)[0].astype(int).tolist())

    def project(
        self,
        source_dim: int,
        source_ids: Iterable[RecordId],
        target_dim: int,
    ) -> Set[RecordId]:
        """
        Vectorized projection:
          source ids on one dimension -> target ids on another dimension.
        """
        ids_arr = np.fromiter((int(x) for x in source_ids), dtype=np.int64)

        if ids_arr.size == 0 or self._ncoords == 0:
            return set()

        mask = np.isin(self.columns[source_dim], ids_arr)
        return set(self.columns[target_dim][mask].astype(int).tolist())

    def project_many(
        self,
        source_dim: int,
        source_ids: Iterable[RecordId],
        target_dims: Iterable[int],
    ) -> Dict[int, Set[RecordId]]:
        """
        Vectorized projection to multiple target dimensions using one mask.
        """
        target_dims = list(target_dims)
        ids_arr = np.fromiter((int(x) for x in source_ids), dtype=np.int64)

        if ids_arr.size == 0 or self._ncoords == 0:
            return {td: set() for td in target_dims}

        mask = np.isin(self.columns[source_dim], ids_arr)

        return {
            td: set(self.columns[td][mask].astype(int).tolist())
            for td in target_dims
        }

    def co_contributors(
        self,
        source_dim: int,
        source_ids: Iterable[RecordId],
        other_dim: int,
    ) -> Set[RecordId]:
        return self.project(source_dim, source_ids, other_dim)

    def approx_memory_bytes(self) -> int:
        return sum(col.nbytes for col in self.columns)
