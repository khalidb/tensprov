"""Memory-optimized indexed tensor provenance structure.

This variant keeps the same logical model as IndexedTensor:
    coordinates = exact non-zero tensor cells / derivation facts
    indexes     = fast access paths from each dimension value to derivations

The difference is implementation-oriented: coordinates are stored column-wise in
compact integer arrays, and each per-dimension posting list is an integer array
rather than a Python set. This reduces object overhead while preserving exact
join tuples and set-oriented query semantics.
"""
from __future__ import annotations

from array import array
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple

RecordId = int
DerivationId = int


def _new_uint_array() -> array:
    # Unsigned 32-bit on common platforms. Suitable for synthetic benchmarks up
    # to ~4B records/derivations. Use 'Q' if larger ids are needed.
    return array("I")


@dataclass
class IndexedTensorArray:
    """A compact indexed sparse tensor optimized for provenance queries.

    Compared with IndexedTensor, this avoids storing Python coordinate tuples and
    Python sets of derivation ids. It stores:
      - one integer array per tensor dimension for the coordinate columns;
      - one dictionary per dimension mapping record id -> integer array of
        derivation ids.

    Querying does not materialize a global derivation set. It streams through the
    posting lists for the selected source ids and collects target ids.
    """

    dims: Tuple[str, ...]
    columns: List[array] = field(init=False)
    by_dim: List[Dict[RecordId, array]] = field(init=False)
    _ncoords: int = 0

    def __post_init__(self) -> None:
        if not self.dims:
            raise ValueError("IndexedTensorArray requires at least one dimension")
        self.columns = [_new_uint_array() for _ in self.dims]
        self.by_dim = [defaultdict(_new_uint_array) for _ in self.dims]

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def ncoords(self) -> int:
        return self._ncoords

    def add(self, coord: Sequence[RecordId]) -> DerivationId:
        if len(coord) != self.ndim:
            raise ValueError(f"coord has length {len(coord)} but tensor has {self.ndim} dims")
        deriv_id = self._ncoords
        values = [int(x) for x in coord]
        for dim, value in enumerate(values):
            self.columns[dim].append(value)
            self.by_dim[dim][value].append(deriv_id)
        self._ncoords += 1
        return deriv_id

    def add_many(self, coords) -> None:
        """
        Vectorized bulk insertion.

        Accepts either:
          - a NumPy array of shape (n, ndim)
          - an iterable of coordinate tuples

        The coordinate columns are appended in bulk.
        The per-dimension inverted index is built by sorting/grouping NumPy arrays,
        avoiding per-coordinate dictionary insertion.
        """
        import numpy as np
        from array import array

        arr = np.asarray(coords, dtype=np.int64)

        if arr.size == 0:
            return

        if arr.ndim != 2 or arr.shape[1] != self.ndim:
            raise ValueError(
                f"coords must have shape (n, {self.ndim}), got {arr.shape}"
            )

        n = arr.shape[0]
        start = self._ncoords
        deriv_ids = np.arange(start, start + n, dtype=np.int64)

        # Append coordinate columns in bulk.
        for dim in range(self.ndim):
            self.columns[dim].extend(arr[:, dim].astype("Q").tolist())

        # Build inverted indexes in bulk per dimension.
        for dim in range(self.ndim):
            values = arr[:, dim]

            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            sorted_derivs = deriv_ids[order]

            # group boundaries
            boundaries = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
            starts = np.r_[0, boundaries]
            ends = np.r_[boundaries, len(sorted_values)]

            dim_index = self.by_dim[dim]

            for s, e in zip(starts, ends):
                key = int(sorted_values[s])
                postings = sorted_derivs[s:e].astype(np.uint64)

                if key in dim_index:
                    dim_index[key].extend(postings.tolist())
                else:
                    dim_index[key] = array("Q", postings.tolist())

        self._ncoords += n

    def derivations_for(self, dim: int, ids: Iterable[RecordId]) -> Set[DerivationId]:
        """Return derivation ids for compatibility with IndexedTensor.

        Most query paths should prefer project(), which avoids materializing this
        set. This method is kept for debugging and API symmetry.
        """
        result: Set[DerivationId] = set()
        idx = self.by_dim[dim]
        seen: Set[int] = set()
        for rid in ids:
            rid = int(rid)
            if rid in seen:
                continue
            seen.add(rid)
            result.update(idx.get(rid, ()))
        return result

    def project(self, source_dim: int, source_ids: Iterable[RecordId], target_dim: int) -> Set[RecordId]:
        idx = self.by_dim[source_dim]
        target_col = self.columns[target_dim]
        result: Set[RecordId] = set()
        seen: Set[int] = set()
        for sid in source_ids:
            sid = int(sid)
            if sid in seen:
                continue
            seen.add(sid)
            for deriv_id in idx.get(sid, ()):  # posting list
                result.add(int(target_col[deriv_id]))
        return result

    def project_many(self, source_dim: int, source_ids: Iterable[RecordId], target_dims: Iterable[int]) -> Dict[int, Set[RecordId]]:
        target_dims = list(target_dims)
        results: Dict[int, Set[RecordId]] = {td: set() for td in target_dims}
        idx = self.by_dim[source_dim]
        seen: Set[int] = set()
        for sid in source_ids:
            sid = int(sid)
            if sid in seen:
                continue
            seen.add(sid)
            for deriv_id in idx.get(sid, ()):  # posting list
                for td in target_dims:
                    results[td].add(int(self.columns[td][deriv_id]))
        return results

    def co_contributors(self, source_dim: int, source_ids: Iterable[RecordId], other_dim: int) -> Set[RecordId]:
        return self.project(source_dim, source_ids, other_dim)

    def approx_memory_bytes(self) -> int:
        """Logical payload size, excluding Python dict/key overhead.

        This mirrors the existing benchmark's logical_memory_mb convention. The
        actual peak memory is reported separately by tracemalloc.
        """
        coord_bytes = sum(col.buffer_info()[1] * col.itemsize for col in self.columns)
        index_bytes = 0
        for dim_index in self.by_dim:
            for postings in dim_index.values():
                index_bytes += postings.buffer_info()[1] * postings.itemsize
        return coord_bytes + index_bytes
