from __future__ import annotations

import numpy as np


class AdaptiveFilterIndex:
    """
    Adaptive provenance index for FilterOperation/dropna with dims=("out", "in").

    Case 1: identity, no rows dropped.
    Case 2: high selectivity, store dropped input positions only.
    Case 3: low/medium selectivity, store kept input indices explicitly.

    External interface remains:
      project(source_dim, source_ids, target_dim)
      derivations_for(dim, ids)
      approx_memory_bytes()
    """

    def __init__(self, input_index_array, output_index_array):
        self.dims = ("out", "in")
        self.ndim = 2

        self.n_input = len(input_index_array)
        self.n_output = len(output_index_array)
        self.n_dropped = self.n_input - self.n_output

        self.case = None

        # Compatibility fields
        self.columns = None
        self.by_dim = None

        # Case-specific storage
        self.input_indices = None
        self.output_indices = None
        self.dropped_positions = None
        self._surviving_positions_cache = None

        self._build(input_index_array, output_index_array)

    @property
    def ncoords(self):
        return self.n_output

    def _build(self, input_index_array, output_index_array):
        # Case 1: no rows dropped. Zero tensor/index construction.
        if self.n_dropped == 0:
            self.case = 1
            self.columns = None
            self.by_dim = None
            return

        input_index_array = np.asarray(input_index_array, dtype=np.uint64)
        output_index_array = np.asarray(output_index_array, dtype=np.uint64)

        # Case 2: high selectivity. More rows kept than dropped.
        # Store only dropped input positions.
        if self.n_dropped < self.n_output:
            self.case = 2

            # Fast path for normal TrackedDataFrame row ids:
            # input ids are 0..n-1 and output ids are surviving row ids.
            if (
                input_index_array.size == self.n_input
                and input_index_array[0] == 0
                and input_index_array[-1] == self.n_input - 1
                and np.all(input_index_array == np.arange(self.n_input, dtype=np.uint64))
            ):
                keep = np.zeros(self.n_input, dtype=bool)
                keep[output_index_array] = True
                self.dropped_positions = np.nonzero(~keep)[0].astype(np.uint64)
            else:
                # General fallback.
                self.dropped_positions = np.setdiff1d(
                    input_index_array,
                    output_index_array,
                    assume_unique=True,
                ).astype(np.uint64, copy=False)

            self.columns = None
            self.by_dim = None
            return



        # Case 3: low/medium selectivity. More rows dropped than kept.
        # Extract kept row indices directly from output_df.index.
        self.case = 3
        self.output_indices = np.arange(self.n_output, dtype=np.uint64)
        self.input_indices = output_index_array.astype(np.uint64, copy=True)

        self.columns = [self.output_indices, self.input_indices]
        self.by_dim = None

    def _surviving_positions(self):
        """
        Lazily reconstruct surviving input positions for Case 2.
        Only needed at query time.
        """
        if self._surviving_positions_cache is None:
            keep = np.ones(self.n_input, dtype=bool)
            keep[self.dropped_positions] = False
            self._surviving_positions_cache = np.nonzero(keep)[0].astype(np.uint64)
        return self._surviving_positions_cache

    def project(self, source_dim, source_ids, target_dim):
        source_ids = np.asarray(list(source_ids), dtype=np.uint64)

        if source_ids.size == 0:
            return set()

        # Case 1: identity mapping.
        if self.case == 1:
            if source_dim == 0 and target_dim == 1:
                valid = source_ids[source_ids < self.n_output]
                return set(valid.astype(int).tolist())

            if source_dim == 1 and target_dim == 0:
                valid = source_ids[source_ids < self.n_output]
                return set(valid.astype(int).tolist())

            raise NotImplementedError(
                f"Unsupported projection source_dim={source_dim}, target_dim={target_dim}"
            )

        # Case 2: sparse dropped representation.
        if self.case == 2:
            surviving_positions = self._surviving_positions()

            if source_dim == 0 and target_dim == 1:
                valid = source_ids[source_ids < self.n_output]
                input_ids = surviving_positions[valid]
                return set(input_ids.astype(int).tolist())

            if source_dim == 1 and target_dim == 0:
                valid = source_ids[source_ids < self.n_input]

                is_dropped = np.isin(valid, self.dropped_positions)
                survivors = valid[~is_dropped]

                out_ids = np.searchsorted(surviving_positions, survivors)
                return set(out_ids.astype(int).tolist())

            raise NotImplementedError(
                f"Unsupported projection source_dim={source_dim}, target_dim={target_dim}"
            )

        # Case 3: explicit kept representation.
        if self.case == 3:
            if source_dim == 0 and target_dim == 1:
                valid = source_ids[source_ids < self.n_output]
                input_ids = self.input_indices[valid]
                return set(input_ids.astype(int).tolist())

            if source_dim == 1 and target_dim == 0:
                mask = np.isin(self.input_indices, source_ids)
                return set(self.output_indices[mask].astype(int).tolist())

            raise NotImplementedError(
                f"Unsupported projection source_dim={source_dim}, target_dim={target_dim}"
            )

        return set()

    def derivations_for(self, dim, ids):
        """
        Compatibility method.
        For filter/dropna mappings, derivation id == output id.
        """
        ids = np.asarray(list(ids), dtype=np.uint64)

        if ids.size == 0:
            return set()

        if dim == 0:
            valid = ids[ids < self.n_output]
            return set(valid.astype(int).tolist())

        if dim == 1:
            if self.case == 1:
                valid = ids[ids < self.n_output]
                return set(valid.astype(int).tolist())

            if self.case == 2:
                surviving_positions = self._surviving_positions()
                mask = np.isin(surviving_positions, ids)
                return set(np.nonzero(mask)[0].astype(int).tolist())

            if self.case == 3:
                mask = np.isin(self.input_indices, ids)
                return set(self.output_indices[mask].astype(int).tolist())

        return set()

    def approx_memory_bytes(self):
        if self.case == 1:
            return 0

        if self.case == 2:
            return int(self.dropped_positions.nbytes)

        if self.case == 3:
            return int(self.output_indices.nbytes + self.input_indices.nbytes)

        return 0