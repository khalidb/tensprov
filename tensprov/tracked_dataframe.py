from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Callable

from tensprov.operations import FilterOperation


class TrackedDataFrame:
    def __init__(self, df: pd.DataFrame, row_ids=None, provenance=None):
        self.df = df.reset_index(drop=True)

        # Each row has a unique ID
        if row_ids is None:
            self.row_ids = list(range(len(df)))
        else:
            self.row_ids = list(row_ids)

        # Provenance = list of operations
        self.provenance = provenance or []

    def filter_mask(self, mask) -> "TrackedDataFrame":
        t_total = time.perf_counter()

        # 1. Underlying pandas operation
        t0 = time.perf_counter()
        mask = pd.Series(mask, index=self.df.index)
        new_df = self.df[mask].reset_index(drop=True)
        pandas_ms = _ms_since(t0)

        # 2. Provenance tensor/index construction
        t0 = time.perf_counter()
        coords = []
        new_row_ids = []

        out_id = 0
        for in_id, keep in zip(self.row_ids, mask):
            if keep:
                coords.append((out_id, in_id))
                new_row_ids.append(in_id)
                out_id += 1

        op = FilterOperation(coords, representation_cls=self.representation_cls)
        tensor_ms = _ms_since(t0)

        # 3. Wrapper construction
        t0 = time.perf_counter()
        result = TrackedDataFrame(
            new_df,
            provenance=self.provenance + [op],
            representation_cls=self.representation_cls,
        )
        wrapper_ms = _ms_since(t0)

        total_ms = _ms_since(t_total)
        other_ms = max(total_ms - pandas_ms - tensor_ms - wrapper_ms, 0.0)

        result._last_profile = {
            "pandas_operation_ms": pandas_ms,
            "tensor_construction_ms": tensor_ms,
            "wrapper_overhead_ms": wrapper_ms,
            "other_ms": other_ms,
        }

        return result

    def filter_rows(self, predicate: Callable[[pd.Series], bool]) -> "TrackedDataFrame":
        """
        Apply a row filter and capture provenance.
        """
        mask = self.df.apply(predicate, axis=1)

        new_df = self.df[mask].reset_index(drop=True)
        new_ids = [rid for rid, keep in zip(self.row_ids, mask) if keep]

        # Build provenance coords: (out_id, in_id)
        coords = [(out_id, in_id) for out_id, in_id in enumerate(new_ids)]

        op = FilterOperation(coords)

        return TrackedDataFrame(
            new_df,
            row_ids=list(range(len(new_df))),  # new IDs for output
            provenance=self.provenance + [op],
        )

    def assign_column(self, name, values):
        new_df = self.df.copy()
        new_df[name] = values

        op = VerticalAugmentationOperation(
            input_columns=self.df.columns,
            output_columns=new_df.columns,
            derived_columns=[name],
            source_columns=[],
        )

        return TrackedDataFrame(
            new_df,
            provenance=self.provenance + [op],
            representation_cls=self.representation_cls,
        )


    def map_column(self, column, func):
        new_df = self.df.copy()
        new_df[column] = new_df[column].map(func)

        op = DataTransformationOperation(
            input_columns=self.df.columns,
            output_columns=new_df.columns,
            transformed_columns=[column],
            source_columns=[column],
        )

        return TrackedDataFrame(
            new_df,
            provenance=self.provenance + [op],
            representation_cls=self.representation_cls,
        )

    def impute_column(self, column, strategy="mean"):
        new_df = self.df.copy()

        if strategy == "mean":
            value = new_df[column].mean()
        elif strategy == "median":
            value = new_df[column].median()
        else:
            value = new_df[column].mode().iloc[0]

        new_df[column] = new_df[column].fillna(value)

        op = ImputeOperation(
            input_columns=self.df.columns,
            output_columns=new_df.columns,
            column=column,
            strategy=strategy,
            fitted_value=value,
        )

        return TrackedDataFrame(
            new_df,
            provenance=self.provenance + [op],
            representation_cls=self.representation_cls,
        )  

    def append(self, other):
        new_df = pd.concat([self.df, other.df], ignore_index=True)

        n_left = len(self.df)
        n_right = len(other.df)

        left_output_indices = np.arange(n_left, dtype=np.uint64)
        left_input_indices = np.arange(n_left, dtype=np.uint64)

        right_output_indices = np.arange(
            n_left,
            n_left + n_right,
            dtype=np.uint64,
        )
        right_input_indices = np.arange(n_right, dtype=np.uint64)

        left_coords = np.column_stack((left_output_indices, left_input_indices))
        right_coords = np.column_stack((right_output_indices, right_input_indices))

        left_op = FilterOperation(
            left_coords,
            name="append_left",
            representation_cls=self.representation_cls,
        )

        right_op = FilterOperation(
            right_coords,
            name="append_right",
            representation_cls=self.representation_cls,
        )

        return TrackedDataFrame(
            new_df,
            provenance=self.provenance + other.provenance + [left_op, right_op],
            representation_cls=self.representation_cls,
        )  
        
    def duplicate_rows(self, times=2):
        """
        Duplicate each row multiple times.
        """
        import pandas as pd

        new_df = pd.concat([self.df] * times, ignore_index=True)

        coords = []
        for i in range(times):
            for in_id in self.df.index:
                out_id = i * len(self.df) + int(in_id)
                coords.append((out_id, int(in_id)))

        op = FilterOperation(
            coords,
            representation_cls=self.representation_cls,
        )

        return TrackedDataFrame(
            new_df,
            provenance=self.provenance + [op],
            representation_cls=self.representation_cls,
        )            
