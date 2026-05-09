import pandas as pd
from typing import Callable, List
import numpy as np

from tensprov.operations import AdaptiveFilterOperation

from tensprov.indexed_tensor_array import IndexedTensorArray
from tensprov.operations import (
    FilterOperation,
    JoinOperation,
    VerticalReductionOperation,
    VerticalAugmentationOperation,
    DataTransformationOperation,
    ImputeOperation,
    AdaptiveFilterOperation,
    AppendOperation,
)

import time


def _ms_since(t0):
    return (time.perf_counter() - t0) * 1000


class TrackedDataFrame:
    def __init__(self, df: pd.DataFrame, provenance=None, representation_cls=None):
        self.df = df.reset_index(drop=True)
        self.row_ids = list(self.df.index)
        self.provenance: List = provenance or []
        self.representation_cls = representation_cls or IndexedTensorArray
        self._last_profile = None



    def select_columns(self, columns):
        new_df = self.df[list(columns)].copy()

        op = VerticalReductionOperation(
            input_columns=self.df.columns,
            output_columns=new_df.columns,
            kept_columns=columns,
        )

        return TrackedDataFrame(
            new_df,
            provenance=self.provenance + [op],
            representation_cls=self.representation_cls,
        )

    def drop_columns(self, columns):
        new_df = self.df.drop(columns=list(columns))

        kept = [c for c in self.df.columns if c not in columns]

        op = VerticalReductionOperation(
            input_columns=self.df.columns,
            output_columns=new_df.columns,
            kept_columns=kept,
        )

        return TrackedDataFrame(
            new_df,
            provenance=self.provenance + [op],
            representation_cls=self.representation_cls,
        )



    def filter_mask(self, mask) -> "TrackedDataFrame":
        t_total = time.perf_counter()

        # 1. Underlying pandas operation
        t0 = time.perf_counter()
        mask = pd.Series(mask, index=self.df.index).to_numpy(dtype=bool)
        new_df = self.df[mask].reset_index(drop=True)
        pandas_ms = _ms_since(t0)

        # 2. Adaptive provenance index construction
        t0 = time.perf_counter()

        n_input = len(self.df)
        n_output = len(new_df)

        if n_input == n_output:
            # Identity case: avoid creating/slicing large numpy arrays.
            op = AdaptiveFilterOperation(
                input_index_array=range(n_input),
                output_index_array=range(n_output),
                name="filter",
            )
        else:
            input_index_array = np.asarray(self.row_ids, dtype=np.uint64)
            output_index_array = input_index_array[mask]

            op = AdaptiveFilterOperation(
                input_index_array=input_index_array,
                output_index_array=output_index_array,
                name="filter",
            )

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


    def filter(self, predicate: Callable[[pd.Series], bool]) -> "TrackedDataFrame":
        # Backward-compatible path.
        # This still evaluates the predicate row-wise, but tensor construction is vectorized.
        mask = self.df.apply(predicate, axis=1)
        return self.filter_mask(mask)


    def dropna_rows(self, subset=None):
        """
        Drop rows with missing values.

        Uses the same adaptive strategy as filter:
        Case 1 — no rows dropped: zero tensor construction
        Case 2 — high selectivity: store dropped positions only
        Case 3 — low selectivity: store kept row indices from output index
        """
        t_total = time.perf_counter()

        # 1. Underlying pandas operation
        t0 = time.perf_counter()

        if subset is None:
            out = self.df.dropna()
        else:
            out = self.df.dropna(subset=subset)

        pandas_ms = _ms_since(t0)

        # 2. Adaptive provenance index construction
        t0 = time.perf_counter()

        n_input = len(self.df)
        n_output = len(out)

        if n_input == n_output:
            # Case 1: identity. Avoid numpy array construction entirely.
            op = AdaptiveFilterOperation(
                input_index_array=range(n_input),
                output_index_array=range(n_output),
                name="dropna",
            )
        else:
            # Cases 2 and 3:
            # Use out.index directly; it already contains kept row ids.
            input_index_array = np.asarray(self.row_ids, dtype=np.uint64)
            output_index_array = out.index.to_numpy(dtype=np.uint64)

            op = AdaptiveFilterOperation(
                input_index_array=input_index_array,
                output_index_array=output_index_array,
                name="dropna",
            )

        tensor_ms = _ms_since(t0)

        # 3. Wrapper construction
        t0 = time.perf_counter()
        new_df = out.reset_index(drop=True)

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



    def merge(self, other: "TrackedDataFrame", on: str) -> "TrackedDataFrame":
        t_total = time.perf_counter()

        # 1. Underlying pandas operation with temporary row identifiers.
        # This lets pandas compute the join mapping for us.
        t0 = time.perf_counter()

        left_tmp = self.df.copy()
        right_tmp = other.df.copy()

        left_tmp["__tp_left_id__"] = np.asarray(self.row_ids, dtype=np.int64)
        right_tmp["__tp_right_id__"] = np.asarray(other.row_ids, dtype=np.int64)

        merged = left_tmp.merge(right_tmp, on=on).reset_index(drop=True)

        left_ids = merged["__tp_left_id__"].to_numpy(dtype=np.int64)
        right_ids = merged["__tp_right_id__"].to_numpy(dtype=np.int64)

        user_cols = [
            c for c in merged.columns
            if c not in ("__tp_left_id__", "__tp_right_id__")
        ]
        new_df = merged[user_cols].copy()

        pandas_ms = _ms_since(t0)

        # 2. Vectorized provenance tensor/index construction
        t0 = time.perf_counter()

        output_ids = np.arange(len(merged), dtype=np.int64)
        coords = np.column_stack((output_ids, left_ids, right_ids))

        op = JoinOperation(coords, representation_cls=self.representation_cls)

        tensor_ms = _ms_since(t0)

        # 3. Wrapper construction
        t0 = time.perf_counter()
        result = TrackedDataFrame(
            new_df,
            provenance=self.provenance + other.provenance + [op],
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

    def append(self, other: "TrackedDataFrame") -> "TrackedDataFrame":
        t_total = time.perf_counter()

        # 1. Underlying pandas operation
        t0 = time.perf_counter()
        new_df = pd.concat([self.df, other.df], ignore_index=True)
        pandas_ms = _ms_since(t0)

        # 2. O(1) provenance construction
        t0 = time.perf_counter()
        op = AppendOperation(
            left_n=len(self.df),
            right_n=len(other.df),
        )
        tensor_ms = _ms_since(t0)

        # 3. Wrapper construction
        t0 = time.perf_counter()
        result = TrackedDataFrame(
            new_df,
            provenance=self.provenance + other.provenance + [op],
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
