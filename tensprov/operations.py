from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from tensprov.indexed_tensor_array import IndexedTensorArray
from tensprov.queries import forward, backward, co_contribution
from tensprov.adaptive_filter_index import AdaptiveFilterIndex


class ProvenanceOperation:
    def __init__(
        self,
        name: str,
        dims: Tuple[str, ...],
        coords: Sequence[Tuple[int, ...]],
        representation_cls=IndexedTensorArray,
    ):   

        self.representation_cls = representation_cls
        self.name = name
        self.dims = dims
        self.index = self.representation_cls(dims)
        self.index.add_many(coords)

    def forward(self, input_ids: Iterable[int], input_dim: int = 1, output_dim: int = 0) -> set[int]:
        return forward(
            self.index,
            input_dim=input_dim,
            input_ids=set(input_ids),
            output_dim=output_dim,
        )

    def backward(self, output_ids: Iterable[int], input_dim: int = 1, output_dim: int = 0) -> set[int]:
        return backward(
            self.index,
            output_ids=set(output_ids),
            input_dim=input_dim,
            output_dim=output_dim,
        )
    def approx_memory_bytes(self):
        if hasattr(self, "index") and hasattr(self.index, "approx_memory_bytes"):
            return self.index.approx_memory_bytes()
        # fallback
        return 0    


class FilterOperation(ProvenanceOperation):
    """
    One-input operation: out <- input.
    Coordinates are (out_id, input_id).
    """

    def __init__(self, coords: Sequence[Tuple[int, int]], name: str = "filter",representation_cls=IndexedTensorArray):
        super().__init__(name=name, dims=("out", "in"), coords=coords, representation_cls=representation_cls)

class AdaptiveFilterOperation:
    """
    Filter/dropna operation using adaptive filter index.
    External interface matches ProvenanceOperation.
    """

    def __init__(self, input_index_array, output_index_array, name="filter"):
        self.name = name
        self.dims = ("out", "in")
        self.index = AdaptiveFilterIndex(input_index_array, output_index_array)

    def forward(self, input_ids, input_dim: int = 1, output_dim: int = 0):
        return self.index.project(input_dim, set(input_ids), output_dim)

    def backward(self, output_ids, input_dim: int = 1, output_dim: int = 0):
        return self.index.project(output_dim, set(output_ids), input_dim)

    def approx_memory_bytes(self):
        return self.index.approx_memory_bytes()

class JoinOperation(ProvenanceOperation):
    """
    Two-input operation: out <- left, right.
    Coordinates are (out_id, left_id, right_id).
    """

    def __init__(self, coords: Sequence[Tuple[int, int, int]], name: str = "join", representation_cls=IndexedTensorArray):
        super().__init__(name=name, dims=("out", "left", "right"), coords=coords, representation_cls=representation_cls)

    def left_to_out(self, left_ids: Iterable[int]) -> set[int]:
        return self.forward(left_ids, input_dim=1, output_dim=0)

    def right_to_out(self, right_ids: Iterable[int]) -> set[int]:
        return self.forward(right_ids, input_dim=2, output_dim=0)

    def out_to_left(self, output_ids: Iterable[int]) -> set[int]:
        return self.backward(output_ids, input_dim=1, output_dim=0)

    def out_to_right(self, output_ids: Iterable[int]) -> set[int]:
        return self.backward(output_ids, input_dim=2, output_dim=0)

    def left_to_right(self, left_ids: Iterable[int]) -> set[int]:
        return co_contribution(
            self.index,
            fixed_dim=1,
            fixed_ids=set(left_ids),
            other_dim=2,
        )

        
class IdentityOperation:
    """
    Logical identity row-provenance operation.

    Used for operations where output row i depends on input row i:
    - vertical reduction
    - vertical augmentation
    - data transformation
    - imputation

    No tensor/index is materialized.
    """

    def __init__(self, name: str, metadata=None, contextual: bool = False):
        self.name = name
        self.metadata = metadata or {}
        self.contextual = contextual

    def forward(self, input_ids, input_dim: int = 1, output_dim: int = 0):
        return set(input_ids)

    def backward(self, output_ids, input_dim: int = 1, output_dim: int = 0):
        return set(output_ids)

    def approx_memory_bytes(self):
        return 0


class VerticalReductionOperation(IdentityOperation):
    def __init__(self, input_columns, output_columns, kept_columns, name: str = "vertical_reduction"):
        super().__init__(
            name=name,
            metadata={
                "input_columns": list(input_columns),
                "output_columns": list(output_columns),
                "kept_columns": list(kept_columns),
            },
            contextual=False,
        )


class VerticalAugmentationOperation(IdentityOperation):
    def __init__(self, input_columns, output_columns, derived_columns, source_columns, name: str = "vertical_augmentation"):
        super().__init__(
            name=name,
            metadata={
                "input_columns": list(input_columns),
                "output_columns": list(output_columns),
                "derived_columns": list(derived_columns),
                "source_columns": list(source_columns),
            },
            contextual=False,
        )


class DataTransformationOperation(IdentityOperation):
    def __init__(self, input_columns, output_columns, transformed_columns, source_columns, name: str = "data_transformation"):
        super().__init__(
            name=name,
            metadata={
                "input_columns": list(input_columns),
                "output_columns": list(output_columns),
                "transformed_columns": list(transformed_columns),
                "source_columns": list(source_columns),
            },
            contextual=False,
        )


class ImputeOperation(IdentityOperation):
    def __init__(
        self,
        input_columns,
        output_columns,
        column,
        strategy,
        fitted_value,
        name: str = "imputation",
    ):
        super().__init__(
            name=name,
            metadata={
                "input_columns": list(input_columns),
                "output_columns": list(output_columns),
                "column": column,
                "strategy": strategy,
                "fitted_value": fitted_value,
            },
            contextual=True,
        )

class AppendOperation:
    """
    Append operation:
      left row i  -> output row i
      right row j -> output row left_n + j

    Uses O(1) metadata instead of materializing a tensor.
    """

    def __init__(self, left_n: int, right_n: int, name: str = "append"):
        self.name = name
        self.left_n = int(left_n)
        self.right_n = int(right_n)
        self.dims = ("out", "left", "right")
        self.contextual = False

    def forward(self, input_ids, input_dim: int = 1, output_dim: int = 0):
        ids = {int(x) for x in input_ids}

        if output_dim != 0:
            raise NotImplementedError("AppendOperation only supports projection to output")

        if input_dim == 1:  # left input
            return {i for i in ids if 0 <= i < self.left_n}

        if input_dim == 2:  # right input
            return {self.left_n + i for i in ids if 0 <= i < self.right_n}

        return set()

    def backward(self, output_ids, input_dim: int = 1, output_dim: int = 0):
        ids = {int(x) for x in output_ids}

        if output_dim != 0:
            raise NotImplementedError("AppendOperation assumes output_dim=0")

        if input_dim == 1:  # output -> left
            return {i for i in ids if 0 <= i < self.left_n}

        if input_dim == 2:  # output -> right
            return {i - self.left_n for i in ids if self.left_n <= i < self.left_n + self.right_n}

        return set()

    def approx_memory_bytes(self):
        return 16

        