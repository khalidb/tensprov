"""Common query helpers for comparable provenance structures."""
from __future__ import annotations

from typing import Iterable, Set


def forward(index, input_dim: int, input_ids: Iterable[int], output_dim: int = 0) -> Set[int]:
    return index.project(input_dim, input_ids, output_dim)


def backward(index, output_ids: Iterable[int], input_dim: int, output_dim: int = 0) -> Set[int]:
    return index.project(output_dim, output_ids, input_dim)


def co_contribution(index, fixed_dim: int, fixed_ids: Iterable[int], other_dim: int) -> Set[int]:
    return index.co_contributors(fixed_dim, fixed_ids, other_dim)
