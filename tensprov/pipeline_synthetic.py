"""Synthetic multi-step pipeline generators for provenance benchmarks."""
from __future__ import annotations

from typing import List, Sequence, Tuple

Coord = Tuple[int, int, int]


def synthetic_join_expansion_pipeline(
    *,
    n_start: int,
    depth: int,
    fanout: int,
    n_right: int,
    seed: int = 0,
) -> List[List[Coord]]:
    """Generate a chain of join-like provenance tensors.

    Each step k has coordinates of the form:
        (out_k, in_{k-1}, right_k)

    The output ids of step k become the input ids of step k+1. This creates
    controlled expansion: a single starting record reaches fanout**depth final
    records when every step has the same fanout.

    Parameters
    ----------
    n_start:
        Number of records in the initial dataset D0.
    depth:
        Number of join-expansion steps.
    fanout:
        Number of output records generated from each input record at each step.
    n_right:
        Number of synthetic right-side records available at each step.
    seed:
        Deterministic offset used when assigning right-side ids.
    """
    if n_start <= 0:
        raise ValueError("n_start must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if fanout <= 0:
        raise ValueError("fanout must be positive")
    if n_right <= 0:
        raise ValueError("n_right must be positive")

    current_ids = list(range(n_start))
    pipeline: List[List[Coord]] = []

    for step in range(depth):
        coords: List[Coord] = []
        next_ids: List[int] = []
        out_id = 0
        for in_id in current_ids:
            for j in range(fanout):
                # Deterministic but non-trivial right-side assignment.
                right_id = (in_id * 1_000_003 + j * 97 + step * 7_919 + seed) % n_right
                coords.append((out_id, in_id, right_id))
                next_ids.append(out_id)
                out_id += 1
        pipeline.append(coords)
        current_ids = next_ids

    return pipeline


def sample_first_ids(n: int, count: int) -> List[int]:
    """Deterministic helper for pipeline benchmarks."""
    return list(range(min(n, max(0, count))))
