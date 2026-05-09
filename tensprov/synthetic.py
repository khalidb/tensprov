"""Synthetic provenance workloads for set-oriented experiments."""
from __future__ import annotations

import random
from typing import List, Tuple

Coord = Tuple[int, ...]


def filter_coords(n_input: int, selectivity: float, seed: int = 0) -> List[Coord]:
    rng = random.Random(seed)
    coords: List[Coord] = []
    out = 0
    for i in range(n_input):
        if rng.random() < selectivity:
            coords.append((out, i))
            out += 1
    return coords


def one_to_one_coords(n: int) -> List[Coord]:
    return [(i, i) for i in range(n)]


def synthetic_join_coords(n_left: int, n_right: int, fanout: int = 1, seed: int = 0) -> List[Coord]:
    if fanout < 1:
        raise ValueError("fanout must be >= 1")
    rng = random.Random(seed)
    coords: List[Coord] = []
    out = 0
    for left in range(n_left):
        rights = rng.sample(range(n_right), k=min(fanout, n_right))
        for right in rights:
            coords.append((out, left, right))
            out += 1
    return coords


def sample_ids(max_id_exclusive: int, fraction: float, seed: int = 0) -> List[int]:
    if max_id_exclusive <= 0:
        return []
    k = max(1, int(max_id_exclusive * fraction))
    rng = random.Random(seed)
    return rng.sample(range(max_id_exclusive), k=min(k, max_id_exclusive))
