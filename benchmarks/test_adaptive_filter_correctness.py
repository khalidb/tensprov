import pandas as pd
import numpy as np

from tensprov import IndexedTensorArray
from tensprov.pandas_wrapper import TrackedDataFrame
from tensprov.operations import FilterOperation


def old_filter_result(df, mask):
    input_ids = np.arange(len(df), dtype=np.uint64)
    kept = input_ids[mask]
    output_ids = np.arange(len(kept), dtype=np.uint64)
    coords = np.column_stack((output_ids, kept))
    return FilterOperation(coords, representation_cls=IndexedTensorArray)


def new_filter_result(df, mask):
    t = TrackedDataFrame(df, representation_cls=IndexedTensorArray)
    return t.filter_mask(mask).provenance[-1]


def check_case(name, keep_count):
    n = 100_000
    df = pd.DataFrame({"x": np.arange(n)})

    mask = np.zeros(n, dtype=bool)
    mask[:keep_count] = True

    old_op = old_filter_result(df, mask)
    new_op = new_filter_result(df, mask)

    test_inputs = {0, 1, 10, 999, keep_count - 1, keep_count, n - 1}
    test_outputs = {0, 1, 10, 999, max(0, keep_count - 1)}

    old_f = old_op.forward(test_inputs)
    new_f = new_op.forward(test_inputs)

    old_b = old_op.backward(test_outputs)
    new_b = new_op.backward(test_outputs)

    print(
        name,
        "case=", new_op.index.case,
        "forward_ok=", old_f == new_f,
        "backward_ok=", old_b == new_b,
        "memory_bytes=", new_op.approx_memory_bytes(),
    )


def main():
    check_case("identity", 100_000)
    check_case("high_selectivity", 95_000)
    check_case("medium_selectivity", 50_000)
    check_case("low_selectivity", 10_000)


if __name__ == "__main__":
    main()
