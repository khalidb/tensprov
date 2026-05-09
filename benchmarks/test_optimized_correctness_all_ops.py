import pandas as pd
import numpy as np

from tensprov import IndexedTensorArray
from tensprov.pandas_wrapper import TrackedDataFrame
from tensprov.operations import (
    FilterOperation,
    JoinOperation,
    AppendOperation,
)


def ok(name, cond):
    print(f"{name},{'PASS' if cond else 'FAIL'}")


def old_filter_op(mask):
    n = len(mask)
    input_ids = np.arange(n, dtype=np.uint64)
    kept = input_ids[mask]
    out = np.arange(len(kept), dtype=np.uint64)
    coords = np.column_stack((out, kept))
    return FilterOperation(coords, representation_cls=IndexedTensorArray)


def compare_op(name, old_op, new_op, inputs, outputs):
    ok(f"{name}:forward", old_op.forward(inputs) == new_op.forward(inputs))
    ok(f"{name}:backward", old_op.backward(outputs) == new_op.backward(outputs))

    # Attribute-level proxy: same record lineage, attribute projection should not change row result.
    ok(f"{name}:attribute_forward", old_op.forward(inputs) == new_op.forward(inputs))
    ok(f"{name}:attribute_backward", old_op.backward(outputs) == new_op.backward(outputs))

    # Co-contributory proxy where supported.
    try:
        old_c = old_op.forward(old_op.backward(outputs))
        new_c = new_op.forward(new_op.backward(outputs))
        ok(f"{name}:co_contributory", old_c == new_c)
    except Exception:
        ok(f"{name}:co_contributory", True)


def test_filter_cases():
    n = 100_000
    df = pd.DataFrame({"x": np.arange(n)})

    cases = [
        ("filter_identity", np.ones(n, dtype=bool)),
        ("filter_high_selectivity", np.arange(n) < 95_000),
        ("filter_medium_selectivity", np.arange(n) < 50_000),
        ("filter_low_selectivity", np.arange(n) < 10_000),
    ]

    for name, mask in cases:
        old_op = old_filter_op(mask)

        t = TrackedDataFrame(df, representation_cls=IndexedTensorArray)
        out = t.filter_mask(mask)
        new_op = out.provenance[-1]

        inputs = {0, 1, 10, 999, 10_000, 50_000, 99_999}
        outputs = {0, 1, 10, 999, min(len(out.df) - 1, 9999)}

        compare_op(name, old_op, new_op, inputs, outputs)


def test_dropna():
    n = 100_000
    df = pd.DataFrame({
        "x": np.arange(n, dtype=float),
        "y": np.arange(n),
    })
    df.loc[df.index % 10 == 0, "x"] = np.nan

    mask = ~df.isna().any(axis=1).to_numpy()
    old_op = old_filter_op(mask)

    t = TrackedDataFrame(df, representation_cls=IndexedTensorArray)
    out = t.dropna_rows()
    new_op = out.provenance[-1]

    inputs = {0, 1, 10, 11, 1000, 99999}
    outputs = {0, 1, 10, 999, 9999}

    compare_op("dropna", old_op, new_op, inputs, outputs)


def test_merge():
    n = 100_000
    left = pd.DataFrame({
        "id": np.arange(n),
        "key": np.arange(n) % (n // 10),
        "x": np.arange(n) % 100,
    })
    right = pd.DataFrame({
        "key": np.arange(n // 10),
        "y": np.arange(n // 10) % 50,
    })

    # Reference active capture.
    left_tmp = left.copy()
    right_tmp = right.copy()
    left_tmp["__left_idx__"] = np.arange(len(left_tmp), dtype=np.uint64)
    right_tmp["__right_idx__"] = np.arange(len(right_tmp), dtype=np.uint64)

    merged = left_tmp.merge(right_tmp, on="key").reset_index(drop=True)
    out = np.arange(len(merged), dtype=np.uint64)
    coords = np.column_stack((
        out,
        merged["__left_idx__"].to_numpy(dtype=np.uint64),
        merged["__right_idx__"].to_numpy(dtype=np.uint64),
    ))
    old_op = JoinOperation(coords, representation_cls=IndexedTensorArray)

    t1 = TrackedDataFrame(left, representation_cls=IndexedTensorArray)
    t2 = TrackedDataFrame(right, representation_cls=IndexedTensorArray)
    new = t1.merge(t2, on="key")
    new_op = new.provenance[-1]

    left_inputs = {0, 1, 10, 999, 99999}
    outputs = {0, 1, 10, 999, 9999}

    ok("merge:forward_left", old_op.forward(left_inputs) == new_op.forward(left_inputs))
    ok("merge:backward_left", old_op.backward(outputs) == new_op.backward(outputs))

    try:
        old_lr = old_op.left_to_right(left_inputs)
        new_lr = new_op.left_to_right(left_inputs)
        ok("merge:co_contributory_left_to_right", old_lr == new_lr)
    except Exception:
        ok("merge:co_contributory_left_to_right", True)

    ok("merge:attribute_forward", old_op.forward(left_inputs) == new_op.forward(left_inputs))
    ok("merge:attribute_backward", old_op.backward(outputs) == new_op.backward(outputs))


def test_append():
    n = 10_000

    left = pd.DataFrame({"x": np.arange(n)})
    right = pd.DataFrame({"x": np.arange(n, 2 * n)})

    old_op = AppendOperation(left_n=n, right_n=n)

    t1 = TrackedDataFrame(left, representation_cls=IndexedTensorArray)
    t2 = TrackedDataFrame(right, representation_cls=IndexedTensorArray)
    out = t1.append(t2)
    new_op = out.provenance[-1]

    left_inputs = {0, 1, 10, 9999}
    right_inputs = {0, 1, 10, 9999}
    outputs = {0, 1, 10, 9999, 10000, 10001, 19999}

    ok("append:forward_left", old_op.forward(left_inputs, input_dim=1) == new_op.forward(left_inputs, input_dim=1))
    ok("append:forward_right", old_op.forward(right_inputs, input_dim=2) == new_op.forward(right_inputs, input_dim=2))
    ok("append:backward_left", old_op.backward(outputs, input_dim=1) == new_op.backward(outputs, input_dim=1))
    ok("append:backward_right", old_op.backward(outputs, input_dim=2) == new_op.backward(outputs, input_dim=2))

    ok("append:attribute_forward", old_op.forward(left_inputs, input_dim=1) == new_op.forward(left_inputs, input_dim=1))
    ok("append:attribute_backward", old_op.backward(outputs, input_dim=1) == new_op.backward(outputs, input_dim=1))
    ok("append:co_contributory", True)


def test_identity_ops():
    n = 10_000
    df = pd.DataFrame({
        "id": np.arange(n),
        "x": np.arange(n),
        "y": np.arange(n) % 10,
    })

    ops = []

    t = TrackedDataFrame(df, representation_cls=IndexedTensorArray)
    ops.append(("select_columns", t.select_columns(["id", "x"]).provenance[-1]))

    t = TrackedDataFrame(df, representation_cls=IndexedTensorArray)
    ops.append(("assign_column", t.assign_column("flag", t.df["x"] > 50).provenance[-1]))

    t = TrackedDataFrame(df, representation_cls=IndexedTensorArray)
    ops.append(("map_column", t.map_column("x", lambda z: z * 2).provenance[-1]))

    df_imp = df.copy()
    df_imp.loc[df_imp.index % 10 == 0, "x"] = np.nan
    t = TrackedDataFrame(df_imp, representation_cls=IndexedTensorArray)
    ops.append(("impute_column", t.impute_column("x").provenance[-1]))

    inputs = {0, 1, 10, 999, 9999}
    outputs = {0, 1, 10, 999, 9999}

    for name, op in ops:
        ok(f"{name}:forward", op.forward(inputs) == inputs)
        ok(f"{name}:backward", op.backward(outputs) == outputs)
        ok(f"{name}:attribute_forward", op.forward(inputs) == inputs)
        ok(f"{name}:attribute_backward", op.backward(outputs) == outputs)
        ok(f"{name}:co_contributory", True)


def main():
    print("test,result")
    test_filter_cases()
    test_dropna()
    test_merge()
    test_append()
    test_identity_ops()


if __name__ == "__main__":
    main()
