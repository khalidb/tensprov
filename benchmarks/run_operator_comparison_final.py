import gc
import time
import sys
import pandas as pd
import numpy as np

from tensprov.pandas_wrapper import TrackedDataFrame
from tensprov.provenance_engine import ProvenanceEngine
from tensprov import IndexedTensorArray, IndexedTensor, COOTensor, ProvenanceGraph
from tensprov.relational_baseline import RelationalProvenance


SIZES = [1_000, 10_000, 50_000, 100_000]
RUNS = 10
WARMUP = 1
QUERY_SIZE = 100


REPRESENTATIONS = {
    "NoProv": None,
    "IndexedTensorArray": IndexedTensorArray,
    "IndexedTensor": IndexedTensor,
    "COOTensor": COOTensor,
    "ProvenanceGraph": ProvenanceGraph,
    "RelationalProvenance": RelationalProvenance,
}


def avg_time(fn):
    times = []
    last = None
    for i in range(WARMUP + RUNS):
        gc.collect()
        t0 = time.perf_counter()
        last = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= WARMUP:
            times.append(elapsed)
    return sum(times) / len(times), last


def provenance_memory_mb(tdf):
    total = 0
    for op in getattr(tdf, "provenance", []):
        if hasattr(op, "approx_memory_bytes"):
            total += op.approx_memory_bytes()
        else:
            total += sys.getsizeof(op)
    return total / (1024 * 1024)


def plain_result(df):
    class Plain:
        def __init__(self, df):
            self.df = df.reset_index(drop=True)
            self.provenance = []
    return Plain(df)


def make_filter(size, rep):
    df = pd.DataFrame({"id": np.arange(size), "x": np.arange(size) % 100})
    if rep is None:
        return plain_result(df[df["x"] > 50])
    t = TrackedDataFrame(df, representation_cls=rep)
    return t.filter_mask(t.df["x"] > 50)


def make_dropna(size, rep):
    df = pd.DataFrame({"id": np.arange(size), "x": (np.arange(size) % 100).astype(float)})
    df.loc[df.index % 10 == 0, "x"] = np.nan
    if rep is None:
        return plain_result(df.dropna().reset_index(drop=True))
    t = TrackedDataFrame(df, representation_cls=rep)
    return t.dropna_rows()


def make_merge(size, rep):
    left = pd.DataFrame({
        "id": np.arange(size),
        "key": np.arange(size) % max(1, size // 10),
        "x": np.arange(size) % 100,
    })
    right = pd.DataFrame({
        "key": np.arange(max(1, size // 10)),
        "y": np.arange(max(1, size // 10)) % 50,
    })
    if rep is None:
        return plain_result(left.merge(right, on="key").reset_index(drop=True))
    t1 = TrackedDataFrame(left, representation_cls=rep)
    t2 = TrackedDataFrame(right, representation_cls=rep)
    return t1.merge(t2, on="key")


def make_append(size, rep):
    df1 = pd.DataFrame({"a": np.arange(size), "b": np.arange(size) * 2})
    df2 = pd.DataFrame({"a": np.arange(size), "b": np.arange(size) * 2})
    if rep is None:
        return plain_result(pd.concat([df1, df2], ignore_index=True))
    t1 = TrackedDataFrame(df1, representation_cls=rep)
    t2 = TrackedDataFrame(df2, representation_cls=rep)
    return t1.append(t2)


def make_select(size, rep):
    df = pd.DataFrame({
        "id": np.arange(size),
        "x": np.arange(size) % 100,
        "y": np.arange(size) % 50,
        "z": np.arange(size) % 25,
    })
    if rep is None:
        return plain_result(df[["id", "x"]].copy())
    t = TrackedDataFrame(df, representation_cls=rep)
    return t.select_columns(["id", "x"])


def make_assign(size, rep):
    df = pd.DataFrame({"id": np.arange(size), "x": np.arange(size) % 100})
    if rep is None:
        out = df.copy()
        out["flag"] = out["x"] > 50
        return plain_result(out)
    t = TrackedDataFrame(df, representation_cls=rep)
    return t.assign_column("flag", t.df["x"] > 50)


def make_map(size, rep):
    df = pd.DataFrame({"id": np.arange(size), "x": np.arange(size) % 100})
    if rep is None:
        out = df.copy()
        out["x"] = out["x"].map(lambda x: x * 2)
        return plain_result(out)
    t = TrackedDataFrame(df, representation_cls=rep)
    return t.map_column("x", lambda x: x * 2)


def make_impute(size, rep):
    df = pd.DataFrame({"id": np.arange(size), "x": (np.arange(size) % 100).astype(float)})
    df.loc[df.index % 10 == 0, "x"] = np.nan
    if rep is None:
        out = df.copy()
        out["x"] = out["x"].fillna(out["x"].mean())
        return plain_result(out)
    t = TrackedDataFrame(df, representation_cls=rep)
    return t.impute_column("x", strategy="mean")


OPS = {
    "filter": make_filter,
    "dropna": make_dropna,
    "merge": make_merge,
    "append": make_append,
    "select_columns": make_select,
    "assign_column": make_assign,
    "map_column": make_map,
    "impute_column": make_impute,
}


def measure_query(tdf):
    if not getattr(tdf, "provenance", []):
        return 0.0

    engine = ProvenanceEngine(tdf.provenance)
    query_set = set(range(min(QUERY_SIZE, len(tdf.df))))

    query_ms, _ = avg_time(lambda: engine.backward(query_set))
    return query_ms


def main():
    print("operator_type,input_size,representation,build_time_ms,memory_mb,query_time_ms")

    for op_name, op_fn in OPS.items():
        for size in SIZES:
            for rep_name, rep_cls in REPRESENTATIONS.items():
                build_ms, tdf = avg_time(lambda: op_fn(size, rep_cls))
                memory_mb = 0.0 if rep_cls is None else provenance_memory_mb(tdf)
                query_ms = 0.0 if rep_cls is None else measure_query(tdf)

                print(
                    f"{op_name},"
                    f"{size},"
                    f"{rep_name},"
                    f"{build_ms:.6f},"
                    f"{memory_mb:.6f},"
                    f"{query_ms:.6f}"
                )


if __name__ == "__main__":
    main()
