import gc
import time
import sys
import pandas as pd

from tensprov import IndexedTensorArray
from tensprov.pandas_wrapper import TrackedDataFrame

import benchmarks.run_retail_scaling as retail_mod
import benchmarks.run_recomputation_scaling as recomp_mod

#SCALES = [1, 5, 10, 25]
#RUNS = 3
SCALES = [1, 5, 10, 25, 50, 100]
RUNS = 10
WARMUP = 1


class PlainTrackedDataFrame:
    def __init__(self, df, provenance=None, representation_cls=None):
        self.df = df.reset_index(drop=True)
        self.provenance = []

    def filter(self, predicate):
        mask = self.df.apply(predicate, axis=1)
        return PlainTrackedDataFrame(self.df[mask].reset_index(drop=True))

    def filter_mask(self, mask):
        mask = pd.Series(mask, index=self.df.index).to_numpy(dtype=bool)
        return PlainTrackedDataFrame(self.df[mask].reset_index(drop=True))

    def dropna_rows(self, subset=None):
        if subset is None:
            mask = ~self.df.isna().any(axis=1)
        else:
            mask = ~self.df[subset].isna().any(axis=1)
        return PlainTrackedDataFrame(self.df[mask].reset_index(drop=True))

    def select_columns(self, columns):
        return PlainTrackedDataFrame(self.df[list(columns)].copy())

    def drop_columns(self, columns):
        return PlainTrackedDataFrame(self.df.drop(columns=list(columns)).copy())

    def assign_column(self, name, values):
        df = self.df.copy()
        df[name] = values
        return PlainTrackedDataFrame(df)

    def map_column(self, column, func):
        df = self.df.copy()
        df[column] = df[column].map(func)
        return PlainTrackedDataFrame(df)

    def impute_column(self, column, strategy="mean"):
        df = self.df.copy()
        if strategy == "mean":
            value = df[column].mean()
        elif strategy == "median":
            value = df[column].median()
        else:
            value = df[column].mode().iloc[0]
        df[column] = df[column].fillna(value)
        return PlainTrackedDataFrame(df)

    def merge(self, other, on):
        return PlainTrackedDataFrame(
            self.df.merge(other.df, on=on).reset_index(drop=True)
        )

    def append(self, other):
        return PlainTrackedDataFrame(
            pd.concat([self.df, other.df], ignore_index=True)
        )


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
    for op in tdf.provenance:
        if hasattr(op, "approx_memory_bytes"):
            total += op.approx_memory_bytes()
        else:
            total += sys.getsizeof(op)
    return total / (1024 * 1024)


def run_plain(module, build_fn, scale):
    original = module.TrackedDataFrame
    module.TrackedDataFrame = PlainTrackedDataFrame
    try:
        return build_fn(None, scale)
    finally:
        module.TrackedDataFrame = original


def run_prov(build_fn, scale):
    return build_fn(IndexedTensorArray, scale)

def run_plain_pandas(module, build_fn, scale):
    return build_fn(None, scale)

def main():
    pipelines = {
        "retail": (
            retail_mod,
            retail_mod.build_scaled_retail_pipeline,
        ),
        "iot": (
            recomp_mod,
            recomp_mod.build_scaled_iot_pipeline,
        ),
        "deep_logistics": (
            recomp_mod,
            recomp_mod.build_scaled_deep_logistics_pipeline,
        ),
    }

    print(
        "pipeline,scale,output_rows,"
        "plain_wrapped_ms,provenance_build_ms,"
        "pure_provenance_overhead_ms,overhead_ratio,memory_mb,"
        "join_output_rows_before_filter,join_tensor_construction_ms"
    )

    for name, (module, build_fn) in pipelines.items():
        for scale in SCALES:
            plain_ms, plain_tdf = avg_time(
                lambda: run_plain_pandas(module, build_fn, scale)
            )

            prov_ms, tdf = avg_time(
                lambda: run_prov(build_fn, scale)
            )

            output_rows = len(tdf.df)
            memory_mb = provenance_memory_mb(tdf)

            pure_overhead = prov_ms - plain_ms
            overhead_ratio = prov_ms / plain_ms if plain_ms > 0 else 0.0

            # Leave blank unless IoT builder is explicitly instrumented.
            join_rows = ""
            join_tensor_ms = ""

            print(
                f"{name},{scale},{output_rows},"
                f"{plain_ms:.3f},{prov_ms:.3f},"
                f"{pure_overhead:.3f},{overhead_ratio:.3f},{memory_mb:.6f},"
                f"{join_rows},{join_tensor_ms}"
            )


if __name__ == "__main__":
    main()
