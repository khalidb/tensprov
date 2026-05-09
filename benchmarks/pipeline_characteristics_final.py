import time
import gc
import pandas as pd

from benchmarks import run_realistic_suite as suite
from tensprov import IndexedTensorArray


class PlainTrackedDataFrame:
    def __init__(self, df, representation_cls=None, provenance=None):
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
        return PlainTrackedDataFrame(self.df.drop(columns=list(columns)))

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
        other_df = other.df if hasattr(other, "df") else other
        return PlainTrackedDataFrame(
            self.df.merge(other_df, on=on).reset_index(drop=True)
        )

    def append(self, other):
        other_df = other.df if hasattr(other, "df") else other
        return PlainTrackedDataFrame(
            pd.concat([self.df, other_df], ignore_index=True)
        )


PIPELINES = {
    "census": ("demographics", suite.build_census_pipeline, [10000, 5]),
    "retail": ("retail", suite.build_retail_pipeline, [5000, 20000, 1000]),
    "healthcare": ("healthcare", suite.build_healthcare_pipeline, [5000, 15000, 3000]),
    "fraud": ("finance", suite.build_fraud_pipeline, [20000, 5000, 2000]),
    "education": ("education", suite.build_education_pipeline, [4000, 12000, 1000]),
    "insurance": ("insurance", suite.build_insurance_pipeline, [5000, 15000, 5000]),
    "iot": ("iot", suite.build_iot_pipeline, [5000, 30000, 5000]),
    "deep_hr": ("hr", suite.build_deep_hr_pipeline, [8000, 50, 8000]),
    "deep_logistics": ("logistics", suite.build_deep_logistics_pipeline, [30000, 5000, 5000]),
    "social": ("social", suite.build_social_pipeline, [5000, 15000, 15000]),
}


def time_once(fn):
    gc.collect()
    t0 = time.perf_counter()
    result = fn()
    return (time.perf_counter() - t0) * 1000, result


def plain_run(build_fn):
    original = suite.TrackedDataFrame
    suite.TrackedDataFrame = PlainTrackedDataFrame
    try:
        return time_once(lambda: build_fn(None))
    finally:
        suite.TrackedDataFrame = original


def prov_run(build_fn):
    return time_once(lambda: build_fn(IndexedTensorArray))


def normalize_op(op):
    name = type(op).__name__
    mapping = {
        "FilterOperation": "filter",
        "AdaptiveFilterOperation": "filter",
        "JoinOperation": "join",
        "AppendOperation": "append",
        "VerticalReductionOperation": "select/drop_columns",
        "VerticalAugmentationOperation": "assign",
        "DataTransformationOperation": "map",
        "ImputeOperation": "impute",
    }
    return mapping.get(name, name)


def main():
    print(
        "pipeline_name,domain,n_ops,pipeline_depth,operator_types,"
        "has_join,has_impute,input_sizes,output_rows,output_cols,"
        "build_time_ms,pipeline_exec_time_ms,provenance_overhead_ratio"
    )

    for name, (domain, build_fn, input_sizes) in PIPELINES.items():
        build_ms, tdf = prov_run(build_fn)
        plain_ms, plain_tdf = plain_run(build_fn)

        operator_types = sorted({normalize_op(op) for op in tdf.provenance})
        has_join = "join" in operator_types
        has_impute = "impute" in operator_types

        ratio = build_ms / plain_ms if plain_ms > 0 else float("nan")

        print(
            f"{name},"
            f"{domain},"
            f"{len(tdf.provenance)},"
            f"{len(tdf.provenance)},"
            f"{'|'.join(operator_types)},"
            f"{has_join},"
            f"{has_impute},"
            f"\"{input_sizes}\","
            f"{len(tdf.df)},"
            f"{len(tdf.df.columns)},"
            f"{build_ms:.3f},"
            f"{plain_ms:.3f},"
            f"{ratio:.3f}"
        )


if __name__ == "__main__":
    main()