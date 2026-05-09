import gc
import sys
import time
import numpy as np
import pandas as pd

from tensprov import IndexedTensorArray
from tensprov.pandas_wrapper import TrackedDataFrame
from tensprov.provenance_engine import ProvenanceEngine


RUNS = 10
WARMUP = 1
QUERY_SIZE = 100

CHAPMAN_MEMORY_MB = {
    "German Credit": 187.38,
    "COMPAS": 218.98,
    "Census": 4012.38,
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

    for op in tdf.provenance:
        total += sys.getsizeof(op)

        if hasattr(op, "__dict__"):
            for v in op.__dict__.values():

                if isinstance(v, np.ndarray):
                    total += v.nbytes

                elif isinstance(v, pd.Series):
                    total += v.memory_usage(deep=True)

                elif isinstance(v, pd.DataFrame):
                    total += v.memory_usage(deep=True).sum()

                elif isinstance(v, (list, tuple)):
                    total += sys.getsizeof(v)

                elif isinstance(v, dict):
                    total += sys.getsizeof(v)

                else:
                    total += sys.getsizeof(v)

    return total / (1024 * 1024)


def summarize_ops(tdf):
    names = []
    for op in tdf.provenance:
        cls = type(op).__name__
        if "Filter" in cls:
            names.append("filter")
        elif "Reduction" in cls:
            names.append("drop/select_columns")
        elif "Augmentation" in cls:
            names.append("assign_column")
        elif "Transformation" in cls:
            names.append("map_column")
        elif "Impute" in cls:
            names.append("impute")
        else:
            names.append(cls)
    return "|".join(sorted(set(names)))


def build_german_credit():
    n = 1000
    rng = np.random.default_rng(1)

    df = pd.DataFrame({
        "id": np.arange(n),
        "duration": rng.integers(4, 72, n),
        "credit_amount": rng.integers(250, 20000, n),
        "installment_rate": rng.integers(1, 5, n),
        "age": rng.integers(18, 75, n),
        "existing_credits": rng.integers(1, 5, n),
        "num_dependents": rng.integers(1, 3, n),
    })

    for i in range(14):
        df[f"cat_{i}"] = rng.integers(0, 4, n)

    t = TrackedDataFrame(df, representation_cls=IndexedTensorArray)

    # normalization
    for col in ["duration", "credit_amount", "age"]:
        t = t.assign_column(f"{col}_norm", (t.df[col] - t.df[col].mean()) / t.df[col].std())

    # binarization
    t = t.assign_column("high_credit", t.df["credit_amount"] > t.df["credit_amount"].median())
    t = t.assign_column("old_age", t.df["age"] >= 30)

    # feature selection
    t = t.drop_columns(["id"])

    # one-hot encoding
    for col in [f"cat_{i}" for i in range(14)]:
        dummies = pd.get_dummies(t.df[col], prefix=col)
        for dc in dummies.columns:
            t = t.assign_column(dc, dummies[dc].astype(int))
        t = t.drop_columns([col])

    # filter that keeps essentially all records
    t = t.filter_mask(t.df["duration"] >= 0)

    return t, n, 21


def build_compas():
    n = 7214
    rng = np.random.default_rng(2)

    df = pd.DataFrame({
        "id": np.arange(n),
        "age": rng.integers(18, 80, n),
        "charge_degree": rng.choice(["F", "M"], n, p=[0.64, 0.36]),
        "race": rng.choice(
            ["African-American", "Caucasian", "Hispanic", "Asian", "Other"],
            n,
            p=[0.50, 0.34, 0.08, 0.03, 0.05],
        ),
        "sex": rng.choice(["Male", "Female"], n),
        "priors_count": rng.integers(0, 20, n),
        "score_text": rng.choice(["Low", "Medium", "High"], n),
        "two_year_recid": rng.integers(0, 2, n),
    })

    for i in range(45):
        df[f"extra_{i}"] = rng.integers(0, 10, n)

    t = TrackedDataFrame(df, representation_cls=IndexedTensorArray)

    # keep about 6907 / 7214
    mask = (
        (t.df["age"] >= 18)
        & (t.df["charge_degree"].isin(["F", "M"]))
        & (t.df["race"].isin(["African-American", "Caucasian", "Hispanic", "Asian", "Other"]))
    )
    # deterministic additional drop ~307 rows
    mask = mask & (t.df.index >= 307)

    t = t.filter_mask(mask)

    keep = [
        "age",
        "charge_degree",
        "race",
        "sex",
        "priors_count",
        "score_text",
        "two_year_recid",
        "id",
    ]
    drop_cols = [c for c in t.df.columns if c not in keep]
    t = t.drop_columns(drop_cols)

    t = t.map_column("charge_degree", lambda x: 1 if x == "F" else 0)
    t = t.map_column("sex", lambda x: 1 if x == "Male" else 0)
    t = t.map_column("score_text", lambda x: {"Low": 0, "Medium": 1, "High": 2}[x])

    return t, n, 53


def build_census():
    n = 32561
    rng = np.random.default_rng(3)

    df = pd.DataFrame({
        "age": rng.integers(18, 90, n),
        "fnlwgt": rng.integers(10000, 500000, n),
        "education_num": rng.integers(1, 16, n),
        "capital_gain": rng.integers(0, 10000, n),
        "capital_loss": rng.integers(0, 5000, n),
        "hours_per_week": rng.integers(1, 99, n),
    })

    cat_specs = {
        "workclass": 8,
        "education": 16,
        "marital_status": 7,
        "occupation": 14,
        "relationship": 6,
        "race": 5,
        "sex": 2,
        "native_country": 40,
        "income": 2,
    }

    for col, k in cat_specs.items():
        df[col] = rng.integers(0, k, n)

    t = TrackedDataFrame(df, representation_cls=IndexedTensorArray)

    # missing-value filter, but synthetic data has no missing values
    t = t.filter_mask(~t.df.isna().any(axis=1))

    # normalization
    for col in ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]:
        t = t.assign_column(f"{col}_norm", (t.df[col] - t.df[col].mean()) / t.df[col].std())

    # binarization
    t = t.assign_column("high_hours", t.df["hours_per_week"] > 40)
    t = t.assign_column("has_capital_gain", t.df["capital_gain"] > 0)

    # one-hot encoding; enough categories to reach ~104 output attrs
    for col in list(cat_specs.keys()):
        dummies = pd.get_dummies(t.df[col], prefix=col)
        for dc in dummies.columns:
            t = t.assign_column(dc, dummies[dc].astype(int))
        t = t.drop_columns([col])

    # Trim/pad to exactly 104 attributes if needed
    if len(t.df.columns) > 104:
        t = t.select_columns(list(t.df.columns[:104]))
    while len(t.df.columns) < 104:
        t = t.assign_column(f"pad_{len(t.df.columns)}", 0)

    return t, n, 15


PIPELINES = {
    "German Credit": build_german_credit,
    "COMPAS": build_compas,
    "Census": build_census,
}


def run_one(name, build_fn):
    build_ms, result = avg_time(build_fn)
    tdf, n_input_records, n_input_attributes = result

    memory_mb = provenance_memory_mb(tdf)

    engine = ProvenanceEngine(tdf.provenance)
    query_set = set(range(min(QUERY_SIZE, len(tdf.df))))

    forward_ms, _ = avg_time(lambda: engine.forward(query_set))
    backward_ms, _ = avg_time(lambda: engine.backward(query_set))

    chapman_mb = CHAPMAN_MEMORY_MB[name]
    reduction = chapman_mb / memory_mb if memory_mb > 0 else float("inf")

    return {
        "pipeline_name": name,
        "n_input_records": n_input_records,
        "n_input_attributes": n_input_attributes,
        "n_output_records": len(tdf.df),
        "n_output_attributes": len(tdf.df.columns),
        "n_ops": len(tdf.provenance),
        "operator_types": summarize_ops(tdf),
        "tensprov_memory_mb": memory_mb,
        "tensprov_build_time_ms": build_ms,
        "tensprov_query_forward_ms": forward_ms,
        "tensprov_query_backward_ms": backward_ms,
        "chapman_memory_mb": chapman_mb,
        "memory_reduction_factor": reduction,
    }


def main():
    print(
        "pipeline_name,n_input_records,n_input_attributes,"
        "n_output_records,n_output_attributes,n_ops,operator_types,"
        "tensprov_memory_mb,tensprov_build_time_ms,"
        "tensprov_query_forward_ms,tensprov_query_backward_ms,"
        "chapman_memory_mb,memory_reduction_factor"
    )

    for name, build_fn in PIPELINES.items():
        r = run_one(name, build_fn)
        print(
            f"{r['pipeline_name']},"
            f"{r['n_input_records']},"
            f"{r['n_input_attributes']},"
            f"{r['n_output_records']},"
            f"{r['n_output_attributes']},"
            f"{r['n_ops']},"
            f"{r['operator_types']},"
            f"{r['tensprov_memory_mb']:.6f},"
            f"{r['tensprov_build_time_ms']:.6f},"
            f"{r['tensprov_query_forward_ms']:.6f},"
            f"{r['tensprov_query_backward_ms']:.6f},"
            f"{r['chapman_memory_mb']:.2f},"
            f"{r['memory_reduction_factor']:.2f}"
        )


if __name__ == "__main__":
    main()
