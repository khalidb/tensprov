import time
import tracemalloc
import gc

from tensprov.pandas_wrapper import TrackedDataFrame
from tensprov.provenance_engine import ProvenanceEngine
from tensprov import COOTensor, IndexedTensor, IndexedTensorArray, ProvenanceGraph
from tensprov.relational_baseline import RelationalProvenance


import sys
import inspect
import tensprov.indexed_tensor_array as indexed_tensor_array_module
import tensprov.pandas_wrapper as pandas_wrapper_module
import tensprov.adaptive_filter_index as adaptive_filter_index_module

print("Using IndexedTensorArray:", indexed_tensor_array_module.__file__, file=sys.stderr)
print("Using pandas_wrapper:", pandas_wrapper_module.__file__, file=sys.stderr)
print("Using adaptive_filter_index:", adaptive_filter_index_module.__file__, file=sys.stderr)
print("IndexedTensorArray class file:", inspect.getfile(IndexedTensorArray), file=sys.stderr)
print("TrackedDataFrame class file:", inspect.getfile(TrackedDataFrame), file=sys.stderr)

RUNS = 10
WARMUP = 1
QUERY_SIZES = [1, 10, 100, 1000]


def avg_time(fn, runs=RUNS, warmup=WARMUP):
    times = []
    last_result = None

    for i in range(warmup + runs):
        gc.collect()
        t0 = time.perf_counter()
        last_result = fn()
        elapsed = (time.perf_counter() - t0) * 1000

        if i >= warmup:
            times.append(elapsed)

    return sum(times) / len(times), last_result


# ------------------------
# PIPELINES
# ------------------------

def build_census_pipeline(representation_cls):
    import pandas as pd

    df_people = pd.DataFrame({
        "id": list(range(10000)),
        "age": [20 + (i % 40) for i in range(10000)],
        "education": [i % 5 for i in range(10000)],
        "income": [30000 + (i % 70000) for i in range(10000)],
        "unused_col": [i % 3 for i in range(10000)],
    })

    df_edu = pd.DataFrame({
        "education": list(range(5)),
        "level": ["low", "mid", "mid", "high", "high"],
    })

    t1 = TrackedDataFrame(df_people, representation_cls=representation_cls)
    t2 = TrackedDataFrame(df_edu, representation_cls=representation_cls)

    t = t1.filter_mask(t1.df["age"] >= 25)
    t = t.drop_columns(["unused_col"])
    t = t.merge(t2, on="education")
    t = t.filter_mask(t.df["level"] == "high")

    return t


def build_retail_pipeline(representation_cls):
    import pandas as pd

    df_customers = pd.DataFrame({
        "cust_id": list(range(5000)),
        "region": [i % 5 for i in range(5000)],
    })

    df_orders = pd.DataFrame({
        "order_id": list(range(20000)),
        "cust_id": [i % 5000 for i in range(20000)],
        "product_id": [i % 1000 for i in range(20000)],
        "amount": [10 + (i % 100) for i in range(20000)],
    })

    df_products = pd.DataFrame({
        "product_id": list(range(1000)),
        "category": [i % 10 for i in range(1000)],
    })

    t1 = TrackedDataFrame(df_customers, representation_cls=representation_cls)
    t2 = TrackedDataFrame(df_orders, representation_cls=representation_cls)
    t3 = TrackedDataFrame(df_products, representation_cls=representation_cls)

    t = t2.dropna_rows()
    t = t.assign_column("high_value", t.df["amount"] > 50)
    t = t.merge(t1, on="cust_id")
    t = t.merge(t3, on="product_id")
    t = t.map_column("category", lambda x: "A" if x < 5 else "B")
    t = t.filter_mask(t.df["category"] == "A")
    t = t.select_columns(["cust_id", "category", "amount"])

    return t


def build_healthcare_pipeline(representation_cls):
    import pandas as pd
    import numpy as np

    df_patients = pd.DataFrame({
        "patient_id": list(range(5000)),
        "age": [20 + (i % 70) for i in range(5000)],
    })

    df_visits = pd.DataFrame({
        "visit_id": list(range(15000)),
        "patient_id": [i % 5000 for i in range(15000)],
        "lab_id": [i % 3000 for i in range(15000)],
    })

    df_labs = pd.DataFrame({
        "lab_id": list(range(3000)),
        "value": [float(i % 100) for i in range(3000)],
    })

    df_labs.loc[df_labs.index % 10 == 0, "value"] = np.nan

    t1 = TrackedDataFrame(df_patients, representation_cls=representation_cls)
    t2 = TrackedDataFrame(df_visits, representation_cls=representation_cls)
    t3 = TrackedDataFrame(df_labs, representation_cls=representation_cls)

    t = t3.dropna_rows()
    t = t.impute_column("value", strategy="mean")
    t = t.assign_column("high_value", t.df["value"] > 50)
    t = t.merge(t2, on="lab_id")
    t = t.merge(t1, on="patient_id")
    t = t.map_column("high_value", lambda x: int(x))
    t = t.filter_mask(t.df["high_value"] == 1)
    t = t.select_columns(["patient_id", "value", "high_value"])

    return t


# (ALL OTHER PIPELINES FOLLOW SAME PATTERN — already correct in your file)
# I fixed ALL filter_mask usages everywhere.




def build_fraud_pipeline(representation_cls):
    import pandas as pd

    df_tx = pd.DataFrame({
        "tx_id": list(range(20000)),
        "account_id": [i % 5000 for i in range(20000)],
        "merchant_id": [i % 2000 for i in range(20000)],
        "amount": [10 + (i % 500) for i in range(20000)],
    })
    df_accounts = pd.DataFrame({
        "account_id": list(range(5000)),
        "region": [i % 5 for i in range(5000)],
    })
    df_merchants = pd.DataFrame({
        "merchant_id": list(range(2000)),
        "category": [i % 10 for i in range(2000)],
    })

    t1 = TrackedDataFrame(df_tx, representation_cls=representation_cls)
    t2 = TrackedDataFrame(df_accounts, representation_cls=representation_cls)
    t3 = TrackedDataFrame(df_merchants, representation_cls=representation_cls)

    t = t1.dropna_rows()
    t = t.assign_column("high_amount", t.df["amount"] > 300)
    t = t.merge(t2, on="account_id")
    t = t.merge(t3, on="merchant_id")
    t = t.map_column("category", lambda x: "risky" if x < 3 else "normal")
    t = t.filter_mask((t.df["high_amount"]) & (t.df["category"] == "risky"))
    t = t.select_columns(["account_id", "merchant_id", "amount"])
    t = t.drop_columns(["merchant_id"])
    return t


def build_education_pipeline(representation_cls):
    import pandas as pd
    import numpy as np

    df_students = pd.DataFrame({
        "student_id": list(range(4000)),
        "age": [18 + (i % 10) for i in range(4000)],
    })
    df_enrollments = pd.DataFrame({
        "enroll_id": list(range(12000)),
        "student_id": [i % 4000 for i in range(12000)],
        "course_id": [i % 1000 for i in range(12000)],
    })
    df_grades = pd.DataFrame({
        "course_id": list(range(1000)),
        "grade": [float(i % 100) for i in range(1000)],
    })
    df_grades.loc[df_grades.index % 7 == 0, "grade"] = np.nan

    t1 = TrackedDataFrame(df_students, representation_cls=representation_cls)
    t2 = TrackedDataFrame(df_enrollments, representation_cls=representation_cls)
    t3 = TrackedDataFrame(df_grades, representation_cls=representation_cls)

    t = t3.dropna_rows()
    t = t.impute_column("grade", strategy="mean")
    t = t.assign_column("passed", t.df["grade"] >= 50)
    t = t.merge(t2, on="course_id")
    t = t.merge(t1, on="student_id")
    t = t.map_column("passed", lambda x: int(x))
    t = t.filter_mask(t.df["passed"] == 0)
    t = t.select_columns(["student_id", "grade", "passed"])
    t = t.drop_columns(["passed"])
    return t


def build_insurance_pipeline(representation_cls):
    import pandas as pd
    import numpy as np

    df_policies = pd.DataFrame({
        "policy_id": list(range(5000)),
        "customer_id": [i % 3000 for i in range(5000)],
    })
    df_claims = pd.DataFrame({
        "claim_id": list(range(15000)),
        "policy_id": [i % 5000 for i in range(15000)],
        "amount": [100 + (i % 1000) for i in range(15000)],
    })
    df_risk = pd.DataFrame({
        "policy_id": list(range(5000)),
        "risk_score": [float(i % 100) for i in range(5000)],
    })
    df_risk.loc[df_risk.index % 8 == 0, "risk_score"] = np.nan

    t1 = TrackedDataFrame(df_policies, representation_cls=representation_cls)
    t2 = TrackedDataFrame(df_claims, representation_cls=representation_cls)
    t3 = TrackedDataFrame(df_risk, representation_cls=representation_cls)

    t = t3.dropna_rows()
    t = t.impute_column("risk_score", strategy="mean")
    t = t.assign_column("high_risk", t.df["risk_score"] > 50)
    t = t.merge(t2, on="policy_id")
    t = t.merge(t1, on="policy_id")
    t = t.map_column("high_risk", lambda x: int(x))
    t = t.filter_mask(t.df["high_risk"] == 1)
    t = t.select_columns(["policy_id", "amount", "high_risk"])
    t = t.drop_columns(["high_risk"])
    t = t.filter_mask(t.df["amount"] > 500)
    return t


def build_iot_pipeline(representation_cls):
    import pandas as pd
    import numpy as np

    df_devices = pd.DataFrame({
        "device_id": list(range(5000)),
        "type": [i % 5 for i in range(5000)],
    })
    df_readings = pd.DataFrame({
        "reading_id": list(range(30000)),
        "device_id": [i % 5000 for i in range(30000)],
        "value": [float(i % 100) for i in range(30000)],
    })
    df_locations = pd.DataFrame({
        "device_id": list(range(5000)),
        "zone": [i % 10 for i in range(5000)],
    })
    df_readings.loc[df_readings.index % 11 == 0, "value"] = np.nan

    t1 = TrackedDataFrame(df_devices, representation_cls=representation_cls)
    t2 = TrackedDataFrame(df_readings, representation_cls=representation_cls)
    t3 = TrackedDataFrame(df_locations, representation_cls=representation_cls)

    t = t2.impute_column("value", strategy="mean")
    t = t.assign_column("high", t.df["value"] > 70)
    t = t.map_column("high", lambda x: int(x))
    t = t.merge(t1, on="device_id")
    t = t.merge(t3, on="device_id")
    t = t.filter_mask(t.df["high"] == 1)
    t = t.select_columns(["device_id", "value", "zone", "high"])
    t = t.drop_columns(["high"])
    t = t.filter_mask(t.df["zone"] < 5)
    t = t.assign_column("scaled", t.df["value"] / 100.0)
    t = t.map_column("scaled", lambda x: round(x, 2))
    t = t.filter_mask(t.df["scaled"] > 0.5)
    return t


def build_deep_hr_pipeline(representation_cls):
    import pandas as pd
    import numpy as np

    df_employees = pd.DataFrame({
        "emp_id": list(range(8000)),
        "dept_id": [i % 50 for i in range(8000)],
        "salary": [30000 + (i % 70000) for i in range(8000)],
    })
    df_departments = pd.DataFrame({
        "dept_id": list(range(50)),
        "region": [i % 5 for i in range(50)],
    })
    df_performance = pd.DataFrame({
        "emp_id": list(range(8000)),
        "score": [float(i % 100) for i in range(8000)],
    })
    df_performance.loc[df_performance.index % 13 == 0, "score"] = np.nan

    t1 = TrackedDataFrame(df_employees, representation_cls=representation_cls)
    t2 = TrackedDataFrame(df_departments, representation_cls=representation_cls)
    t3 = TrackedDataFrame(df_performance, representation_cls=representation_cls)

    t = t3.impute_column("score", strategy="mean")
    t = t.assign_column("high_perf", t.df["score"] > 60)
    t = t.map_column("high_perf", lambda x: int(x))
    t = t.merge(t1, on="emp_id")
    t = t.merge(t2, on="dept_id")
    t = t.filter_mask(t.df["high_perf"] == 1)
    t = t.select_columns(["emp_id", "dept_id", "salary", "region"])
    t = t.assign_column("salary_k", t.df["salary"] / 1000.0)
    t = t.map_column("salary_k", lambda x: round(x, 1))
    t = t.filter_mask(t.df["salary_k"] > 31)
    t = t.drop_columns(["salary_k"])
    t = t.assign_column("flag", t.df["region"] < 3)
    t = t.map_column("flag", lambda x: int(x))
    t = t.filter_mask(t.df["flag"] == 1)
    return t


def build_deep_logistics_pipeline(representation_cls):
    import pandas as pd
    import numpy as np

    df_packages = pd.DataFrame({
        "pkg_id": list(range(30000)),
        "route_id": [i % 5000 for i in range(30000)],
        "weight": [1 + (i % 50) for i in range(30000)],
    })
    df_routes = pd.DataFrame({
        "route_id": list(range(5000)),
        "zone": [i % 10 for i in range(5000)],
    })
    df_hubs = pd.DataFrame({
        "route_id": list(range(5000)),
        "hub_score": [float(i % 100) for i in range(5000)],
    })
    df_hubs.loc[df_hubs.index % 9 == 0, "hub_score"] = np.nan

    t1 = TrackedDataFrame(df_packages, representation_cls=representation_cls)
    t2 = TrackedDataFrame(df_routes, representation_cls=representation_cls)
    t3 = TrackedDataFrame(df_hubs, representation_cls=representation_cls)

    t = t3.impute_column("hub_score", strategy="mean")
    t = t.assign_column("good_hub", t.df["hub_score"] > 50)
    t = t.map_column("good_hub", lambda x: int(x))
    t = t.merge(t1, on="route_id")
    t = t.merge(t2, on="route_id")
    t = t.filter_mask(t.df["good_hub"] == 1)
    t = t.assign_column("heavy", t.df["weight"] > 20)
    t = t.map_column("heavy", lambda x: int(x))
    t = t.filter_mask(t.df["heavy"] == 1)
    t = t.select_columns(["pkg_id", "route_id", "weight", "zone"])
    t = t.assign_column("scaled_w", t.df["weight"] / 50.0)
    t = t.map_column("scaled_w", lambda x: round(x, 2))
    t = t.filter_mask(t.df["scaled_w"] > 0.4)
    t = t.drop_columns(["scaled_w"])
    t = t.assign_column("zone_flag", t.df["zone"] < 5)
    t = t.map_column("zone_flag", lambda x: int(x))
    t = t.filter_mask(t.df["zone_flag"] == 1)
    t = t.filter_mask(t.df["weight"] > 25)
    return t


def build_social_pipeline(representation_cls):
    import pandas as pd

    df_users = pd.DataFrame({
        "user_id": list(range(5000)),
        "age": [18 + (i % 50) for i in range(5000)],
    })
    df_posts = pd.DataFrame({
        "post_id": list(range(15000)),
        "user_id": [i % 5000 for i in range(15000)],
        "likes": [i % 200 for i in range(15000)],
    })
    df_interactions = pd.DataFrame({
        "post_id": list(range(15000)),
        "shares": [i % 50 for i in range(15000)],
    })

    t1 = TrackedDataFrame(df_users, representation_cls=representation_cls)
    t2 = TrackedDataFrame(df_posts, representation_cls=representation_cls)
    t3 = TrackedDataFrame(df_interactions, representation_cls=representation_cls)

    t = t2.assign_column("popular", t2.df["likes"] > 100)
    t = t.map_column("popular", lambda x: int(x))
    t = t.merge(t1, on="user_id")
    t = t.merge(t3, on="post_id")
    t = t.assign_column("score", t.df["likes"] + t.df["shares"])
    t = t.map_column("score", lambda x: int(x))
    t = t.select_columns(["user_id", "likes", "shares", "score"])
    t = t.drop_columns(["shares"])
    t = t.filter_mask(t.df["score"] > 120)
    return t

# ------------------------
# RUNNER
# ------------------------

def run_pipeline(name, build_fn, deletion_start, representation_cls):
    # Timing: DO NOT include tracemalloc.
    build_times = []
    last_tdf = None

    for i in range(WARMUP + RUNS):
        gc.collect()
        t0 = time.perf_counter()
        tdf = build_fn(representation_cls)
        elapsed = (time.perf_counter() - t0) * 1000

        if i >= WARMUP:
            build_times.append(elapsed)

        last_tdf = tdf

    build_ms = sum(build_times) / len(build_times)
    tdf = last_tdf

    # Memory: measure separately, not included in build_ms.
    gc.collect()
    tracemalloc.start()
    mem_tdf = build_fn(representation_cls)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = peak / (1024 * 1024)

    engine = ProvenanceEngine(tdf.provenance)

    results = []

    for k in QUERY_SIZES:
        forward_ms, _ = avg_time(
            lambda: query_deletion(
                engine,
                start_id=deletion_start,
                query_size=k,
            )
        )

        backward_ms, _ = avg_time(
            lambda: query_debugging(
                engine,
                query_size=k,
            )
        )

        q3_ms, _ = avg_time(
            lambda: query_q3_attribute_forward(
                engine,
                tdf,
                start_id=deletion_start,
                query_size=k,
            )
        )

        q9_ms, _ = avg_time(
            lambda: query_q9_all_transformations(tdf)
        )

        q10_ms, _ = avg_time(
            lambda: query_q10_co_contributory(
                engine,
                tdf,
                start_id=deletion_start,
                query_size=k,
            )
        )

        results.append({
            "pipeline": name,
            "representation": representation_cls.__name__,
            "query_size": k,
            "n_ops": len(tdf.provenance),
            "final_rows": len(tdf.df),
            "build_ms": build_ms,
            "memory_mb": memory_mb,
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "q3_ms": q3_ms,
            "q9_ms": q9_ms,
            "q10_ms": q10_ms,
        })

    return results


def query_deletion(engine, start_id=0, query_size=1):
    return engine.forward(set(range(start_id, start_id + query_size)))


def query_debugging(engine, query_size=1):
    return engine.backward(set(range(query_size)))


def query_q3_attribute_forward(engine, tdf, start_id=0, query_size=1, attr=None):
    records = set(range(start_id, start_id + query_size))
    out_records = engine.forward(records)

    if attr is None:
        attr = tdf.df.columns[0] if len(tdf.df.columns) > 0 else "__attr__"

    return {
        (rid, attr)
        for rid in out_records
        if isinstance(rid, int) and 0 <= rid < len(tdf.df)
    }


def query_q9_all_transformations(tdf):
    return [
        {
            "position": i,
            "operation": getattr(op, "name", op.__class__.__name__),
            "contextual": getattr(op, "contextual", False),
            "metadata": getattr(op, "metadata", {}),
        }
        for i, op in enumerate(tdf.provenance)
    ]


def query_q10_co_contributory(engine, tdf, start_id=0, query_size=1):
    seed_records = set(range(start_id, start_id + query_size))
    affected_outputs = engine.forward(seed_records)
    return engine.backward(affected_outputs)


def main():
    pipelines = {
        "census": (build_census_pipeline, 81),
        "retail": (build_retail_pipeline, 81),
        "healthcare": (build_healthcare_pipeline, 81),
        "fraud": (build_fraud_pipeline, 301),
        "education": (build_education_pipeline, 1),
        "insurance": (build_insurance_pipeline, 451),
        "iot": (build_iot_pipeline, 71),
        "deep_hr": (build_deep_hr_pipeline, 1061),
        "deep_logistics": (build_deep_logistics_pipeline, 84),
        "social": (build_social_pipeline, 150),
    }

    representations = {
        "IndexedTensorArray": IndexedTensorArray,
        "IndexedTensor": IndexedTensor,
        "COOTensor": COOTensor,
        "ProvenanceGraph": ProvenanceGraph,
        "RelationalProvenance": RelationalProvenance,
    }

    print(
        "pipeline,representation,query_size,n_ops,final_rows,"
        "build_ms,memory_mb,forward_ms,backward_ms,q3_ms,q9_ms,q10_ms"
    )

    for rep_name, rep_cls in representations.items():
        for pipeline_name, (build_fn, deletion_start) in pipelines.items():
            results = run_pipeline(
                pipeline_name,
                build_fn,
                deletion_start,
                rep_cls,
            )

            for r in results:
                print(
                    f"{r['pipeline']},"
                    f"{rep_name},"
                    f"{r['query_size']},"
                    f"{r['n_ops']},"
                    f"{r['final_rows']},"
                    f"{r['build_ms']:.6f},"
                    f"{r['memory_mb']:.6f},"
                    f"{r['forward_ms']:.6f},"
                    f"{r['backward_ms']:.6f},"
                    f"{r['q3_ms']:.6f},"
                    f"{r['q9_ms']:.6f},"
                    f"{r['q10_ms']:.6f}"
                )


if __name__ == "__main__":
    main()