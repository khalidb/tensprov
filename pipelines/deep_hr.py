"""
Pipeline: deep_hr (HR)
Domain:   Human Resources
Depth:    14 operations
Operators: impute, assign, map, join, join, filter, vertical reduction,
           assign, map, filter, vertical reduction, assign, map, filter

Corresponds to Table 9 (pipeline 'deep_hr') in the TensProv paper.
Input datasets: employees (8,000), departments (50), performance scores (8,000).
"""

import pandas as pd
import numpy as np
from tensprov.pandas_wrapper import TrackedDataFrame


def build_pipeline(representation_cls):
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

    t = t3.impute_column("score", strategy="mean")                   # data transformation (contextual)
    t = t.assign_column("high_perf", t.df["score"] > 60)            # vertical augmentation
    t = t.map_column("high_perf", lambda x: int(x))                 # data transformation
    t = t.merge(t1, on="emp_id")                                      # join
    t = t.merge(t2, on="dept_id")                                     # join
    t = t.filter_mask(t.df["high_perf"] == 1)                       # horizontal reduction
    t = t.select_columns(["emp_id", "dept_id", "salary", "region"]) # vertical reduction
    t = t.assign_column("salary_k", t.df["salary"] / 1000.0)        # vertical augmentation
    t = t.map_column("salary_k", lambda x: round(x, 1))             # data transformation
    t = t.filter_mask(t.df["salary_k"] > 31)                        # horizontal reduction
    t = t.drop_columns(["salary_k"])                                  # vertical reduction
    t = t.assign_column("flag", t.df["region"] < 3)                 # vertical augmentation
    t = t.map_column("flag", lambda x: int(x))                      # data transformation
    t = t.filter_mask(t.df["flag"] == 1)                             # horizontal reduction

    return t


if __name__ == "__main__":
    from tensprov import IndexedTensorArray
    result = build_pipeline(IndexedTensorArray)
    print(f"Output records: {len(result.df)}")
    print(result.df.head())
