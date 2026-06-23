"""
Pipeline: healthcare (Healthcare)
Domain:   Healthcare
Depth:    8 operations
Operators: dropna, impute, assign, join, join, map, filter, vertical reduction

Corresponds to Table 9 (pipeline 'healthcare') in the TensProv paper.
Input datasets: patients (5,000), visits (15,000), lab results (3,000).
"""

import pandas as pd
import numpy as np
from tensprov.pandas_wrapper import TrackedDataFrame


def build_pipeline(representation_cls):
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

    t = t3.dropna_rows()                                         # horizontal reduction
    t = t.impute_column("value", strategy="mean")                # data transformation (contextual)
    t = t.assign_column("high_value", t.df["value"] > 50)       # vertical augmentation
    t = t.merge(t2, on="lab_id")                                 # join
    t = t.merge(t1, on="patient_id")                             # join
    t = t.map_column("high_value", lambda x: int(x))            # data transformation
    t = t.filter_mask(t.df["high_value"] == 1)                  # horizontal reduction
    t = t.select_columns(["patient_id", "value", "high_value"]) # vertical reduction

    return t


if __name__ == "__main__":
    from tensprov import IndexedTensorArray
    result = build_pipeline(IndexedTensorArray)
    print(f"Output records: {len(result.df)}")
    print(result.df.head())
