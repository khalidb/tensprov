"""
Pipeline: insurance (Insurance)
Domain:   Insurance
Depth:    10 operations
Operators: dropna, impute, assign, join, join, map, filter, vertical reduction,
           vertical reduction, filter

Corresponds to Table 9 (pipeline 'insurance') in the TensProv paper.
Input datasets: policies (5,000), claims (15,000), risk scores (5,000).
"""

import pandas as pd
import numpy as np
from tensprov.pandas_wrapper import TrackedDataFrame


def build_pipeline(representation_cls):
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

    t = t3.dropna_rows()                                          # horizontal reduction
    t = t.impute_column("risk_score", strategy="mean")           # data transformation (contextual)
    t = t.assign_column("high_risk", t.df["risk_score"] > 50)   # vertical augmentation
    t = t.merge(t2, on="policy_id")                               # join
    t = t.merge(t1, on="policy_id")                               # join
    t = t.map_column("high_risk", lambda x: int(x))              # data transformation
    t = t.filter_mask(t.df["high_risk"] == 1)                   # horizontal reduction
    t = t.select_columns(["policy_id", "amount", "high_risk"])  # vertical reduction
    t = t.drop_columns(["high_risk"])                             # vertical reduction
    t = t.filter_mask(t.df["amount"] > 500)                      # horizontal reduction

    return t


if __name__ == "__main__":
    from tensprov import IndexedTensorArray
    result = build_pipeline(IndexedTensorArray)
    print(f"Output records: {len(result.df)}")
    print(result.df.head())
