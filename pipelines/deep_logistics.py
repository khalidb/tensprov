"""
Pipeline: deep_logistics (Logistics)
Domain:   Logistics
Depth:    18 operations
Operators: impute, assign, map, join, join, filter, assign, map, filter,
           vertical reduction, assign, map, filter, vertical reduction,
           assign, map, filter, filter

Corresponds to Table 9 (pipeline 'deep_logistics') in the TensProv paper.
Input datasets: packages (30,000), routes (5,000), hubs (5,000).
"""

import pandas as pd
import numpy as np
from tensprov.pandas_wrapper import TrackedDataFrame


def build_pipeline(representation_cls):
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

    t = t3.impute_column("hub_score", strategy="mean")            # data transformation (contextual)
    t = t.assign_column("good_hub", t.df["hub_score"] > 50)      # vertical augmentation
    t = t.map_column("good_hub", lambda x: int(x))               # data transformation
    t = t.merge(t1, on="route_id")                                # join
    t = t.merge(t2, on="route_id")                                # join
    t = t.filter_mask(t.df["good_hub"] == 1)                     # horizontal reduction
    t = t.assign_column("heavy", t.df["weight"] > 20)            # vertical augmentation
    t = t.map_column("heavy", lambda x: int(x))                  # data transformation
    t = t.filter_mask(t.df["heavy"] == 1)                        # horizontal reduction
    t = t.select_columns(["pkg_id", "route_id", "weight", "zone"]) # vertical reduction
    t = t.assign_column("scaled_w", t.df["weight"] / 50.0)       # vertical augmentation
    t = t.map_column("scaled_w", lambda x: round(x, 2))          # data transformation
    t = t.filter_mask(t.df["scaled_w"] > 0.4)                    # horizontal reduction
    t = t.drop_columns(["scaled_w"])                               # vertical reduction
    t = t.assign_column("zone_flag", t.df["zone"] < 5)           # vertical augmentation
    t = t.map_column("zone_flag", lambda x: int(x))              # data transformation
    t = t.filter_mask(t.df["zone_flag"] == 1)                    # horizontal reduction
    t = t.filter_mask(t.df["weight"] > 25)                       # horizontal reduction

    return t


if __name__ == "__main__":
    from tensprov import IndexedTensorArray
    result = build_pipeline(IndexedTensorArray)
    print(f"Output records: {len(result.df)}")
    print(result.df.head())
