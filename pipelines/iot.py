"""
Pipeline: iot (IoT)
Domain:   Internet of Things
Depth:    12 operations
Operators: impute, assign, map, join, join, filter, vertical reduction,
           vertical reduction, filter, assign, map, filter

Corresponds to Table 9 (pipeline 'iot') in the TensProv paper.
Input datasets: devices (5,000), readings (30,000), locations (5,000).
"""

import pandas as pd
import numpy as np
from tensprov.pandas_wrapper import TrackedDataFrame


def build_pipeline(representation_cls):
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

    t = t2.impute_column("value", strategy="mean")               # data transformation (contextual)
    t = t.assign_column("high", t.df["value"] > 70)             # vertical augmentation
    t = t.map_column("high", lambda x: int(x))                  # data transformation
    t = t.merge(t1, on="device_id")                              # join
    t = t.merge(t3, on="device_id")                              # join
    t = t.filter_mask(t.df["high"] == 1)                        # horizontal reduction
    t = t.select_columns(["device_id", "value", "zone", "high"]) # vertical reduction
    t = t.drop_columns(["high"])                                  # vertical reduction
    t = t.filter_mask(t.df["zone"] < 5)                         # horizontal reduction
    t = t.assign_column("scaled", t.df["value"] / 100.0)        # vertical augmentation
    t = t.map_column("scaled", lambda x: round(x, 2))           # data transformation
    t = t.filter_mask(t.df["scaled"] > 0.5)                     # horizontal reduction

    return t


if __name__ == "__main__":
    from tensprov import IndexedTensorArray
    result = build_pipeline(IndexedTensorArray)
    print(f"Output records: {len(result.df)}")
    print(result.df.head())
