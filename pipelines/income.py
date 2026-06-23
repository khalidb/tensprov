"""
Pipeline: income (Demographics)
Domain:   Demographics
Depth:    4 operations
Operators: filter, vertical reduction, join, filter

Corresponds to Table 9 (pipeline 'income') in the TensProv paper.
Input datasets: people (10,000 records), education levels (5 records).
"""

import pandas as pd
from tensprov.pandas_wrapper import TrackedDataFrame


def build_pipeline(representation_cls):
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

    t = t1.filter_mask(t1.df["age"] >= 25)        # horizontal reduction
    t = t.drop_columns(["unused_col"])              # vertical reduction
    t = t.merge(t2, on="education")                 # join
    t = t.filter_mask(t.df["level"] == "high")     # horizontal reduction

    return t


if __name__ == "__main__":
    from tensprov import IndexedTensorArray
    result = build_pipeline(IndexedTensorArray)
    print(f"Output records: {len(result.df)}")
    print(result.df.head())
