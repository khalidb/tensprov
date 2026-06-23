"""
Pipeline: fraud (Finance)
Domain:   Finance
Depth:    8 operations
Operators: dropna, assign, join, join, map, filter, vertical reduction, vertical reduction

Corresponds to Table 9 (pipeline 'fraud') in the TensProv paper.
Input datasets: transactions (20,000), accounts (5,000), merchants (2,000).
"""

import pandas as pd
from tensprov.pandas_wrapper import TrackedDataFrame


def build_pipeline(representation_cls):
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

    t = t1.dropna_rows()                                                          # horizontal reduction
    t = t.assign_column("high_amount", t.df["amount"] > 300)                     # vertical augmentation
    t = t.merge(t2, on="account_id")                                              # join
    t = t.merge(t3, on="merchant_id")                                             # join
    t = t.map_column("category", lambda x: "risky" if x < 3 else "normal")      # data transformation
    t = t.filter_mask((t.df["high_amount"]) & (t.df["category"] == "risky"))    # horizontal reduction
    t = t.select_columns(["account_id", "merchant_id", "amount"])                # vertical reduction
    t = t.drop_columns(["merchant_id"])                                           # vertical reduction

    return t


if __name__ == "__main__":
    from tensprov import IndexedTensorArray
    result = build_pipeline(IndexedTensorArray)
    print(f"Output records: {len(result.df)}")
    print(result.df.head())
