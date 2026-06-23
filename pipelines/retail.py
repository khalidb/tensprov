"""
Pipeline: retail (Retail)
Domain:   Retail
Depth:    7 operations
Operators: dropna, assign, join, join, map, filter, vertical reduction

Corresponds to Table 9 (pipeline 'retail') in the TensProv paper.
Input datasets: customers (5,000), orders (20,000), products (1,000).
"""

import pandas as pd
from tensprov.pandas_wrapper import TrackedDataFrame


def build_pipeline(representation_cls):
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

    t = t2.dropna_rows()                                        # horizontal reduction
    t = t.assign_column("high_value", t.df["amount"] > 50)     # vertical augmentation
    t = t.merge(t1, on="cust_id")                               # join
    t = t.merge(t3, on="product_id")                            # join
    t = t.map_column("category", lambda x: "A" if x < 5 else "B")  # data transformation
    t = t.filter_mask(t.df["category"] == "A")                  # horizontal reduction
    t = t.select_columns(["cust_id", "category", "amount"])     # vertical reduction

    return t


if __name__ == "__main__":
    from tensprov import IndexedTensorArray
    result = build_pipeline(IndexedTensorArray)
    print(f"Output records: {len(result.df)}")
    print(result.df.head())
