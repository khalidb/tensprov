"""
Pipeline: social (Social Media)
Domain:   Social Media
Depth:    9 operations
Operators: assign, map, join, join, assign, map, vertical reduction,
           vertical reduction, filter

Corresponds to Table 9 (pipeline 'social') in the TensProv paper.
Input datasets: users (5,000), posts (15,000), interactions (15,000).
"""

import pandas as pd
from tensprov.pandas_wrapper import TrackedDataFrame


def build_pipeline(representation_cls):
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

    t = t2.assign_column("popular", t2.df["likes"] > 100)        # vertical augmentation
    t = t.map_column("popular", lambda x: int(x))                # data transformation
    t = t.merge(t1, on="user_id")                                 # join
    t = t.merge(t3, on="post_id")                                 # join
    t = t.assign_column("score", t.df["likes"] + t.df["shares"]) # vertical augmentation
    t = t.map_column("score", lambda x: int(x))                  # data transformation
    t = t.select_columns(["user_id", "likes", "shares", "score"]) # vertical reduction
    t = t.drop_columns(["shares"])                                 # vertical reduction
    t = t.filter_mask(t.df["score"] > 120)                       # horizontal reduction

    return t


if __name__ == "__main__":
    from tensprov import IndexedTensorArray
    result = build_pipeline(IndexedTensorArray)
    print(f"Output records: {len(result.df)}")
    print(result.df.head())
