"""
Pipeline: education (Education)
Domain:   Education
Depth:    9 operations
Operators: dropna, impute, assign, join, join, map, filter, vertical reduction, vertical reduction

Corresponds to Table 9 (pipeline 'education') in the TensProv paper.
Input datasets: students (4,000), enrollments (12,000), grades (1,000).
"""

import pandas as pd
import numpy as np
from tensprov.pandas_wrapper import TrackedDataFrame


def build_pipeline(representation_cls):
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

    t = t3.dropna_rows()                                          # horizontal reduction
    t = t.impute_column("grade", strategy="mean")                 # data transformation (contextual)
    t = t.assign_column("passed", t.df["grade"] >= 50)           # vertical augmentation
    t = t.merge(t2, on="course_id")                               # join
    t = t.merge(t1, on="student_id")                              # join
    t = t.map_column("passed", lambda x: int(x))                 # data transformation
    t = t.filter_mask(t.df["passed"] == 0)                       # horizontal reduction
    t = t.select_columns(["student_id", "grade", "passed"])      # vertical reduction
    t = t.drop_columns(["passed"])                                # vertical reduction

    return t


if __name__ == "__main__":
    from tensprov import IndexedTensorArray
    result = build_pipeline(IndexedTensorArray)
    print(f"Output records: {len(result.df)}")
    print(result.df.head())
