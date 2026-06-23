# Evaluation Pipelines

This folder contains the ten data preparation pipelines used in the TensProv evaluation (Table 9 of the paper). Each pipeline is defined as a standalone Python file with a `build_pipeline(representation_cls)` function.

| File | Domain | Depth | Input records |
|------|--------|-------|---------------|
| `income.py` | Demographics | 4 ops | 10,000 people + 5 education levels |
| `retail.py` | Retail | 7 ops | 5,000 customers + 20,000 orders + 1,000 products |
| `healthcare.py` | Healthcare | 8 ops | 5,000 patients + 15,000 visits + 3,000 lab results |
| `fraud.py` | Finance | 8 ops | 20,000 transactions + 5,000 accounts + 2,000 merchants |
| `education.py` | Education | 9 ops | 4,000 students + 12,000 enrollments + 1,000 grades |
| `insurance.py` | Insurance | 10 ops | 5,000 policies + 15,000 claims + 5,000 risk scores |
| `iot.py` | IoT | 12 ops | 5,000 devices + 30,000 readings + 5,000 locations |
| `deep_hr.py` | HR | 14 ops | 8,000 employees + 50 departments + 8,000 performance scores |
| `deep_logistics.py` | Logistics | 18 ops | 30,000 packages + 5,000 routes + 5,000 hubs |
| `social.py` | Social Media | 9 ops | 5,000 users + 15,000 posts + 15,000 interactions |

## Running a pipeline

```python
from tensprov import IndexedTensorArray
from pipelines.retail import build_pipeline

result = build_pipeline(IndexedTensorArray)
print(f"Output records: {len(result.df)}")
```

## Operator annotations

Each pipeline file annotates every operation with its category from the TensProv taxonomy (horizontal reduction, vertical reduction, data transformation, vertical augmentation, join). Operations marked as `(contextual)` materialise their output as described in Section 3.5 of the paper.
