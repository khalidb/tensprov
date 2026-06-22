# TensProv

TensProv is a lightweight provenance tracking framework for tabular data pipelines in Python.
It provides efficient provenance capture and lineage query execution using tensor-inspired
provenance representations and optimised adaptive capture strategies.

The repository contains:

- The TensProv provenance engine
- Multiple provenance representations
- Synthetic and realistic benchmark pipelines
- Experimental scripts used for evaluation
- Reproducibility artifacts and benchmark results

---

# Repository Structure

```text
tensprov/
├── benchmarks/     # Benchmark and evaluation scripts
├── results/        # Final experimental results
├── tensprov/       # Core TensProv implementation
├── tests/          # Correctness tests
├── README.md
├── LICENSE
└── requirements.txt
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/khalidb/tensprov.git
cd tensprov
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Tests

Run the correctness tests:

```bash
PYTHONPATH=. pytest
```

Run optimised operator correctness validation:

```bash
PYTHONPATH=. python -m benchmarks.test_optimized_correctness_all_ops
```

---

# Reproducing Paper Results

The following table maps each experimental result from the paper to the script that produces it.

| Paper result                                      | Script                                               |
|---------------------------------------------------|------------------------------------------------------|
| Table 10: operator-level capture cost             | `benchmarks.run_operator_comparison_final`           |
| Table 11: representation comparison (merge)       | `benchmarks.run_operator_comparison_final`           |
| Table 9, Fig 2, Fig 3, Table 12: pipeline-level   | `benchmarks.run_realistic_suite`                     |
| Fig 4, Table 13: capture overhead at scale        | `benchmarks.run_capture_overhead_scaling_fair`       |
| Table 14: on-demand recomputation vs re-execution | `benchmarks.run_original_pipelines_comparison`       |

---

# Main Experimental Scripts

## Realistic Pipeline Benchmark

Reproduces pipeline-level query performance, memory consumption, and query type coverage
(Tables 9, 12 and Figures 2, 3 in the paper).

```bash
PYTHONPATH=. python -m benchmarks.run_realistic_suite
```

## Operator Comparison

Reproduces operator-level provenance capture cost and representation comparison
(Tables 10 and 11 in the paper).

```bash
PYTHONPATH=. python -m benchmarks.run_operator_comparison_final
```

## Scalability Overhead Evaluation

Reproduces capture overhead ratios across scale factors
(Figure 4 and Table 13 in the paper).

```bash
PYTHONPATH=. python -m benchmarks.run_capture_overhead_scaling_fair
```

## Original Pipeline Evaluation

Reproduces on-demand recomputation results compared to full pipeline re-execution
(Table 14 in the paper).

```bash
PYTHONPATH=. python -m benchmarks.run_original_pipelines_comparison
```

---

# Core Components

## Provenance Representations

TensProv includes several provenance representations:

- `IndexedTensorArray` — main optimised representation (TensProv)
- `IndexedTensor` — standard tensor representation
- `COOTensor` — sparse coordinate format baseline (SparseTensor)
- `ProvenanceGraph` — graph-based baseline (ProvenanceGraph)
- `RelationalProvenance` — relational table baseline (RelationalProvenance)

## Optimised Capture Strategies

The optimised implementation includes adaptive provenance capture strategies for:

- Filter operations
- DropNA operations
- Merge/join operations
- Append operations

These optimisations significantly reduce provenance overhead while preserving lineage correctness.

---

# Results

Final benchmark results are stored in:

```text
results/
```

Including:

- Operator comparison benchmarks
- Pipeline scalability measurements
- Realistic pipeline evaluations
- Original pipeline reproductions

---

# Notes

The optimised pandas integration uses:

```python
tensprov.pandas_wrapper
```

The older:

```python
tensprov.tracked_dataframe
```

module is retained for backward-compatible tests.

---

# License

This project is released under the MIT License.
