# TensProv

TensProv is a lightweight provenance tracking framework for tabular data pipelines in Python.  
It provides efficient provenance capture and lineage query execution using tensor-inspired provenance representations and optimized adaptive capture strategies.

The repository contains:

- The TensProv provenance engine
- Multiple provenance representations
- Synthetic and realistic benchmark pipelines
- Experimental scripts used for evaluation
- Reproducibility artifacts and benchmark results

---

# Repository Structure

```text
tensprov_lean/
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
git clone https://github.com/<your-username>/tensprov.git
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

Run optimized operator correctness validation:

```bash
PYTHONPATH=. python -m benchmarks.test_optimized_correctness_all_ops
```

---

# Main Experimental Scripts

## Realistic Pipeline Benchmark

```bash
PYTHONPATH=. python -m benchmarks.run_realistic_suite
```

## Operator Comparison

```bash
PYTHONPATH=. python -m benchmarks.run_operator_comparison_final
```

## Scalability Overhead Evaluation

```bash
PYTHONPATH=. python -m benchmarks.run_capture_overhead_scaling_fair
```

## Original Pipeline Evaluation

```bash
PYTHONPATH=. python -m benchmarks.run_original_pipelines_comparison
```

---

# Core Components

## Provenance Representations

TensProv includes several provenance representations:

- IndexedTensorArray (main optimized representation)
- IndexedTensor
- COOTensor
- ProvenanceGraph
- RelationalProvenance

## Optimized Capture Strategies

The optimized implementation includes adaptive provenance capture strategies for:

- Filter operations
- DropNA operations
- Merge/join operations
- Append operations

These optimizations significantly reduce provenance overhead while preserving lineage correctness.

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

The optimized pandas integration uses:

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
