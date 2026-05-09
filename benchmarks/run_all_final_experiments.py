PYTHONPATH=. python -m benchmarks.run_operator_comparison_by_representation > operator_comparison_by_representation.csv

PYTHONPATH=. python -m benchmarks.profile_filter_selectivity_breakdown > filter_selectivity_breakdown.csv

PYTHONPATH=. python -m benchmarks.run_operator_microbench_final > operator_microbench_final.csv

PYTHONPATH=. python -m benchmarks.pipeline_characteristics_final > pipeline_characteristics.csv

PYTHONPATH=. python -m benchmarks.run_realistic_suite > results_repeated_10times.csv

PYTHONPATH=. python -m benchmarks.run_capture_overhead_scaling_fair > capture_overhead_scaling_fair.csv
