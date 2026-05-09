from tensprov.indexed_tensor import IndexedTensor
from tensprov.indexed_tensor_array import IndexedTensorArray
from tensprov.graph_baseline import ProvenanceGraph
from tensprov.coo_baseline import COOTensor
from tensprov.synthetic import synthetic_join_coords
from tensprov.queries import forward, backward, co_contribution


DIMS = ("out", "left", "right")


def run_all_structures(coords):
    structures = {
        "indexed": IndexedTensor(DIMS),
        "array": IndexedTensorArray(DIMS),
        "graph": ProvenanceGraph(DIMS),
        "coo": COOTensor(DIMS),
    }

    for s in structures.values():
        s.add_many(coords)

    return structures


def assert_all_equal(results):
    ref_name = next(iter(results))
    ref = results[ref_name]

    for name, value in results.items():
        assert value == ref, f"{name} mismatch: expected {ref}, got {value}"


def test_forward_left_to_output():
    coords = synthetic_join_coords(n_left=100, n_right=200, fanout=3, seed=1)
    structures = run_all_structures(coords)

    left_ids = {1, 2, 3, 10}

    results = {
        name: forward(s, input_dim=1, input_ids=left_ids, output_dim=0)
        for name, s in structures.items()
    }

    assert_all_equal(results)


def test_backward_output_to_left():
    coords = synthetic_join_coords(n_left=100, n_right=200, fanout=3, seed=1)
    structures = run_all_structures(coords)

    output_ids = {0, 1, 2, 25, 50}

    results = {
        name: backward(s, output_ids=output_ids, input_dim=1, output_dim=0)
        for name, s in structures.items()
    }

    assert_all_equal(results)


def test_co_contribution_left_to_right():
    coords = synthetic_join_coords(n_left=100, n_right=200, fanout=3, seed=1)
    structures = run_all_structures(coords)

    left_ids = {4, 5, 6}

    results = {
        name: co_contribution(s, fixed_dim=1, fixed_ids=left_ids, other_dim=2)
        for name, s in structures.items()
    }

    assert_all_equal(results)


def test_single_record_queries():
    coords = synthetic_join_coords(n_left=100, n_right=200, fanout=3, seed=1)
    structures = run_all_structures(coords)

    left_ids = {7}

    results = {
        name: forward(s, input_dim=1, input_ids=left_ids, output_dim=0)
        for name, s in structures.items()
    }

    assert_all_equal(results)

from tensprov.provenance_engine import ProvenanceEngine


def build_pipeline(structure_class, coords_list):
    """
    Build a list of steps (one per operation).
    """
    steps = []

    for coords in coords_list:
        step = structure_class(DIMS)
        step.add_many(coords)
        steps.append(step)

    return ProvenanceEngine(steps)


def test_multi_step_forward():
    # Create 3 pipeline steps
    coords1 = synthetic_join_coords(n_left=100, n_right=200, fanout=2, seed=1)
    coords2 = synthetic_join_coords(n_left=200, n_right=300, fanout=2, seed=2)
    coords3 = synthetic_join_coords(n_left=300, n_right=400, fanout=2, seed=3)

    coords_list = [coords1, coords2, coords3]

    engines = {
        "indexed": build_pipeline(IndexedTensor, coords_list),
        "array": build_pipeline(IndexedTensorArray, coords_list),
        "graph": build_pipeline(ProvenanceGraph, coords_list),
        "coo": build_pipeline(COOTensor, coords_list),
    }

    input_ids = {1, 2, 3}

    results = {
        name: engine.forward(input_ids)
        for name, engine in engines.items()
    }

    assert_all_equal(results)


def test_multi_step_backward():
    # Same pipeline
    coords1 = synthetic_join_coords(n_left=100, n_right=200, fanout=2, seed=1)
    coords2 = synthetic_join_coords(n_left=200, n_right=300, fanout=2, seed=2)
    coords3 = synthetic_join_coords(n_left=300, n_right=400, fanout=2, seed=3)

    coords_list = [coords1, coords2, coords3]

    engines = {
        "indexed": build_pipeline(IndexedTensor, coords_list),
        "array": build_pipeline(IndexedTensorArray, coords_list),
        "graph": build_pipeline(ProvenanceGraph, coords_list),
        "coo": build_pipeline(COOTensor, coords_list),
    }

    # Pick some output IDs
    output_ids = {0, 1, 2, 10}

    results = {
        name: engine.backward(output_ids)
        for name, engine in engines.items()
    }

    assert_all_equal(results)    

from tensprov.operations import FilterOperation, JoinOperation


def test_filter_operation():
    coords = [(0, 2), (1, 4), (2, 8)]
    op = FilterOperation(coords)

    assert op.forward({2, 8}) == {0, 2}
    assert op.backward({0, 2}) == {2, 8}


def test_join_operation():
    coords = [
        (0, 10, 100),
        (1, 10, 101),
        (2, 11, 101),
    ]
    op = JoinOperation(coords)

    assert op.left_to_out({10}) == {0, 1}
    assert op.out_to_left({0, 2}) == {10, 11}
    assert op.out_to_right({0, 2}) == {100, 101}
    assert op.left_to_right({10}) == {100, 101}   

def test_operation_pipeline():
    # Step 1: filter (identity-like)
    filter_coords = [(i, i) for i in range(10)]
    f = FilterOperation(filter_coords)

    # Step 2: join
    join_coords = [
        (0, 1, 100),
        (1, 2, 101),
        (2, 3, 102),
    ]
    j = JoinOperation(join_coords)

    engine = ProvenanceEngine([f, j])

    result = engine.forward({1, 2, 3})

    assert result == {0, 1, 2}

import pandas as pd
from tensprov.tracked_dataframe import TrackedDataFrame
from tensprov.provenance_engine import ProvenanceEngine


def test_tracked_dataframe_filter():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [10, 20, 30, 40],
    })

    tdf = TrackedDataFrame(df)

    # Filter rows where a > 2
    tdf2 = tdf.filter_rows(lambda row: row["a"] > 2)

    # Check data
    assert list(tdf2.df["a"]) == [3, 4]

    # Build provenance engine
    engine = ProvenanceEngine(tdf2.provenance)

    # Forward: original IDs → filtered IDs
    result = engine.forward({2, 3})  # rows 3 and 4
    assert result == {0, 1}

    # Backward: filtered IDs → original IDs
    result_back = engine.backward({0, 1})
    assert result_back == {2, 3}    

def test_pandas_merge_provenance():
    import pandas as pd
    from tensprov.pandas_wrapper import TrackedDataFrame
    from tensprov.provenance_engine import ProvenanceEngine

    df1 = pd.DataFrame({
        "key": [1, 2, 3],
        "val1": ["a", "b", "c"],
    })

    df2 = pd.DataFrame({
        "key": [2, 3, 4],
        "val2": ["x", "y", "z"],
    })

    t1 = TrackedDataFrame(df1)
    t2 = TrackedDataFrame(df2)

    t3 = t1.merge(t2, on="key")

    assert len(t3.df) == 2

    engine = ProvenanceEngine(t3.provenance, input_dim=1, output_dim=0)

    result_left = engine.backward({0, 1})

    # Joined keys are 2 and 3, corresponding to left row IDs 1 and 2.
    assert result_left == {1, 2}

def test_pandas_pipeline_provenance():
    import pandas as pd
    from tensprov.pandas_wrapper import TrackedDataFrame
    from tensprov.provenance_engine import ProvenanceEngine

    # Left table
    df1 = pd.DataFrame({
        "key": [1, 2, 3, 4],
        "age": [20, 35, 40, 25],
    })

    # Right table
    df2 = pd.DataFrame({
        "key": [2, 3, 4],
        "city": ["A", "B", "C"],
    })

    t1 = TrackedDataFrame(df1)
    t2 = TrackedDataFrame(df2)

    # Step 1: filter left (age > 30)
    t1f = t1.filter(lambda row: row["age"] > 30)
    # Keeps keys 2 and 3 → original IDs {1,2}

    # Step 2: merge
    t3 = t1f.merge(t2, on="key")
    # Should produce rows for keys 2 and 3

    # Step 3: final filter (keep only key == 3)
    t4 = t3.filter(lambda row: row["key"] == 3)

    assert len(t4.df) == 1

    engine = ProvenanceEngine(t4.provenance, input_dim=1, output_dim=0)

    # Trace back the single output row
    result = engine.backward({0})

    # Expected:
    # key 3 came from original left row index 2
    assert result == {2}    
