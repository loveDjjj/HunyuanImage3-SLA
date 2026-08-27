from sampling.sample_vllm_trajectories import _group_rows_by_seed


def test_trajectory_rows_are_grouped_by_seed_without_reordering_groups():
    rows = [
        {"id": "a", "seed": "7"},
        {"id": "b", "seed": 9},
        {"id": "c", "seed": 7},
        {"id": "d"},
    ]

    grouped = _group_rows_by_seed(rows)

    assert [row["id"] for row in grouped[7]] == ["a", "c"]
    assert [row["id"] for row in grouped[9]] == ["b"]
    assert [row["id"] for row in grouped[42]] == ["d"]
