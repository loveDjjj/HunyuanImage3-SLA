import hashlib

from sampling.sample_latents import sample_partition, select_rank_rows, select_source_rows


def test_numeric_sample_ids_use_modulo_partitioning():
    assert sample_partition("17", 8) == 1


def test_text_sample_ids_have_stable_partitioning():
    expected = int.from_bytes(hashlib.sha256(b"0:abc").digest()[:8], "little") % 4
    assert sample_partition("abc", 4) == expected


def test_uneven_modulo_partitions_still_cover_every_selected_sample():
    rows = [{"id": sample_id} for sample_id in (1, 2, 4, 7, 10, 13, 16, 19)]
    selected = select_source_rows(rows, target_count=7)

    partitions = [select_rank_rows(selected, rank, 3, completed=set()) for rank in range(3)]

    assert sorted(row["id"] for partition in partitions for row in partition) == [1, 2, 4, 7, 10, 13, 16]
    assert [len(partition) for partition in partitions] == [0, 6, 1]


def test_rank_selection_skips_samples_completed_by_an_old_world_size():
    rows = [{"id": sample_id} for sample_id in range(6)]

    remaining = [select_rank_rows(rows, rank, 2, completed={"1", "4"}) for rank in range(2)]

    assert sorted(row["id"] for partition in remaining for row in partition) == [0, 2, 3, 5]
