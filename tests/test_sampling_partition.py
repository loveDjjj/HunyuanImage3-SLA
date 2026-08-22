import hashlib

from sampling.sample_latents import sample_partition


def test_numeric_sample_ids_use_modulo_partitioning():
    assert sample_partition("17", 8) == 1


def test_text_sample_ids_have_stable_partitioning():
    expected = int.from_bytes(hashlib.sha256(b"0:abc").digest()[:8], "little") % 4
    assert sample_partition("abc", 4) == expected
