import json

from tools.build_badcase_validation_manifest import build_manifest


def test_badcase_validation_manifest_preserves_prompt_seed_and_limit(tmp_path):
    cases = [
        {"index": "7", "prompt": "first", "seed": "101"},
        {"index": "8", "prompt": "second", "seed": "202"},
    ]
    source = tmp_path / "cases.json"
    output = tmp_path / "prompts.jsonl"
    source.write_text(json.dumps(cases), encoding="utf-8")

    rows = build_manifest(source, output, limit=1)

    assert rows == [
        {
            "id": "badcase_t2i_7",
            "prompt": "first",
            "seed": 101,
            "source_index": "7",
        }
    ]
    assert json.loads(output.read_text()) == rows[0]
