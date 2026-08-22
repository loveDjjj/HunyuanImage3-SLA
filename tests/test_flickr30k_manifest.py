import json
import subprocess
import sys


def test_flickr30k_manifest_uses_unique_train_images(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (images / name).write_bytes(b"image")
    annotations = {"images": [
        {"filename": "a.jpg", "imgid": 1, "split": "train", "sentences": [{"raw": "first"}, {"raw": "second"}]},
        {"filename": "b.jpg", "imgid": 2, "split": "train", "sentences": [{"raw": "third"}]},
        {"filename": "missing.jpg", "imgid": 3, "split": "train", "sentences": [{"raw": "ignored"}]},
    ]}
    source = tmp_path / "dataset_flickr30k.json"
    source.write_text(json.dumps(annotations), encoding="utf-8")
    output = tmp_path / "metadata.jsonl"
    subprocess.run([sys.executable, "tools/prepare_flickr30k_manifest.py", "--annotations", str(source), "--images-dir", str(images), "--output", str(output), "--sample-count", "2"], check=True)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["1", "2"]
    assert rows[0]["caption"] in {"first", "second"}
