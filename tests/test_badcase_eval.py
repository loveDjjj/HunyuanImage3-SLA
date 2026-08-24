import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from tools.prepare_badcase_eval import prepare_task
from tools.run_badcase_eval import i2i_request, request_with_retries, t2i_request


def png_bytes(color="red"):
    output = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(output, format="PNG")
    return output.getvalue()


class DownloadResponse:
    status_code = 200
    headers = {"Content-Type": "image/png"}

    def __init__(self, content):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        del chunk_size
        yield self.content


class DownloadSession:
    def __init__(self, content):
        self.content = content
        self.headers = {}

    def get(self, url, timeout, stream):
        del url, timeout, stream
        return DownloadResponse(self.content)


def test_prepare_i2i_downloads_inputs_and_baseline(tmp_path, monkeypatch):
    task_dir = tmp_path / "badcase_ti2i"
    task_dir.mkdir()
    cases = [
        {
            "task": "i2i",
            "index": "687",
            "seed": "188198",
            "prompt": "edit",
            "inputs": ["/old/input.png"],
            "image_urls": ["https://example.test/source_687.png?signature=x"],
            "baseline_url": "https://example.test/baseline_687.png?signature=x",
        }
    ]
    (task_dir / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    monkeypatch.setattr(
        "tools.prepare_badcase_eval.requests.Session",
        lambda: DownloadSession(png_bytes()),
    )

    downloaded, failures = prepare_task(task_dir, "badcase_ti2i", timeout=1)

    saved = json.loads((task_dir / "cases.json").read_text())
    assert downloaded == 2
    assert failures == []
    assert Path(saved[0]["inputs"][0]).is_file()
    assert Path(saved[0]["inputs"][0]).parent.name == "687"
    assert Path(saved[0]["baseline_image"]).is_file()
    assert Path(saved[0]["baseline_image"]).parent.name == "687"
    assert (task_dir / "output_images").is_dir()


def test_prepare_t2i_downloads_only_baseline(tmp_path, monkeypatch):
    task_dir = tmp_path / "badcase_t2i"
    task_dir.mkdir()
    cases = [
        {
            "task": "t2i",
            "index": "10",
            "seed": "7",
            "prompt": "draw",
            "baseline_url": "https://example.test/baseline.png",
        }
    ]
    (task_dir / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    monkeypatch.setattr(
        "tools.prepare_badcase_eval.requests.Session",
        lambda: DownloadSession(png_bytes()),
    )

    _, failures = prepare_task(task_dir, "badcase_t2i", timeout=1)

    saved = json.loads((task_dir / "cases.json").read_text())
    assert failures == []
    assert Path(saved[0]["baseline_image"]).is_file()
    assert not (task_dir / "input_images").exists()


def test_prepare_i2i_input_still_downloads_when_baseline_fails(tmp_path, monkeypatch):
    task_dir = tmp_path / "badcase_ti2i"
    task_dir.mkdir()
    cases = [
        {
            "task": "i2i",
            "index": "11",
            "seed": "7",
            "prompt": "edit",
            "image_urls": ["https://example.test/input.png"],
        }
    ]
    (task_dir / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    monkeypatch.setattr(
        "tools.prepare_badcase_eval.requests.Session",
        lambda: DownloadSession(png_bytes()),
    )

    _, failures = prepare_task(task_dir, "badcase_ti2i", timeout=1)

    saved = json.loads((task_dir / "cases.json").read_text())
    assert failures == ["index=11 baseline: case has no baseline_url"]
    assert Path(saved[0]["inputs"][0]).is_file()


def args(**overrides):
    values = {
        "model": None,
        "steps": 8,
        "t2i_size": "1024x1024",
        "i2i_size": "auto",
        "bot_task": None,
        "system_prompt_type": None,
        "retries": 0,
        "timeout": 10,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ApiResponse:
    status_code = 200
    text = ""

    def json(self):
        return {"data": [{"b64_json": base64.b64encode(png_bytes("blue")).decode()}]}


def test_api_requests_use_distilled_steps_seed_and_correct_endpoints(
    tmp_path, monkeypatch
):
    calls = []

    def fake_request(session, method, url, retries, timeout, **kwargs):
        del session, retries, timeout
        calls.append((method, url, kwargs))
        return ApiResponse()

    monkeypatch.setattr("tools.run_badcase_eval.request_with_retries", fake_request)
    row = {"prompt": "draw", "seed": "405965"}
    assert t2i_request(object(), "http://server", row, args()) == png_bytes("blue")
    assert calls[0][1].endswith("/v1/images/generations")
    assert calls[0][2]["json"]["num_inference_steps"] == 8
    assert calls[0][2]["json"]["seed"] == 405965

    source = tmp_path / "input.png"
    source.write_bytes(png_bytes())
    assert i2i_request(
        object(), "http://server", {**row, "inputs": [str(source)]}, args()
    ) == png_bytes("blue")
    assert calls[1][1].endswith("/v1/images/edits")
    assert calls[1][2]["data"]["num_inference_steps"] == "8"
    assert calls[1][2]["files"][0][0] == "image"


def test_multipart_retry_rewinds_file_handle(tmp_path):
    source = tmp_path / "input.png"
    content = png_bytes()
    source.write_bytes(content)

    class RetrySession:
        def __init__(self):
            self.bodies = []

        def request(self, method, url, timeout, **kwargs):
            del method, url, timeout
            handle = kwargs["files"][0][1][1]
            self.bodies.append(handle.read())
            return SimpleNamespace(
                status_code=503 if len(self.bodies) == 1 else 200, text="retry"
            )

    session = RetrySession()
    with source.open("rb") as handle:
        response = request_with_retries(
            session,
            "POST",
            "http://server/v1/images/edits",
            retries=1,
            timeout=1,
            files=[("image", (source.name, handle, "image/png"))],
        )
    assert response.status_code == 200
    assert session.bodies == [content, content]
