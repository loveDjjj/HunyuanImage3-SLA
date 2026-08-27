import json

from common.training_metrics import MetricsLogger
from tools.plot_training_metrics import plot_metrics


def test_metrics_logger_truncates_future_records_on_resume(tmp_path):
    logger = MetricsLogger(tmp_path, resume_step=0, ema_decay=0.5)
    logger.append({"step": 1, "loss": 4.0})
    logger.append({"step": 2, "loss": 2.0})

    resumed = MetricsLogger(tmp_path, resume_step=1, ema_decay=0.5)
    record = resumed.append({"step": 2, "loss": 2.0})

    rows = [json.loads(line) for line in resumed.path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert record["loss_ema"] == 3.0


def test_metrics_plot_writes_png_and_html(tmp_path):
    logger = MetricsLogger(tmp_path, resume_step=0)
    logger.append(
        {
            "step": 1,
            "loss": 1.0,
            "gradient_norm": 2.0,
            "proj_l_grad_norm": 1.0,
            "step_seconds": 3.0,
            "samples_per_second": 4.0,
            "validation_mse": 0.5,
            "validation_mse_by_step": [0.5] * 8,
        }
    )
    png = tmp_path / "training_metrics.png"
    html = tmp_path / "index.html"
    plot_metrics(logger.path, png, html)
    assert png.stat().st_size > 0
    assert "training_metrics.png" in html.read_text()
