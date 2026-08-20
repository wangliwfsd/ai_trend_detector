from pathlib import Path

import numpy as np
import yaml

from ai_trend_radar.pipeline import run_pipeline
from ai_trend_radar.providers import EmbeddingProvider
from ai_trend_radar.sample import sample_items


def test_sample_pipeline_runs_without_network_or_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = yaml.safe_load(Path("config.example.yaml").read_text())
    config["radar"]["database"] = str(tmp_path / "radar.db")
    config["radar"]["output_dir"] = str(tmp_path / "reports")
    result = run_pipeline(config, sample=True)
    assert result["collected"] == 22
    assert 3 <= result["trends"] <= 5
    report = result["markdown_path"].read_text()
    assert "AI Trend Radar" in report
    assert "必读" in report
    assert "做什么" in report
    assert "怎么做" in report
    assert "有什么不同" in report
    assert result["json_path"].exists()
    assert result["speech_path"].exists()
    assert result["audio_path"] is None
    speech = result["speech_path"].read_text()
    assert "目标时长：约 15 分钟" in speech
    assert "必读信号" in speech
    assert not (tmp_path / "radar.db").exists()
    assert (tmp_path / "radar-sample.db").exists()
    assert result["markdown_path"].parent.name == "sample"


class QuotaExhaustedProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError("429 RESOURCE_EXHAUSTED")


def test_runtime_embedding_error_falls_back_to_local(tmp_path: Path, monkeypatch):
    config = yaml.safe_load(Path("config.example.yaml").read_text())
    config["radar"]["database"] = str(tmp_path / "radar.db")
    config["radar"]["output_dir"] = str(tmp_path / "reports")
    config["llm"]["provider"] = "heuristic"
    monkeypatch.setattr(
        "ai_trend_radar.pipeline.collect_all",
        lambda _, progress=None: (sample_items(), []),
    )
    monkeypatch.setattr("ai_trend_radar.pipeline.make_embedding_provider", lambda _: QuotaExhaustedProvider())
    result = run_pipeline(config, sample=False)
    assert result["trends"] >= 3
    assert any("429 RESOURCE_EXHAUSTED" in warning for warning in result["warnings"])
