from pathlib import Path

from typer.testing import CliRunner

from ai_trend_radar.cli import app


def test_audio_command_uses_existing_script_without_running_pipeline(tmp_path: Path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    script = reports / "2026-08-20-script.md"
    script.write_text(
        "# AI 趋势雷达口播稿｜2026-08-20\n\n"
        "> 目标时长：约 15 分钟 · Provider: gemini\n\n"
        "大家好，这是需要朗读的正文。\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "radar:\n"
        "  timezone: Australia/Perth\n"
        f"  database: {tmp_path / 'radar.db'}\n"
        f"  output_dir: {reports}\n"
        "audio:\n"
        "  provider: gemini\n"
        f"  cache_dir: {tmp_path / 'cache'}\n",
        encoding="utf-8",
    )
    captured = {}

    monkeypatch.setattr(
        "ai_trend_radar.cli.run_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pipeline must not run")),
    )
    monkeypatch.setattr("ai_trend_radar.cli.make_tts_provider", lambda _: object())

    def fake_synthesize(text, output_path, provider, cache_dir, **kwargs):
        captured["text"] = text
        captured["output_path"] = output_path
        output_path.write_bytes(b"mp3")
        output_path.with_name("latest.mp3").write_bytes(b"mp3")
        return {"chunks": 1, "cache_hits": 0, "cache_misses": 1}

    monkeypatch.setattr("ai_trend_radar.cli.synthesize_episode", fake_synthesize)
    result = CliRunner().invoke(
        app,
        ["audio", "--config", str(config), "--script", str(script)],
    )

    assert result.exit_code == 0, result.output
    assert captured["text"] == "大家好，这是需要朗读的正文。"
    assert captured["output_path"] == reports / "2026-08-20.mp3"
    assert "不采集、不聚类、不生成总结或口播稿" in result.output
    assert "新生成 1" in result.output


def test_audio_command_reports_missing_script(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "radar:\n"
        "  timezone: UTC\n"
        f"  database: {tmp_path / 'radar.db'}\n"
        f"  output_dir: {tmp_path / 'reports'}\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["audio", "--config", str(config)])

    assert result.exit_code == 1
    assert "找不到口播稿" in result.output
