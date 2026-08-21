from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import typer

from .audio import make_tts_provider, synthesize_episode
from .config import load_config, load_env_file
from .pipeline import run_pipeline

app = typer.Typer(
    no_args_is_help=True,
    help="Build a daily AI trend radar with Markdown, JSON, a spoken script, and MP3.",
)


@app.callback()
def main() -> None:
    """AI research and engineering trend radar."""


@app.command()
def run(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="YAML configuration file"),
    sample: bool = typer.Option(False, "--sample", help="Use built-in data; no network or API key needed"),
) -> None:
    """Collect signals and write trends, a spoken script, and MP3 audio."""
    config = _resolve_config(config)
    typer.echo(f"AI Trend Radar 启动：{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")
    typer.echo(f"配置文件：{config.resolve()}")
    env_loaded = load_env_file(config.resolve().parent / ".env")
    if env_loaded:
        typer.echo("环境变量：已安全读取项目 .env（不会覆盖终端变量）")

    def show_progress(message: str) -> None:
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        typer.echo(f"[{timestamp}] {message}")

    result = run_pipeline(load_config(config), sample=sample, progress=show_progress)
    typer.echo(
        f"Done: {result['collected']} collected, {result['history_items']} in history, "
        f"{result['trends']} trends"
    )
    typer.echo(f"Markdown: {result['markdown_path']}")
    typer.echo(f"JSON: {result['json_path']}")
    if result["speech_path"]:
        typer.echo(f"Speech script: {result['speech_path']}")
    if result["audio_path"]:
        typer.echo(f"Audio: {result['audio_path']}")
        typer.echo(
            f"Audio cache: {result['audio_cache_hits']} hits, "
            f"{result['audio_cache_misses']} misses"
        )
    if result["embedding_cache_hits"] or result["embedding_cache_misses"]:
        typer.echo(
            f"Embedding cache: {result['embedding_cache_hits']} hits, "
            f"{result['embedding_cache_misses']} misses"
        )
    if result["warnings"]:
        typer.echo(f"Warnings: {len(result['warnings'])} (included in report)")


@app.command("audio")
def audio_only(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="YAML configuration file"),
    script: Path | None = typer.Option(
        None,
        "--script",
        "-s",
        help="Existing speech script; defaults to reports/latest-script.md",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="MP3 output path; defaults to reports/<script-date>.mp3",
    ),
) -> None:
    """Turn an existing speech script into MP3 without rebuilding the radar."""
    config_path = _resolve_config(config)
    typer.echo(f"AI Trend Radar 音频模式：{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")
    typer.echo("模式：仅生成音频，不采集、不聚类、不生成总结或口播稿")
    typer.echo(f"配置文件：{config_path.resolve()}")
    if load_env_file(config_path.resolve().parent / ".env"):
        typer.echo("环境变量：已安全读取项目 .env（不会覆盖终端变量）")
    loaded = load_config(config_path)
    output_dir = Path(loaded["radar"]["output_dir"])
    script_path = script or output_dir / "latest-script.md"
    if not script_path.exists():
        typer.echo(f"错误：找不到口播稿 {script_path}", err=True)
        raise typer.Exit(code=1)
    raw_script = script_path.read_text(encoding="utf-8")
    spoken_text = _extract_spoken_text(raw_script)
    if not spoken_text:
        typer.echo(f"错误：口播稿没有可朗读正文 {script_path}", err=True)
        raise typer.Exit(code=1)
    date = _infer_script_date(
        script_path,
        raw_script,
        loaded["radar"].get("timezone", "UTC"),
    )
    output_path = output or output_dir / f"{date}.mp3"
    typer.echo(f"口播稿：{script_path.resolve()}")
    typer.echo(f"音频输出：{output_path.resolve()}")

    def show_progress(message: str) -> None:
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        typer.echo(f"[{timestamp}] {message}")

    audio_config = loaded.get("audio", {})
    try:
        provider = make_tts_provider(loaded)
        typer.echo(f"TTS provider：{provider.__class__.__name__}")
        if hasattr(provider, "active"):
            typer.echo(f"TTS 首选模型：{provider.active.namespace}")
        stats = synthesize_episode(
            spoken_text,
            output_path,
            provider,
            Path(audio_config.get("cache_dir", "data/audio-cache")),
            style=audio_config.get(
                "style",
                "语速平稳、清晰、自然，像专业科技播客主播；英文缩写逐字母清楚发音。",
            ),
            chunk_chars=int(audio_config.get("chunk_chars", 700)),
            pause_ms=int(audio_config.get("pause_ms", 450)),
            bitrate=audio_config.get("bitrate", "128k"),
            cache_days=int(audio_config.get("cache_days", 14)),
            max_workers=int(audio_config.get("max_workers", 1)),
            progress=show_progress,
        )
    except Exception as exc:
        typer.echo(f"音频生成失败：{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"完成：{output_path.resolve()}")
    typer.echo(
        f"音频分段：{stats['chunks']}，缓存命中 {stats['cache_hits']}，"
        f"新生成 {stats['cache_misses']}"
    )


def _resolve_config(config: Path) -> Path:
    if not config.exists() and config.name == "config.yaml":
        example = Path("config.example.yaml")
        if example.exists():
            return example
    return config


def _extract_spoken_text(value: str) -> str:
    lines = value.strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].lstrip().startswith(">"):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def _infer_script_date(script_path: Path, value: str, timezone_name: str) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", script_path.name)
    if not match:
        match = re.search(r"^# .*?(\d{4}-\d{2}-\d{2})\s*$", value, re.MULTILINE)
    if match:
        return match.group(1)
    return datetime.now(ZoneInfo(timezone_name)).date().isoformat()


if __name__ == "__main__":
    app()
