from __future__ import annotations

from pathlib import Path
from datetime import datetime

import typer

from .config import load_config
from .pipeline import run_pipeline

app = typer.Typer(no_args_is_help=True, help="Build a daily AI research and engineering trend radar.")


@app.callback()
def main() -> None:
    """AI research and engineering trend radar."""


@app.command()
def run(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="YAML configuration file"),
    sample: bool = typer.Option(False, "--sample", help="Use built-in data; no network or API key needed"),
) -> None:
    """Collect signals, detect 7/30-day trends, and write Markdown + JSON."""
    if not config.exists() and config.name == "config.yaml":
        example = Path("config.example.yaml")
        if example.exists():
            config = example
    typer.echo(f"AI Trend Radar 启动：{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")
    typer.echo(f"配置文件：{config.resolve()}")

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
    if result["embedding_cache_hits"] or result["embedding_cache_misses"]:
        typer.echo(
            f"Embedding cache: {result['embedding_cache_hits']} hits, "
            f"{result['embedding_cache_misses']} misses"
        )
    if result["warnings"]:
        typer.echo(f"Warnings: {len(result['warnings'])} (included in report)")


if __name__ == "__main__":
    app()
