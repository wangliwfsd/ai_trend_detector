from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai_trend_radar.config import load_env_file
from ai_trend_radar.models import Item
from ai_trend_radar.storage import Store
from ai_trend_radar.trends import _item_quality


def test_env_file_loads_without_overriding_shell(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EXISTING_VALUE", "from-shell")
    monkeypatch.setenv("EMPTY_VALUE", "")
    env = tmp_path / ".env"
    env.write_text("EXISTING_VALUE=from-file\nEMPTY_VALUE=filled\nNEW_VALUE='loaded'\n")
    assert load_env_file(env)
    assert __import__("os").environ["EXISTING_VALUE"] == "from-shell"
    assert __import__("os").environ["EMPTY_VALUE"] == "filled"
    assert __import__("os").environ["NEW_VALUE"] == "loaded"


def test_legacy_github_trending_rows_migrate_to_stable_uid(tmp_path: Path):
    path = tmp_path / "radar.db"
    first = Store(path)
    now = datetime.now(timezone.utc)
    first.upsert(
        [
            Item(
                "github-trending:2026-08-18:org/repo",
                "GitHub Trending",
                "repository",
                "org/repo",
                "https://github.com/org/repo",
                now - timedelta(days=1),
            ),
            Item(
                "github-trending:2026-08-19:org/repo",
                "GitHub Trending",
                "repository",
                "org/repo",
                "https://github.com/org/repo",
                now,
            ),
        ]
    )
    first.close()

    migrated = Store(path)
    try:
        items = migrated.recent(now + timedelta(minutes=1), 30)
        assert [item.uid for item in items] == ["github-trending:org/repo"]
    finally:
        migrated.close()


def test_item_quality_prefers_new_and_penalizes_recent_recommendation():
    now = datetime.now(timezone.utc)
    base = Item("a", "arXiv", "paper", "A", "https://a", now)
    new = Item("b", "arXiv", "paper", "B", "https://b", now, metadata={"is_new_today": True})
    repeated = Item(
        "c",
        "arXiv",
        "paper",
        "C",
        "https://c",
        now,
        metadata={"is_new_today": True, "recently_recommended": True},
    )
    assert _item_quality(new, now) > _item_quality(base, now)
    assert _item_quality(repeated, now) < _item_quality(base, now)
