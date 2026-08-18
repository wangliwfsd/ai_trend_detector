from __future__ import annotations

import hashlib
import html
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit, urlunsplit

import feedparser
import httpx
from bs4 import BeautifulSoup

from .models import Item

USER_AGENT = "ai-trend-radar/0.1 (personal research trend monitor)"


def collect_all(
    config: dict[str, Any],
    client: httpx.Client | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[Item], list[str]]:
    own_client = client is None
    client = client or httpx.Client(timeout=25, follow_redirects=True, headers={"User-Agent": USER_AGENT})
    items: list[Item] = []
    warnings: list[str] = []
    collectors: list[tuple[str, Callable[[], list[Item]]]] = []
    section = config.get("collectors", {})

    if section.get("arxiv", {}).get("enabled", True):
        collectors.append(("arXiv", lambda: collect_arxiv(client, section["arxiv"])))
    if section.get("huggingface", {}).get("enabled", True):
        collectors.append(("Hugging Face", lambda: collect_huggingface(client, section["huggingface"])))
    if section.get("github_trending", {}).get("enabled", False):
        collectors.append(("GitHub Trending", lambda: collect_github_trending(client, section["github_trending"])))
    if section.get("github_releases", {}).get("enabled", False):
        collectors.append(("GitHub Releases", lambda: collect_github_releases(client, section["github_releases"])))
    if section.get("rss", {}).get("enabled", False):
        collectors.append(("RSS", lambda: collect_rss(client, section["rss"])))

    for name, operation in collectors:
        if progress:
            progress(f"正在采集 {name}…")
        try:
            collected = operation()
            items.extend(collected)
            if progress:
                progress(f"{name} 完成：{len(collected)} 条")
        except Exception as exc:  # one weak source must not stop the report
            warnings.append(f"{name}: {type(exc).__name__}: {exc}")
            if progress:
                progress(f"{name} 失败，跳过并继续：{type(exc).__name__}")
    if own_client:
        client.close()
    deduplicated = deduplicate(items)
    if progress:
        progress(f"采集完成：原始 {len(items)} 条，去重后 {len(deduplicated)} 条")
    return deduplicated, warnings


def collect_arxiv(client: httpx.Client, config: dict[str, Any]) -> list[Item]:
    categories = config.get("categories", ["cs.CL", "cs.LG", "cs.AI"])
    query = " OR ".join(f"cat:{category}" for category in categories)
    params = {
        "search_query": query,
        "start": 0,
        "max_results": int(config.get("max_results", 250)),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = client.get(f"https://export.arxiv.org/api/query?{urlencode(params)}")
    response.raise_for_status()
    feed = feedparser.parse(response.content)
    result: list[Item] = []
    for entry in feed.entries:
        arxiv_id = entry.id.rsplit("/", 1)[-1].split("v", 1)[0]
        result.append(
            Item(
                uid=f"arxiv:{arxiv_id}",
                source="arXiv",
                kind="paper",
                title=_clean(entry.title),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                published_at=_parse_date(entry.published),
                summary=_clean(entry.get("summary", "")),
                authors=[author.name for author in entry.get("authors", [])],
                categories=[tag.term for tag in entry.get("tags", [])],
                metadata={"arxiv_id": arxiv_id},
            )
        )
    return result


def collect_huggingface(client: httpx.Client, config: dict[str, Any]) -> list[Item]:
    limit = int(config.get("limit", 100))
    response = client.get("https://huggingface.co/api/daily_papers", params={"limit": limit})
    response.raise_for_status()
    result: list[Item] = []
    for row in response.json()[:limit]:
        paper = row.get("paper", row)
        paper_id = str(paper.get("id") or paper.get("paperId") or row.get("id", ""))
        if not paper_id:
            continue
        published = paper.get("publishedAt") or row.get("publishedAt") or paper.get("submittedOnDailyAt")
        result.append(
            Item(
                uid=f"hf:{paper_id}",
                source="Hugging Face Papers",
                kind="paper",
                title=_clean(paper.get("title", "")),
                url=f"https://huggingface.co/papers/{paper_id}",
                published_at=_parse_date(published),
                summary=_clean(paper.get("summary") or paper.get("ai_summary") or ""),
                authors=[a.get("name", "") if isinstance(a, dict) else str(a) for a in paper.get("authors", [])],
                metrics={
                    "upvotes": float(paper.get("upvotes", row.get("upvotes", 0)) or 0),
                    "github_stars": float(paper.get("githubStars", 0) or 0),
                },
                metadata={"paper_id": paper_id},
            )
        )
    return result


def collect_github_trending(client: httpx.Client, config: dict[str, Any]) -> list[Item]:
    result: list[Item] = []
    now = datetime.now(timezone.utc)
    for language in config.get("languages", ["python"]):
        response = client.get(
            f"https://github.com/trending/{language}", params={"since": config.get("since", "daily")}
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for article in soup.select("article.Box-row"):
            anchor = article.select_one("h2 a")
            if not anchor:
                continue
            repo = re.sub(r"\s+", "", anchor.get_text()).strip("/")
            description = article.select_one("p")
            stars_today = article.select_one("span.float-sm-right")
            metric = _first_number(stars_today.get_text(" ") if stars_today else "")
            result.append(
                Item(
                    uid=f"github-trending:{now.date()}:{repo}",
                    source="GitHub Trending",
                    kind="repository",
                    title=repo,
                    url=f"https://github.com/{repo}",
                    published_at=now,
                    summary=_clean(description.get_text(" ") if description else ""),
                    metrics={"stars_today": float(metric)},
                    categories=[language],
                )
            )
    return result


def collect_github_releases(client: httpx.Client, config: dict[str, Any]) -> list[Item]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    result: list[Item] = []
    for repo in config.get("repositories", []):
        response = client.get(f"https://api.github.com/repos/{repo}/releases", params={"per_page": 10}, headers=headers)
        response.raise_for_status()
        for release in response.json():
            if release.get("draft"):
                continue
            tag = release.get("tag_name", release.get("name", "release"))
            result.append(
                Item(
                    uid=f"github-release:{repo}:{release['id']}",
                    source="GitHub Releases",
                    kind="release",
                    title=f"{repo} {tag}",
                    url=release["html_url"],
                    published_at=_parse_date(release.get("published_at") or release.get("created_at")),
                    summary=_clean(release.get("body", ""))[:3000],
                    metadata={"repository": repo, "tag": tag},
                )
            )
    return result


def collect_rss(client: httpx.Client, config: dict[str, Any]) -> list[Item]:
    result: list[Item] = []
    for source in config.get("feeds", []):
        response = client.get(source["url"])
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        for entry in feed.entries[:30]:
            url = entry.get("link", "")
            if not url:
                continue
            published = entry.get("published") or entry.get("updated")
            result.append(
                Item(
                    uid=f"rss:{hashlib.sha1(url.encode()).hexdigest()[:20]}",
                    source=source["name"],
                    kind="blog",
                    title=_clean(entry.get("title", "")),
                    url=url,
                    published_at=_parse_date(published),
                    summary=_clean(entry.get("summary") or entry.get("description") or ""),
                    authors=[entry.get("author", "")] if entry.get("author") else [],
                )
            )
    return result


def deduplicate(items: list[Item]) -> list[Item]:
    """Merge cross-lists and mirrored paper signals while preserving provenance."""
    merged: dict[str, Item] = {}
    for item in items:
        arxiv_match = re.search(
            r"(?:arxiv:|hf:|arxiv\.org/(?:abs|pdf)/|huggingface\.co/papers/)(\d{4}\.\d{4,5})",
            f"{item.uid} {item.url}",
        )
        key = f"arxiv:{arxiv_match.group(1)}" if arxiv_match else _fingerprint(item.title, item.url)
        if key not in merged:
            item.metadata.setdefault("signals", [item.source])
            merged[key] = item
            continue
        current = merged[key]
        signals = set(current.metadata.get("signals", [current.source]))
        signals.add(item.source)
        current.metadata["signals"] = sorted(signals)
        current.metrics.update({k: max(current.metrics.get(k, 0), v) for k, v in item.metrics.items()})
        current.categories = sorted(set(current.categories + item.categories))
        if len(item.summary) > len(current.summary):
            current.summary = item.summary
    return list(merged.values())


def _fingerprint(title: str, url: str) -> str:
    normalized_title = re.sub(r"[^a-z0-9]+", "", title.lower())
    if normalized_title:
        return f"title:{hashlib.sha1(normalized_title.encode()).hexdigest()}"
    parts = urlsplit(url)
    clean_url = urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
    return f"url:{hashlib.sha1(clean_url.encode()).hexdigest()}"


def _parse_date(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        try:
            return parsedate_to_datetime(value).astimezone(timezone.utc)
        except (TypeError, ValueError):
            pass
        return datetime.now(timezone.utc)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", BeautifulSoup(html.unescape(value or ""), "html.parser").get_text(" ")).strip()


def _first_number(value: str) -> int:
    match = re.search(r"[\d,]+", value)
    return int(match.group(0).replace(",", "")) if match else 0
