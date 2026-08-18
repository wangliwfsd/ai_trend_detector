from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .collectors import collect_all, deduplicate
from .models import Item
from .providers import (
    CachedEmbeddingProvider,
    HeuristicNarrator,
    LocalEmbeddingProvider,
    make_embedding_provider,
    make_narrator,
)
from .report import render_markdown, write_reports
from .sample import SAMPLE_AS_OF, sample_items
from .storage import Store
from .trends import detect_trends


def run_pipeline(
    config: dict[str, Any],
    sample: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    notify = progress or (lambda _: None)
    warnings: list[str] = []
    as_of = SAMPLE_AS_OF if sample else datetime.now(timezone.utc)
    fresh_items: list[Item]
    notify("[1/6] 准备采集数据源")
    if sample:
        notify("正在载入内置离线样例…")
        fresh_items = sample_items()
        notify(f"离线样例完成：{len(fresh_items)} 条")
    else:
        fresh_items, collector_warnings = collect_all(config, progress=notify)
        warnings.extend(collector_warnings)

    notify("[2/6] 正在更新 SQLite 历史库")
    store = Store(config["radar"]["database"])
    cache_hits = 0
    cache_misses = 0
    try:
        stored_count = store.upsert(deduplicate(fresh_items))
        items = store.recent(as_of, int(config["radar"].get("lookback_days", 30)))
        notify(f"历史库完成：本次写入 {stored_count} 条，30 天窗口 {len(items)} 条")
        notify("[3/6] 准备 embedding 与主题聚类")
        if sample:
            embeddings = LocalEmbeddingProvider()
            notify("离线样例使用本地 HashingVectorizer")
        else:
            try:
                embeddings = make_embedding_provider(config)
                provider_name = embeddings.__class__.__name__
                notify(f"Embedding provider：{provider_name}")
                if config.get("embedding", {}).get("cache", True):
                    embeddings = CachedEmbeddingProvider(embeddings, store, progress=notify)
            except Exception as exc:
                warnings.append(f"Embedding provider 初始化失败，已降级为本地：{type(exc).__name__}: {exc}")
                embeddings = LocalEmbeddingProvider()
        try:
            trends = detect_trends(items, as_of, embeddings, config)
        except Exception as exc:
            warnings.append(f"Embedding 运行失败，已降级为本地：{type(exc).__name__}: {exc}")
            notify(f"Embedding 失败，正在切换本地降级：{type(exc).__name__}")
            trends = detect_trends(items, as_of, LocalEmbeddingProvider(), config)
        notify(f"主题聚类完成：发现 {len(trends)} 个候选趋势")
        if isinstance(embeddings, CachedEmbeddingProvider):
            cache_hits = embeddings.hits
            cache_misses = embeddings.misses
    finally:
        store.close()

    top_n = int(config["radar"].get("top_trends", 5))
    candidate_limit = max(top_n, int(config.get("llm", {}).get("candidate_trends", 8)))
    trends = trends[:candidate_limit]
    notify(f"[4/6] 正在为 {len(trends)} 个候选生成趋势名称、排序、摘要和必读方法概览")
    if sample:
        narrator = HeuristicNarrator()
        notify("离线样例使用启发式摘要")
    else:
        try:
            narrator = make_narrator(config)
            notify(f"摘要 provider：{narrator.__class__.__name__}")
        except Exception as exc:
            warnings.append(f"Gemini 摘要降级为本地：{type(exc).__name__}: {exc}")
            notify(f"Gemini 摘要不可用，改用启发式摘要：{type(exc).__name__}")
            narrator = HeuristicNarrator()
    try:
        trends = narrator.enrich(trends, config["radar"].get("report_language", "zh-CN"))
    except Exception as exc:
        warnings.append(f"摘要运行失败，已降级为本地：{type(exc).__name__}: {exc}")
        notify(f"摘要请求失败，正在使用启发式摘要：{type(exc).__name__}")
        trends = HeuristicNarrator().enrich(
            trends, config["radar"].get("report_language", "zh-CN")
        )
    notify("摘要与排序完成")
    trends = trends[: max(3, min(5, top_n))]
    notify("[5/6] 正在渲染 Markdown 与 JSON")
    markdown = render_markdown(
        trends,
        as_of,
        config["radar"].get("timezone", "UTC"),
        int(config["radar"].get("must_reads_per_trend", 2)),
        warnings,
    )
    markdown_path, json_path = write_reports(
        config["radar"]["output_dir"], markdown, trends, as_of, config["radar"].get("timezone", "UTC")
    )
    notify(f"[6/6] 完成：已写入 {markdown_path.name} 和 {json_path.name}")
    return {
        "collected": stored_count,
        "history_items": len(items),
        "trends": len(trends),
        "warnings": warnings,
        "embedding_cache_hits": cache_hits,
        "embedding_cache_misses": cache_misses,
        "markdown_path": Path(markdown_path),
        "json_path": Path(json_path),
    }
