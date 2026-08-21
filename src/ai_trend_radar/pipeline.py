from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from .audio import (
    make_tts_provider,
    normalize_language,
    resolve_audio_chunk_chars,
    resolve_audio_language,
    resolve_audio_style,
    synthesize_episode,
)
from .collectors import collect_all, deduplicate
from .deep_reading import enrich_must_reads
from .models import Item, SpeechScript
from .providers import (
    CachedEmbeddingProvider,
    HeuristicNarrator,
    HeuristicSpeechWriter,
    LocalEmbeddingProvider,
    make_embedding_provider,
    make_narrator,
    make_speech_writer,
)
from .report import render_markdown, write_reports, write_speech_script
from .sample import SAMPLE_AS_OF, sample_items
from .storage import Store
from .trends import arrange_must_reads, detect_trends


def run_pipeline(
    config: dict[str, Any],
    sample: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    config = deepcopy(config)
    config["radar"]["report_language"] = normalize_language(
        config["radar"].get("report_language", "zh-CN")
    )
    # Validate the independently configurable speech/audio language before any API calls.
    resolve_audio_language(config)
    notify = progress or (lambda _: None)
    warnings: list[str] = []
    as_of = SAMPLE_AS_OF if sample else datetime.now(timezone.utc)
    if sample:
        database = Path(config["radar"]["database"])
        config["radar"]["database"] = str(database.with_name(f"{database.stem}-sample{database.suffix}"))
        config["radar"]["output_dir"] = str(Path(config["radar"]["output_dir"]) / "sample")
    fresh_items: list[Item]
    notify("[1/9] 准备采集数据源")
    if sample:
        notify("正在载入内置离线样例…")
        fresh_items = sample_items()
        notify(f"离线样例完成：{len(fresh_items)} 条")
    else:
        fresh_items, collector_warnings = collect_all(config, progress=notify)
        warnings.extend(collector_warnings)

    notify("[2/9] 正在更新 SQLite 历史库")
    store = Store(config["radar"]["database"])
    cache_hits = 0
    cache_misses = 0
    try:
        stored_count = store.upsert(deduplicate(fresh_items))
        items = store.recent(as_of, int(config["radar"].get("lookback_days", 30)))
        _mark_item_context(items, as_of, config["radar"])
        notify(f"历史库完成：本次写入 {stored_count} 条，30 天窗口 {len(items)} 条")
        notify("[3/9] 准备 embedding 与主题聚类")
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
    notify(f"[4/9] 正在为 {len(trends)} 个候选生成趋势名称、排序和摘要")
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
    narrated_candidate_count = len(trends)
    try:
        trends = narrator.enrich(trends, config["radar"].get("report_language", "zh-CN"))
    except Exception as exc:
        warnings.append(f"摘要运行失败，已降级为本地：{type(exc).__name__}: {exc}")
        notify(f"摘要请求失败，正在使用启发式摘要：{type(exc).__name__}")
        trends = HeuristicNarrator().enrich(
            trends, config["radar"].get("report_language", "zh-CN")
        )
    rejected_count = narrated_candidate_count - len(trends)
    if rejected_count > 0:
        notify(f"一致性审查：剔除 {rejected_count} 个错配或重复候选趋势")
    notify("摘要与排序完成")
    if hasattr(narrator, "model"):
        notify(f"趋势摘要实际模型：{narrator.model}")
    trends = _select_unique_labels(trends, max(3, min(5, top_n)))
    must_reads = int(config["radar"].get("must_reads_per_trend", 2))
    for trend in trends:
        arrange_must_reads(trend, must_reads)
    deep_read_hits = 0
    deep_read_misses = 0
    if sample:
        notify("[5/9] 离线样例跳过原文深读")
    elif config.get("deep_reading", {}).get("enabled", True):
        notify(f"[5/9] 正在深读最终必读项：论文 PDF / 工程正文（最多 {len(trends) * must_reads} 篇）")
        try:
            deep_read_hits, deep_read_misses, deep_warnings = enrich_must_reads(
                trends,
                config,
                config["radar"].get("report_language", "zh-CN"),
                must_reads,
                progress=notify,
            )
            warnings.extend(deep_warnings)
            notify(f"原文深读完成：缓存 {deep_read_hits}，新分析 {deep_read_misses}")
        except Exception as exc:
            warnings.append(f"原文深读初始化失败，已保留摘要级分析：{type(exc).__name__}: {exc}")
            notify(f"原文深读不可用，保留摘要级分析：{type(exc).__name__}: {exc}")
    else:
        notify("[5/9] 原文深读已在配置中关闭")
    speech_path: Path | None = None
    speech: SpeechScript | None = None
    speech_config = config.get("speech", {})
    if speech_config.get("enabled", True):
        target_minutes = max(5, min(30, int(speech_config.get("target_minutes", 15))))
        speech_language = resolve_audio_language(config)
        notify(
            f"[6/9] 正在基于原文分析生成约 {target_minutes} 分钟的每日口播稿"
            f"（{speech_language}）"
        )
        if sample:
            speech_writer = HeuristicSpeechWriter()
            notify("离线样例使用本地口播稿生成器")
        else:
            try:
                speech_writer = make_speech_writer(config)
                notify(f"口播稿 provider：{speech_writer.__class__.__name__}")
            except Exception as exc:
                warnings.append(f"Gemini 口播稿降级为本地：{type(exc).__name__}: {exc}")
                notify(f"Gemini 口播稿不可用，改用本地生成：{type(exc).__name__}")
                speech_writer = HeuristicSpeechWriter()
        report_date = (
            as_of.astimezone(ZoneInfo(config["radar"].get("timezone", "UTC")))
            .date()
            .isoformat()
        )
        try:
            speech = speech_writer.write(
                trends,
                speech_language,
                target_minutes,
                report_date,
            )
        except Exception as exc:
            warnings.append(f"口播稿生成失败，已降级为本地：{type(exc).__name__}: {exc}")
            notify(f"口播稿请求失败，正在使用本地生成：{type(exc).__name__}: {exc}")
            speech = HeuristicSpeechWriter().write(
                trends,
                speech_language,
                target_minutes,
                report_date,
            )
        speech_path = write_speech_script(
            config["radar"]["output_dir"],
            speech,
            as_of,
            config["radar"].get("timezone", "UTC"),
            speech_language,
        )
        if hasattr(speech_writer, "model"):
            notify(f"口播稿实际模型：{speech_writer.model}")
        notify(f"口播稿完成：{speech_path.name}")
    else:
        notify("[6/9] 口播稿已在配置中关闭")

    audio_path: Path | None = None
    audio_stats = {"chunks": 0, "cache_hits": 0, "cache_misses": 0}
    audio_config = config.get("audio", {})
    if sample:
        notify("[7/9] 离线样例跳过音频合成")
    elif not audio_config.get("enabled", False):
        notify("[7/9] 音频合成已在配置中关闭")
    elif speech is None:
        warnings.append("音频合成已跳过：口播稿未启用或未生成")
        notify("[7/9] 音频合成已跳过：没有口播稿")
    else:
        notify("[7/9] 准备分段合成音频并拼接 MP3")
        try:
            tts_provider = make_tts_provider(config)
            notify(f"TTS provider：{tts_provider.__class__.__name__}")
            notify(f"TTS language：{resolve_audio_language(config)}")
            if hasattr(tts_provider, "active"):
                notify(f"TTS 首选模型：{tts_provider.active.namespace}")
            local_date = as_of.astimezone(
                ZoneInfo(config["radar"].get("timezone", "UTC"))
            ).date().isoformat()
            target_audio_path = Path(config["radar"]["output_dir"]) / f"{local_date}.mp3"
            audio_stats = synthesize_episode(
                speech.content,
                target_audio_path,
                tts_provider,
                Path(audio_config.get("cache_dir", "data/audio-cache")),
                style=resolve_audio_style(config),
                chunk_chars=resolve_audio_chunk_chars(config),
                pause_ms=int(audio_config.get("pause_ms", 450)),
                bitrate=audio_config.get("bitrate", "128k"),
                cache_days=int(audio_config.get("cache_days", 14)),
                max_workers=int(audio_config.get("max_workers", 1)),
                progress=notify,
            )
            audio_path = target_audio_path
            notify(f"MP3 完成：{audio_path.name}（{audio_stats['chunks']} 段）")
        except Exception as exc:
            audio_path = None
            warnings.append(f"音频合成失败，文本报告不受影响：{type(exc).__name__}: {exc}")
            notify(f"音频合成失败，继续输出文本报告：{type(exc).__name__}: {exc}")

    notify("[8/9] 正在渲染 Markdown 与 JSON")
    markdown = render_markdown(
        trends,
        as_of,
        config["radar"].get("timezone", "UTC"),
        must_reads,
        warnings,
        config["radar"].get("report_language", "zh-CN"),
    )
    markdown_path, json_path = write_reports(
        config["radar"]["output_dir"],
        markdown,
        trends,
        as_of,
        config["radar"].get("timezone", "UTC"),
        must_reads,
        config["radar"].get("report_language", "zh-CN"),
    )
    notify(f"[9/9] 完成：已写入 {markdown_path.name} 和 {json_path.name}")
    return {
        "collected": stored_count,
        "history_items": len(items),
        "trends": len(trends),
        "warnings": warnings,
        "embedding_cache_hits": cache_hits,
        "embedding_cache_misses": cache_misses,
        "deep_read_cache_hits": deep_read_hits,
        "deep_read_cache_misses": deep_read_misses,
        "markdown_path": Path(markdown_path),
        "json_path": Path(json_path),
        "speech_path": Path(speech_path) if speech_path else None,
        "audio_path": Path(audio_path) if audio_path else None,
        "audio_chunks": audio_stats["chunks"],
        "audio_cache_hits": audio_stats["cache_hits"],
        "audio_cache_misses": audio_stats["cache_misses"],
    }


def _mark_item_context(items: list[Item], as_of: datetime, radar_config: dict[str, Any]) -> None:
    timezone_name = radar_config.get("timezone", "UTC")
    local_date = as_of.astimezone(ZoneInfo(timezone_name)).date()
    recent_urls = _recent_recommended_urls(
        Path(radar_config["output_dir"]), local_date, int(radar_config.get("recommendation_cooldown_days", 3))
    )
    for item in items:
        first_seen = item.metadata.get("first_seen_at")
        if first_seen:
            try:
                seen_at = datetime.fromisoformat(str(first_seen).replace("Z", "+00:00"))
                item.metadata["is_new_today"] = seen_at.astimezone(ZoneInfo(timezone_name)).date() == local_date
            except ValueError:
                item.metadata["is_new_today"] = False
        else:
            item.metadata["is_new_today"] = False
        item.metadata["recently_recommended"] = item.url in recent_urls


def _recent_recommended_urls(output_dir: Path, current_date, days: int) -> set[str]:
    urls: set[str] = set()
    for offset in range(1, days + 1):
        date = current_date - timedelta(days=offset)
        json_path = output_dir / f"{date.isoformat()}.json"
        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                for trend in payload.get("trends", []):
                    urls.update(item.get("url", "") for item in trend.get("must_reads", []))
            except (json.JSONDecodeError, OSError):
                pass
        markdown_path = output_dir / f"{date.isoformat()}.md"
        if markdown_path.exists():
            urls.update(
                re.findall(r"^- \[[^]]+\]\(([^)]+)\)", markdown_path.read_text(encoding="utf-8"), re.MULTILINE)
            )
    urls.discard("")
    return urls


def _select_unique_labels(trends, limit: int):
    selected = []
    labels: set[str] = set()
    for trend in trends:
        normalized = re.sub(r"\W+", " ", trend.label.casefold()).strip()
        if normalized in labels:
            continue
        labels.add(normalized)
        selected.append(trend)
        if len(selected) >= limit:
            break
    return selected
