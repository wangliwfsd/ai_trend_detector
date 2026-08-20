from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from .models import Item, Trend
from .providers import EmbeddingProvider

TOPICS: dict[str, tuple[str, ...]] = {
    "Speculative / Parallel Decoding": ("speculative decoding", "parallel decoding", "draft model", "multi-token"),
    "LLM Serving & Inference": ("serving", "inference", "vllm", "sglang", "throughput", "latency"),
    "Quantization & Compression": ("quantization", "quantized", "int4", "fp8", "compression", "low-bit"),
    "KV Cache & Long Context": ("kv cache", "long context", "context window", "attention cache", "paged attention"),
    "GPU Kernels": ("gpu kernel", "cuda", "triton", "flashattention", "flash attention"),
    "Distributed AI Systems": ("distributed inference", "distributed training", "tensor parallel", "pipeline parallel"),
    "Mixture of Experts": ("mixture of experts", "moe", "expert routing", "sparse expert"),
    "Agent Infrastructure": ("agent", "tool use", "computer use", "multi-agent", "agentic"),
    "Reasoning & RL": ("reasoning", "reinforcement learning", "rlhf", "reward model", "test-time compute"),
    "Multimodal Models": ("multimodal", "vision-language", "vlm", "image generation", "video generation"),
}


def detect_trends(
    items: list[Item],
    as_of: datetime,
    embedding_provider: EmbeddingProvider,
    config: dict[str, Any],
) -> list[Trend]:
    if not items:
        return []
    vectors = embedding_provider.embed([item.text() for item in items])
    cluster_config = config.get("clustering", {})
    labels = DBSCAN(
        eps=float(cluster_config.get("eps", 0.32)),
        min_samples=int(cluster_config.get("min_samples", 2)),
        metric="cosine",
    ).fit_predict(vectors)

    groups: dict[int, list[Item]] = defaultdict(list)
    noise: list[Item] = []
    for item, label in zip(items, labels, strict=True):
        (noise if int(label) == -1 else groups[int(label)]).append(item)

    # Keyword grouping rescues semantically useful sparse topics from DBSCAN noise.
    next_id = max(groups.keys(), default=-1) + 1
    keyword_groups: dict[str, list[Item]] = defaultdict(list)
    for item in noise:
        topic = infer_label([item])
        if topic != "Emerging AI Research":
            keyword_groups[topic].append(item)
    for topic, topic_items in keyword_groups.items():
        if len(topic_items) >= int(cluster_config.get("min_samples", 2)):
            groups[next_id] = topic_items
            next_id += 1

    preferences = {key.lower(): float(weight) for key, weight in config.get("preferences", {}).get("keywords", {}).items()}
    trends = [_score_cluster(cluster_id, cluster_items, as_of, preferences) for cluster_id, cluster_items in groups.items()]
    return sorted(trends, key=lambda trend: trend.score, reverse=True)


def _score_cluster(cluster_id: int, items: list[Item], as_of: datetime, preferences: dict[str, float]) -> Trend:
    seven_days_ago = as_of - timedelta(days=7)
    count_7d = sum(item.published_at >= seven_days_ago for item in items)
    count_30d = len(items)
    prior_count = max(0, count_30d - count_7d)
    prior_weekly_rate = prior_count * 7 / 23
    velocity = (count_7d + 1) / (prior_weekly_rate + 1) - 1
    source_count = len({item.source for item in items})
    new_count = sum(bool(item.metadata.get("is_new_today")) for item in items)
    combined = " ".join(item.text().lower() for item in items)
    preference_score = sum(weight for keyword, weight in preferences.items() if keyword in combined)
    attention = sum(math.log1p(sum(item.metrics.values())) for item in items)
    score = (
        1.8 * math.log1p(count_7d)
        + 1.2 * math.log1p(count_30d)
        + 1.4 * max(-0.5, min(3.0, velocity))
        + 0.8 * math.log1p(source_count)
        + 0.35 * preference_score
        + 0.12 * attention
        + 1.0 * math.log1p(new_count)
    )
    ranked_items = sorted(items, key=lambda item: _item_quality(item, as_of), reverse=True)
    return Trend(
        cluster_id=cluster_id,
        label=infer_label(items),
        score=score,
        velocity=velocity,
        count_7d=count_7d,
        count_30d=count_30d,
        source_count=source_count,
        items=ranked_items,
        new_count=new_count,
    )


def infer_label(items: list[Item]) -> str:
    text = " ".join(item.text().lower() for item in items)
    scores = {label: sum(text.count(term) for term in terms) for label, terms in TOPICS.items()}
    label, count = max(scores.items(), key=lambda pair: pair[1])
    if count:
        return label
    words = re.findall(r"[a-z][a-z0-9-]{3,}", text)
    stop = {"with", "from", "that", "this", "using", "models", "model", "based", "learning", "towards"}
    common = [word for word, _ in Counter(word for word in words if word not in stop).most_common(3)]
    return " / ".join(word.title() for word in common) if common else "Emerging AI Research"


def _item_quality(item: Item, as_of: datetime) -> float:
    age_days = max(0.0, (as_of - item.published_at).total_seconds() / 86400)
    source_weight = {"Hugging Face Papers": 1.4, "GitHub Releases": 1.3, "GitHub Trending": 1.2}.get(item.source, 1.0)
    metric_score = math.log1p(sum(item.metrics.values()))
    cross_signal = len(item.metadata.get("signals", []))
    novelty_bonus = 5.0 if item.metadata.get("is_new_today") else 0.0
    repeat_penalty = 8.0 if item.metadata.get("recently_recommended") else 0.0
    return (
        source_weight
        + metric_score
        + 0.5 * cross_signal
        + novelty_bonus
        - repeat_penalty
        - 0.03 * age_days
    )
