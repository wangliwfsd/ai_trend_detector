import json
from datetime import datetime, timezone
from types import SimpleNamespace

from ai_trend_radar.gemini_utils import QuotaAwareModelPool
from ai_trend_radar.models import Item, Trend
from ai_trend_radar.providers import GeminiNarrator


def make_trend(cluster_id: int, title: str) -> Trend:
    item = Item(
        uid=f"paper:{cluster_id}",
        source="arXiv",
        kind="paper",
        title=title,
        url=f"https://example.com/{cluster_id}",
        published_at=datetime.now(timezone.utc),
        summary="A grounded source summary.",
    )
    return Trend(cluster_id, title, 3.0, 0.5, 2, 4, 1, [item])


def row(trend: Trend, *, coherent: bool, redundant: int = -1) -> dict:
    item = trend.items[0]
    return {
        "cluster_id": trend.cluster_id,
        "label": trend.label,
        "summary": "七天窗口出现两个相互印证的系统信号。",
        "why_it_matters": "先在生产 trace 上复现，再决定是否采用。",
        "coherent": coherent,
        "coherence_reason": "围绕同一 serving 瓶颈。" if coherent else "方法和问题不一致。",
        "evidence_basis": "一篇论文和一个 release。",
        "confidence": "medium",
        "counterevidence": "缺少跨硬件复现。",
        "relevant_urls": [item.url] if coherent else [],
        "redundant_with_cluster_id": redundant,
        "rank_adjustment": 0.2,
        "item_explanations": [
            {
                "url": item.url,
                "purpose": "降低延迟。",
                "approach": "拆分调度。",
                "difference": "改变执行粒度。",
            }
        ],
    }


def test_gemini_narrator_rejects_incoherent_and_redundant_clusters():
    primary = make_trend(1, "Serving")
    incoherent = make_trend(2, "Mixed")
    redundant = make_trend(3, "Serving duplicate")
    payload = {
        "trends": [
            row(primary, coherent=True),
            row(incoherent, coherent=False),
            row(redundant, coherent=True, redundant=1),
        ]
    }

    narrator = GeminiNarrator.__new__(GeminiNarrator)
    narrator.client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(text=json.dumps(payload))
        )
    )
    narrator.types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
    narrator.pool = QuotaAwareModelPool(["fake-gemini"])
    narrator.model = "fake-gemini"

    result = narrator.enrich([primary, incoherent, redundant], "zh-CN")

    assert result == [primary]
    assert primary.relevant_urls == [primary.items[0].url]
    assert primary.confidence == "medium"
