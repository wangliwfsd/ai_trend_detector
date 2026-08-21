import json
from datetime import datetime, timezone
from types import SimpleNamespace

from ai_trend_radar.models import Item, Trend
from ai_trend_radar.providers import GeminiSpeechWriter, HeuristicSpeechWriter
from ai_trend_radar.gemini_utils import QuotaAwareModelPool


class FakeModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        script = "太短" if self.calls == 1 else "详" * 3600
        return SimpleNamespace(text=json.dumps({"title": "今日 AI 趋势", "script": script}))


class InvalidJSONThenValidModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(text='{"title":"今日 AI 趋势","script":"未结束')
        return SimpleNamespace(
            text=json.dumps({"title": "今日 AI 趋势", "script": "详" * 3600})
        )


class EnglishModels:
    def __init__(self):
        self.prompt = ""

    def generate_content(self, **kwargs):
        self.prompt = kwargs["contents"]
        return SimpleNamespace(
            text=json.dumps(
                {"title": "AI Trend Radar", "script": "word " * 2250}
            )
        )


def test_gemini_speech_writer_retries_when_first_draft_is_too_short():
    item = Item(
        uid="paper:1",
        source="arXiv",
        kind="paper",
        title="Efficient Serving",
        url="https://example.com/paper",
        published_at=datetime.now(timezone.utc),
        summary="A grounded abstract.",
        metadata={
            "method_explanation": {
                "purpose": "Reduce serving cost.",
                "approach": "Reuse cached states.",
                "difference": "Avoid recomputation.",
            }
        },
    )
    trend = Trend(1, "LLM Serving", 3.0, 0.5, 2, 5, 2, [item])
    writer = GeminiSpeechWriter.__new__(GeminiSpeechWriter)
    models = FakeModels()
    writer.client = SimpleNamespace(models=models)
    writer.types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
    writer.model = "fake-gemini"
    writer.pool = QuotaAwareModelPool(["fake-gemini"])

    result = writer.write([trend], "zh-CN", 15, "2026-08-20")

    assert models.calls == 2
    assert len(result.content) == 3600
    assert result.provider == "gemini"


def test_gemini_speech_writer_retries_invalid_json():
    item = Item(
        uid="paper:2",
        source="arXiv",
        kind="paper",
        title="Reliable Agents",
        url="https://example.com/agent",
        published_at=datetime.now(timezone.utc),
        summary="A grounded abstract.",
    )
    trend = Trend(2, "Agent Infrastructure", 2.0, 0.2, 2, 4, 1, [item])
    writer = GeminiSpeechWriter.__new__(GeminiSpeechWriter)
    models = InvalidJSONThenValidModels()
    writer.client = SimpleNamespace(models=models)
    writer.types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
    writer.model = "fake-gemini"
    writer.pool = QuotaAwareModelPool(["fake-gemini"])

    result = writer.write([trend], "zh-CN", 15, "2026-08-20")

    assert models.calls == 2
    assert len(result.content) == 3600


def test_gemini_speech_writer_uses_word_target_for_english():
    trend = Trend(2, "Agent Infrastructure", 2.0, 0.2, 2, 4, 1, [])
    writer = GeminiSpeechWriter.__new__(GeminiSpeechWriter)
    models = EnglishModels()
    writer.client = SimpleNamespace(models=models)
    writer.types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
    writer.model = "fake-gemini"
    writer.pool = QuotaAwareModelPool(["fake-gemini"])

    result = writer.write([trend], "en-US", 15, "2026-08-21")

    assert len(result.content.split()) == 2250
    assert "2250 English words" in models.prompt


def test_heuristic_speech_uses_deep_fields_without_repeated_generic_advice():
    item = Item(
        uid="paper:3",
        source="arXiv",
        kind="paper",
        title="Paged KV Scheduling",
        url="https://arxiv.org/abs/2608.12345",
        published_at=datetime.now(timezone.utc),
        metadata={
            "method_explanation": {
                "purpose": "降低突发流量下的尾延迟。",
                "approach": "按页调度 KV block。",
                "difference": "将驱逐粒度从请求改为页。",
                "evidence": "在三种 serving trace 上对比连续 batching，并做了页大小消融。",
                "limitations": "实验只覆盖单机 GPU。",
                "applicability": "适用于存在 KV 碎片的在线解码负载。",
                "expert_takeaway": "可复用部分是页级调度器，而非模型结构。",
            }
        },
    )
    trend = Trend(3, "KV cache scheduling", 3.0, 0.5, 2, 5, 2, [item])

    result = HeuristicSpeechWriter().write([trend], "zh-CN", 15, "2026-08-20")

    assert "实验依据" in result.content
    assert "单机 GPU" in result.content
    assert "阅读时建议重点检查" not in result.content
