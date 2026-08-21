import json
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from ai_trend_radar.deep_reading import GeminiDeepReader, enrich_must_reads, extract_arxiv_id
from ai_trend_radar.models import Item, Trend
from ai_trend_radar.gemini_utils import QuotaAwareModelPool


def make_item(**overrides):
    values = {
        "uid": "hf:2608.12345",
        "source": "Hugging Face Papers",
        "kind": "paper",
        "title": "Efficient Inference",
        "url": "https://huggingface.co/papers/2608.12345",
        "published_at": datetime.now(timezone.utc),
        "summary": "We introduce a scheduler.",
        "metadata": {"paper_id": "2608.12345"},
    }
    values.update(overrides)
    return Item(**values)


def test_extract_arxiv_id_from_huggingface_metadata():
    assert extract_arxiv_id(make_item()) == "2608.12345"


def test_deep_reader_caches_structured_full_paper_analysis(tmp_path):
    result = {
        "purpose": "Reduce decode stalls.",
        "approach": "Schedule KV pages by pressure.",
        "difference": "Uses page-level rather than request-level eviction.",
        "experimental_setup": "Three serving traces on one GPU.",
        "baseline_fairness": "The same model and traces are used for three baselines.",
        "ablations_and_mechanism": "Page size is ablated.",
        "key_evidence": "Tail latency falls on all three traces.",
        "unproven_claims": "Multi-node scaling is not established.",
        "limitations": "Only single-node GPUs are evaluated.",
        "applicability": "Online decoding with fragmented KV memory.",
        "adoption_prerequisites": "A paged KV allocator is required.",
        "replication_checks": "Replay production traces; verify output parity.",
        "verdict": "值得复现：先验证多并发尾延迟。",
    }

    class FakeModels:
        calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            assert kwargs["contents"][0] == (b"%PDF fake", "application/pdf")
            return SimpleNamespace(text=json.dumps(result))

    class FakeHTTP:
        def get(self, url):
            assert url == "https://arxiv.org/pdf/2608.12345"
            return SimpleNamespace(
                content=b"%PDF fake",
                headers={"content-type": "application/pdf"},
                raise_for_status=lambda: None,
            )

    reader = GeminiDeepReader.__new__(GeminiDeepReader)
    models = FakeModels()
    reader.client = SimpleNamespace(models=models)
    reader.types = SimpleNamespace(
        Part=SimpleNamespace(from_bytes=lambda data, mime_type: (data, mime_type)),
        GenerateContentConfig=lambda **kwargs: kwargs,
    )
    reader.model = "fake-gemini"
    reader.pool = QuotaAwareModelPool(["fake-gemini"])
    reader.cache_dir = tmp_path
    reader.max_pdf_bytes = 1_000_000
    reader.http = FakeHTTP()

    first, first_cached = reader.analyze(make_item(), "zh-CN")
    second, second_cached = reader.analyze(make_item(), "zh-CN")

    assert first["source_scope"] == "full_paper"
    assert first == second
    assert first_cached is False
    assert second_cached is True
    assert models.calls == 1


def test_enrich_must_reads_uses_bounded_parallel_workers(monkeypatch, tmp_path):
    active = 0
    maximum = 0
    lock = threading.Lock()

    class FakeReader:
        def __init__(self, **kwargs):
            self.model = "fake"

        def analyze(self, item, language):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return {
                "source_scope": "full_paper",
                "model_used": "fake",
                "purpose": item.title,
            }, False

    monkeypatch.setattr("ai_trend_radar.deep_reading.GeminiDeepReader", FakeReader)
    items = [make_item(uid=f"hf:2608.1234{i}", url=f"https://example.com/{i}") for i in range(4)]
    trend = Trend(1, "Serving", 3.0, 0.5, 4, 4, 1, items)
    config = {
        "llm": {"models": ["fake"]},
        "deep_reading": {
            "enabled": True,
            "provider": "gemini",
            "models": ["fake"],
            "cache_dir": str(tmp_path),
            "max_workers": 2,
            "max_in_flight_per_model": 2,
        },
    }

    hits, misses, warnings = enrich_must_reads([trend], config, "zh-CN", 4)

    assert (hits, misses, warnings) == (0, 4, [])
    assert maximum == 2
    assert all(item.metadata["method_explanation"]["purpose"] == item.title for item in items)


def test_enrich_must_reads_stops_submitting_after_quota_failure(monkeypatch, tmp_path):
    calls = []

    class FakeReader:
        def __init__(self, **kwargs):
            self.model = "fake"

        def analyze(self, item, language):
            calls.append(item.uid)
            if item.uid.endswith("0"):
                raise RuntimeError("429 RESOURCE_EXHAUSTED: daily quota")
            time.sleep(0.08)
            return {"source_scope": "full_paper", "model_used": "fake"}, False

    monkeypatch.setattr("ai_trend_radar.deep_reading.GeminiDeepReader", FakeReader)
    items = [make_item(uid=f"item-{i}", url=f"https://example.com/{i}") for i in range(5)]
    trend = Trend(1, "Serving", 3.0, 0.5, 5, 5, 1, items)
    config = {
        "deep_reading": {
            "enabled": True,
            "provider": "gemini",
            "models": ["fake"],
            "cache_dir": str(tmp_path),
            "max_workers": 2,
        }
    }

    _, _, warnings = enrich_must_reads([trend], config, "zh-CN", 5)

    assert len(calls) == 2
    assert any("停止派发" in warning for warning in warnings)
