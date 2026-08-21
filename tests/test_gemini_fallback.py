import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ai_trend_radar.gemini_utils import QuotaAwareModelPool, reset_exhausted_models


def setup_function():
    reset_exhausted_models()


def test_model_pool_switches_only_after_quota_exhaustion_and_remembers_it():
    calls = []

    def operation(model):
        calls.append(model)
        if model == "strong":
            raise RuntimeError("429 RESOURCE_EXHAUSTED: daily quota exceeded")
        return "ok"

    first = QuotaAwareModelPool(["strong", "fallback"])
    assert first.call(operation) == "ok"
    assert first.last_model == "fallback"

    second = QuotaAwareModelPool(["strong", "fallback"])
    assert second.call(operation) == "ok"
    assert calls == ["strong", "fallback", "fallback"]


def test_model_pool_does_not_hide_non_quota_errors():
    pool = QuotaAwareModelPool(["strong", "fallback"])
    calls = []

    with pytest.raises(RuntimeError, match="invalid request"):
        pool.call(lambda model: calls.append(model) or (_ for _ in ()).throw(RuntimeError("400 invalid request")))

    assert calls == ["strong"]


def test_model_pool_retries_503_then_switches_and_skips_unavailable_model():
    calls = []

    def operation(model):
        calls.append(model)
        if model == "busy":
            raise RuntimeError("503 UNAVAILABLE: model is experiencing high demand")
        return "ok"

    first = QuotaAwareModelPool(["busy", "backup"], transient_retries=1)
    assert first.call(operation) == "ok"
    assert calls == ["busy", "busy", "backup"]

    second = QuotaAwareModelPool(["busy", "backup"], transient_retries=1)
    assert second.call(operation) == "ok"
    assert calls[-1] == "backup"


def test_model_pool_shares_per_model_concurrency_limit_across_instances():
    active = 0
    maximum = 0
    lock = threading.Lock()

    def operation(model):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return model

    pools = [
        QuotaAwareModelPool(["shared"], max_in_flight_per_model=2)
        for _ in range(6)
    ]
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(lambda pool: pool.call(operation), pools))

    assert results == ["shared"] * 6
    assert maximum == 2


def test_model_pool_reports_each_retry_and_model_switch(monkeypatch):
    messages = []
    calls = []

    def operation(model):
        calls.append(model)
        if model == "busy":
            raise RuntimeError("503 UNAVAILABLE: high demand")
        return "ok"

    monkeypatch.setattr("ai_trend_radar.gemini_utils.time.sleep", lambda _: None)
    pool = QuotaAwareModelPool(["busy", "backup"], transient_retries=1)

    assert pool.call(operation, progress=messages.append) == "ok"
    assert calls == ["busy", "busy", "backup"]
    assert any("第 1/2 次请求失败" in message and "1 秒后重试" in message for message in messages)
    assert any("连续 2 次失败" in message and "切换下一模型" in message for message in messages)
