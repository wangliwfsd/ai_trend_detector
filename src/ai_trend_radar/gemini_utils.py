from __future__ import annotations

from collections.abc import Callable, Iterable
import threading
import time
from typing import TypeVar


T = TypeVar("T")
_EXHAUSTED_MODELS: set[str] = set()
_UNAVAILABLE_MODELS: set[str] = set()
_STATE_LOCK = threading.Lock()
_MODEL_LIMITERS: dict[tuple[str, int], threading.BoundedSemaphore] = {}


def model_chain(section: dict, default: Iterable[str]) -> list[str]:
    configured = section.get("models")
    if isinstance(configured, list):
        values = [str(value).strip() for value in configured if str(value).strip()]
        if values:
            return list(dict.fromkeys(values))
    single = str(section.get("model", "")).strip()
    return [single] if single else list(default)


def is_quota_exhausted(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    value = str(exc).casefold()
    return "resource_exhausted" in value or bool(
        "429" in value and ("quota" in value or "rate limit" in value)
    )


def is_transient_server_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        return status in {408, 409, 500, 502, 503, 504}
    value = str(exc).casefold()
    return any(
        marker in value
        for marker in (
            "503 unavailable",
            "500 internal",
            "502 bad gateway",
            "504 gateway",
            "high demand",
            "temporarily unavailable",
        )
    )


class QuotaAwareModelPool:
    """Try stronger models first and permanently skip observed daily-quota failures."""

    def __init__(
        self,
        models: Iterable[str],
        transient_retries: int = 2,
        max_in_flight_per_model: int | None = None,
    ):
        self.models = list(dict.fromkeys(value for value in models if value))
        if not self.models:
            raise ValueError("At least one Gemini model is required")
        self.last_model = self.models[0]
        self.transient_retries = max(0, transient_retries)
        self.max_in_flight_per_model = (
            max(1, int(max_in_flight_per_model))
            if max_in_flight_per_model is not None
            else None
        )

    def call(
        self,
        operation: Callable[[str], T],
        progress: Callable[[str], None] | None = None,
    ) -> T:
        notify = progress or (lambda _: None)
        model_errors: list[str] = []
        for model in self.models:
            if _model_is_blocked(model):
                notify(f"跳过已熔断模型 {model}")
                continue
            for attempt in range(self.transient_retries + 1):
                limiter = self._limiter(model)
                if limiter is not None:
                    limiter.acquire()
                try:
                    # A request that waited behind another worker must re-check the shared
                    # circuit breaker before it consumes quota.
                    if _model_is_blocked(model):
                        break
                    result = operation(model)
                    self.last_model = model
                    return result
                except Exception as exc:
                    if is_quota_exhausted(exc):
                        _mark_model(_EXHAUSTED_MODELS, model)
                        model_errors.append(f"{model} quota: {exc}")
                        notify(
                            f"{model} 第 {attempt + 1}/{self.transient_retries + 1} 次请求遇到配额限制："
                            f"{_short_error(exc)}；切换下一模型"
                        )
                        break
                    if not is_transient_server_error(exc):
                        notify(
                            f"{model} 请求失败且不可重试：{type(exc).__name__}: "
                            f"{_short_error(exc)}"
                        )
                        raise
                    if attempt < self.transient_retries:
                        delay = 2**attempt
                        notify(
                            f"{model} 第 {attempt + 1}/{self.transient_retries + 1} 次请求失败："
                            f"{type(exc).__name__}: {_short_error(exc)}；{delay} 秒后重试"
                        )
                        time.sleep(delay)
                        continue
                    _mark_model(_UNAVAILABLE_MODELS, model)
                    model_errors.append(f"{model} unavailable after retries: {exc}")
                    notify(
                        f"{model} 连续 {self.transient_retries + 1} 次失败，标记本轮不可用；"
                        "切换下一模型"
                    )
                    break
                finally:
                    if limiter is not None:
                        limiter.release()
        detail = "; ".join(model_errors) or "all configured models were already marked unavailable"
        raise RuntimeError(f"No configured Gemini model is currently available: {detail}")

    def _limiter(self, model: str) -> threading.BoundedSemaphore | None:
        limit = self.max_in_flight_per_model
        if limit is None:
            return None
        key = (model, limit)
        with _STATE_LOCK:
            limiter = _MODEL_LIMITERS.get(key)
            if limiter is None:
                limiter = threading.BoundedSemaphore(limit)
                _MODEL_LIMITERS[key] = limiter
            return limiter


def _model_is_blocked(model: str) -> bool:
    with _STATE_LOCK:
        return model in _EXHAUSTED_MODELS or model in _UNAVAILABLE_MODELS


def _mark_model(registry: set[str], model: str) -> None:
    with _STATE_LOCK:
        registry.add(model)


def _short_error(exc: Exception, limit: int = 220) -> str:
    return " ".join(str(exc).split())[:limit]


def reset_exhausted_models() -> None:
    """Test helper; a new CLI process naturally starts with a clean daily-quota registry."""
    with _STATE_LOCK:
        _EXHAUSTED_MODELS.clear()
        _UNAVAILABLE_MODELS.clear()
        _MODEL_LIMITERS.clear()
