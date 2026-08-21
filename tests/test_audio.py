import io
import base64
import wave
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

from ai_trend_radar.audio import (
    FallbackTTSProvider,
    LocalHTTPProvider,
    GeminiTTSProvider,
    PCMChunk,
    TTSProvider,
    make_tts_provider,
    resolve_audio_chunk_chars,
    resolve_audio_language,
    resolve_audio_style,
    split_for_tts,
    synthesize_episode,
)


class FakeTTSProvider(TTSProvider):
    def __init__(self):
        self.calls = 0

    @property
    def namespace(self) -> str:
        return "fake:voice:v1"

    def synthesize(self, text: str, style: str) -> PCMChunk:
        self.calls += 1
        return PCMChunk(b"\0\0" * 240)


def wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24_000)
        target.writeframes(b"\0\0" * 240)
    return output.getvalue()


def test_split_for_tts_keeps_content_and_respects_limit():
    text = "第一段介绍。" * 30 + "\n\n" + "第二段结论！" * 30
    chunks = split_for_tts(text, max_chars=120)
    assert len(chunks) > 2
    assert all(len(value) <= 120 for value in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_split_for_tts_handles_english_sentences_without_tiny_tail():
    text = " ".join(
        f"Sentence {index} explains a technical result with enough supporting context."
        for index in range(45)
    )

    chunks = split_for_tts(text, max_chars=360)

    assert all(len(value) <= 360 for value in chunks)
    assert len(chunks[-1]) >= 72
    assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace(" ", "")


def test_audio_chunks_are_cached(tmp_path: Path, monkeypatch):
    provider = FakeTTSProvider()

    def fake_join(paths, output_path, pause_ms, bitrate):
        assert all(path.exists() for path in paths)
        output_path.write_bytes(b"fake-mp3")

    monkeypatch.setattr("ai_trend_radar.audio._join_to_mp3", fake_join)
    kwargs = {
        "script": "第一段。第二段。第三段。" * 20,
        "output_path": tmp_path / "reports" / "2026-08-20.mp3",
        "provider": provider,
        "cache_dir": tmp_path / "cache",
        "style": "平稳",
        "chunk_chars": 100,
    }
    first = synthesize_episode(**kwargs)
    first_calls = provider.calls
    second = synthesize_episode(**kwargs)

    assert first["cache_misses"] == first_calls
    assert second["cache_hits"] == first_calls
    assert second["cache_misses"] == 0
    assert provider.calls == first_calls
    assert (tmp_path / "reports" / "latest.mp3").exists()


def test_parallel_safe_local_tts_synthesizes_chunks_concurrently(tmp_path: Path, monkeypatch):
    class ParallelProvider(FakeTTSProvider):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.maximum = 0
            self.lock = threading.Lock()

        @property
        def max_parallel_workers(self) -> int:
            return 3

        def synthesize(self, text: str, style: str) -> PCMChunk:
            with self.lock:
                self.calls += 1
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return PCMChunk(b"\0\0" * 240)

    provider = ParallelProvider()

    def fake_join(paths, output_path, pause_ms, bitrate):
        assert len(paths) >= 3
        assert all(path.exists() for path in paths)
        output_path.write_bytes(b"fake-mp3")

    monkeypatch.setattr("ai_trend_radar.audio._join_to_mp3", fake_join)
    script = "\n\n".join(f"第{i}段" + "内容" * 45 + "。" for i in range(6))

    stats = synthesize_episode(
        script,
        tmp_path / "report.mp3",
        provider,
        tmp_path / "cache",
        style="自然",
        chunk_chars=100,
        max_workers=3,
    )

    assert stats["cache_misses"] >= 3
    assert provider.maximum == 3


def test_local_http_provider_uses_openai_compatible_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/speech"
        assert b'"response_format":"wav"' in request.content
        return httpx.Response(200, content=wav_bytes())

    provider = LocalHTTPProvider(
        "http://tts.local",
        "tts-1",
        "default",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = provider.synthesize("你好", "自然")

    assert result.sample_rate == 24_000
    assert result.channels == 1
    assert result.data


def test_gemini_provider_decodes_pcm_audio():
    pcm = b"\x01\x02" * 100

    class FakeInteractions:
        def create(self, **kwargs):
            assert kwargs["response_format"] == {"type": "audio"}
            assert kwargs["generation_config"]["speech_config"] == [{"voice": "Charon"}]
            assert "single-speaker English" in kwargs["input"]
            return SimpleNamespace(
                output_audio=SimpleNamespace(data=base64.b64encode(pcm).decode("ascii"))
            )

    provider = GeminiTTSProvider.__new__(GeminiTTSProvider)
    provider.client = SimpleNamespace(interactions=FakeInteractions())
    provider.model = "fake-tts"
    provider.voice = "Charon"
    provider.retries = 1
    provider.language = "en-US"

    result = provider.synthesize("你好", "平稳")

    assert result.data == pcm
    assert result.sample_rate == 24_000


def test_audio_language_selects_local_voice_and_style():
    config = {
        "radar": {"report_language": "zh-CN"},
        "audio": {
            "language": "en-US",
            "provider": "local_http",
            "base_url": "http://tts.local",
            "model": "kokoro",
            "voices": {"zh-CN": "zf_xiaoxiao", "en-US": "af_heart"},
            "styles": {"zh-CN": "中文风格", "en-US": "English style"},
        },
    }

    provider = make_tts_provider(config)

    assert isinstance(provider, LocalHTTPProvider)
    assert provider.language == "en-US"
    assert provider.voice == "af_heart"
    assert resolve_audio_language(config) == "en-US"
    assert resolve_audio_style(config) == "English style"
    config["audio"]["chunk_chars_by_language"] = {"zh-CN": 700, "en-US": 1800}
    assert resolve_audio_chunk_chars(config) == 1800


def test_gemini_provider_does_not_retry_non_transient_400():
    class FakeInteractions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("Error code: 400 - invalid request")

    interactions = FakeInteractions()
    provider = GeminiTTSProvider.__new__(GeminiTTSProvider)
    provider.client = SimpleNamespace(interactions=interactions)
    provider.model = "fake-tts"
    provider.voice = "Charon"
    provider.retries = 3

    try:
        provider.synthesize("你好", "平稳")
    except RuntimeError as exc:
        assert "after 1 attempt(s)" in str(exc)
    else:
        raise AssertionError("Expected Gemini TTS to fail")
    assert interactions.calls == 1


def test_tts_fallback_switches_on_quota_and_keeps_selected_provider():
    class QuotaGemini(GeminiTTSProvider):
        def __init__(self):
            self.model = "quota-tts"
            self.voice = "Charon"

        def synthesize(self, text, style):
            raise RuntimeError("429 RESOURCE_EXHAUSTED: daily quota")

    local = FakeTTSProvider()
    provider = FallbackTTSProvider([QuotaGemini(), local])

    result = provider.synthesize("你好", "自然")

    assert result.data
    assert provider.namespace == local.namespace
    assert local.calls == 1
