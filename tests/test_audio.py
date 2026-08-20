import io
import base64
import wave
from pathlib import Path
from types import SimpleNamespace

import httpx

from ai_trend_radar.audio import (
    LocalHTTPProvider,
    GeminiTTSProvider,
    PCMChunk,
    TTSProvider,
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
            return SimpleNamespace(
                output_audio=SimpleNamespace(data=base64.b64encode(pcm).decode("ascii"))
            )

    provider = GeminiTTSProvider.__new__(GeminiTTSProvider)
    provider.client = SimpleNamespace(interactions=FakeInteractions())
    provider.model = "fake-tts"
    provider.voice = "Charon"
    provider.retries = 1

    result = provider.synthesize("你好", "平稳")

    assert result.data == pcm
    assert result.sample_rate == 24_000


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
