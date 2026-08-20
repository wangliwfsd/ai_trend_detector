from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx


@dataclass(slots=True)
class PCMChunk:
    data: bytes
    sample_rate: int = 24_000
    channels: int = 1
    sample_width: int = 2


class TTSProvider(ABC):
    @property
    @abstractmethod
    def namespace(self) -> str: ...

    @abstractmethod
    def synthesize(self, text: str, style: str) -> PCMChunk: ...


class GeminiTTSProvider(TTSProvider):
    def __init__(self, model: str, voice: str, retries: int = 3):
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is required for Gemini TTS")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install the Gemini extra: pip install -e '.[gemini]'") from exc
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        self.voice = voice
        self.retries = max(1, retries)

    @property
    def namespace(self) -> str:
        return f"gemini:{self.model}:{self.voice}:pcm24k-v1"

    def synthesize(self, text: str, style: str) -> PCMChunk:
        prompt = f"""Synthesize a single-speaker Mandarin Chinese podcast narration.
Audio profile: knowledgeable and approachable technology analyst.
Director notes: {style}
Read only the text between TRANSCRIPT_START and TRANSCRIPT_END. Do not read the labels,
director notes, or any formatting instructions aloud.

TRANSCRIPT_START
{text}
TRANSCRIPT_END"""
        last_error: Exception | None = None
        attempts_used = 0
        for attempt in range(self.retries):
            attempts_used = attempt + 1
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Interactions usage is experimental.*",
                        category=UserWarning,
                    )
                    interaction = self.client.interactions.create(
                        model=self.model,
                        input=prompt,
                        response_format={"type": "audio"},
                        generation_config={"speech_config": [{"voice": self.voice}]},
                    )
                audio = interaction.output_audio
                if not audio or not getattr(audio, "data", None):
                    raise RuntimeError("Gemini TTS returned no audio data")
                value = audio.data
                pcm = base64.b64decode(value) if isinstance(value, str) else bytes(value)
                if not pcm:
                    raise RuntimeError("Gemini TTS returned empty audio data")
                return PCMChunk(pcm)
            except Exception as exc:
                last_error = exc
                if not _retryable_tts_error(exc):
                    break
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Gemini TTS failed after {attempts_used} attempt(s): {last_error}")


class LocalHTTPProvider(TTSProvider):
    """OpenAI-compatible local TTS server using POST /v1/audio/speech."""

    def __init__(
        self,
        base_url: str,
        model: str,
        voice: str,
        speed: float = 1.0,
        timeout_seconds: float = 300,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.voice = voice
        self.speed = speed
        self.client = client or httpx.Client(timeout=timeout_seconds)

    @property
    def namespace(self) -> str:
        return f"local-http:{self.base_url}:{self.model}:{self.voice}:{self.speed}:wav-v1"

    def synthesize(self, text: str, style: str) -> PCMChunk:
        response = self.client.post(
            f"{self.base_url}/v1/audio/speech",
            json={
                "model": self.model,
                "voice": self.voice,
                "input": text,
                "response_format": "wav",
                "speed": self.speed,
            },
        )
        response.raise_for_status()
        return pcm_from_wav(response.content)


def make_tts_provider(config: dict[str, Any]) -> TTSProvider:
    section = config.get("audio", {})
    provider = section.get("provider", "gemini")
    if provider == "gemini":
        return GeminiTTSProvider(
            model=section.get("model", "gemini-3.1-flash-tts-preview"),
            voice=section.get("voice", "Charon"),
            retries=int(section.get("retries", 3)),
        )
    if provider in {"local", "local_http", "openai_compatible"}:
        return LocalHTTPProvider(
            base_url=section.get("base_url", "http://127.0.0.1:8880"),
            model=section.get("model", "tts-1"),
            voice=section.get("voice", "default"),
            speed=float(section.get("speed", 1.0)),
            timeout_seconds=float(section.get("timeout_seconds", 300)),
        )
    raise ValueError(f"Unsupported audio provider: {provider}")


def _retryable_tts_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        return status in {408, 409, 429} or status >= 500
    match = re.search(r"(?:Error code:|code['\"]?:)\s*(\d{3})", str(exc), re.IGNORECASE)
    if match:
        value = int(match.group(1))
        return value in {408, 409, 429} or value >= 500
    return True


def split_for_tts(text: str, max_chars: int = 700) -> list[str]:
    max_chars = max(100, max_chars)
    paragraphs = [value.strip() for value in re.split(r"\n\s*\n", text) if value.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        units.extend(
            value.strip()
            for value in re.split(r"(?<=[。！？!?；;])", paragraph)
            if value.strip()
        )

    chunks: list[str] = []
    current = ""
    for unit in units:
        for piece in _split_oversized(unit, max_chars):
            candidate = f"{current}{piece}" if not current else f"{current}\n{piece}"
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_oversized(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        cut = max(window.rfind(mark) for mark in ("，", ",", "、", "：", ":", " "))
        if cut < max_chars // 2:
            cut = max_chars
        else:
            cut += 1
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def synthesize_episode(
    script: str,
    output_path: Path,
    provider: TTSProvider,
    cache_dir: Path,
    style: str,
    chunk_chars: int = 700,
    pause_ms: int = 450,
    bitrate: str = "128k",
    cache_days: int = 14,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    notify = progress or (lambda _: None)
    chunks = split_for_tts(script, chunk_chars)
    if not chunks:
        raise RuntimeError("The speech script is empty")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _prune_cache(cache_dir, cache_days)
    wav_paths: list[Path] = []
    hits = 0
    misses = 0
    for index, text in enumerate(chunks, 1):
        key = _audio_cache_key(provider.namespace, style, text)
        path = cache_dir / f"{key}.wav"
        if _valid_wav(path):
            hits += 1
            notify(f"音频分段 {index}/{len(chunks)}：缓存命中")
        else:
            misses += 1
            notify(f"音频分段 {index}/{len(chunks)}：正在合成（{len(text)} 字符）")
            pcm = provider.synthesize(text, style)
            _write_wav(path, pcm)
        wav_paths.append(path)
    notify(f"音频缓存：{hits} 命中，{misses} 条新生成")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _join_to_mp3(wav_paths, output_path, pause_ms=pause_ms, bitrate=bitrate)
    shutil.copyfile(output_path, output_path.with_name("latest.mp3"))
    return {"chunks": len(chunks), "cache_hits": hits, "cache_misses": misses}


def _prune_cache(cache_dir: Path, max_age_days: int) -> None:
    if max_age_days <= 0:
        return
    cutoff = time.time() - max_age_days * 86_400
    for path in cache_dir.glob("*.wav"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def _audio_cache_key(namespace: str, style: str, text: str) -> str:
    return hashlib.sha256(f"{namespace}\0{style}\0{text}".encode("utf-8")).hexdigest()


def _write_wav(path: Path, pcm: PCMChunk) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.wav")
    with wave.open(str(temporary), "wb") as output:
        output.setnchannels(pcm.channels)
        output.setsampwidth(pcm.sample_width)
        output.setframerate(pcm.sample_rate)
        output.writeframes(pcm.data)
    temporary.replace(path)


def pcm_from_wav(value: bytes) -> PCMChunk:
    try:
        with wave.open(io.BytesIO(value), "rb") as source:
            return PCMChunk(
                data=source.readframes(source.getnframes()),
                sample_rate=source.getframerate(),
                channels=source.getnchannels(),
                sample_width=source.getsampwidth(),
            )
    except (wave.Error, EOFError) as exc:
        raise RuntimeError("Local TTS server did not return a valid WAV file") from exc


def _valid_wav(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb") as source:
            return source.getnframes() > 0 and source.getframerate() > 0
    except (wave.Error, EOFError):
        return False


def _join_to_mp3(wav_paths: list[Path], output_path: Path, pause_ms: int, bitrate: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to create MP3 output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ai-trend-audio-", dir=output_path.parent
    ) as temporary_dir:
        joined_wav = Path(temporary_dir) / "joined.wav"
        temporary_mp3 = Path(temporary_dir) / "episode.mp3"
        _join_wav(wav_paths, joined_wav, pause_ms)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(joined_wav),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(temporary_mp3),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg MP3 conversion failed: {result.stderr.strip()}")
        temporary_mp3.replace(output_path)


def _join_wav(wav_paths: list[Path], output_path: Path, pause_ms: int) -> None:
    params: tuple[int, int, int] | None = None
    frames: list[bytes] = []
    for path in wav_paths:
        with wave.open(str(path), "rb") as source:
            current = (source.getnchannels(), source.getsampwidth(), source.getframerate())
            if params is None:
                params = current
            elif current != params:
                raise RuntimeError("TTS chunks use inconsistent WAV formats")
            frames.append(source.readframes(source.getnframes()))
    if params is None:
        raise RuntimeError("No TTS chunks were generated")
    channels, sample_width, sample_rate = params
    silence = b"\0" * int(sample_rate * max(0, pause_ms) / 1000) * channels * sample_width
    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(sample_rate)
        output.writeframes(silence.join(frames))
