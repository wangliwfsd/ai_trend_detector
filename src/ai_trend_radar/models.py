from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class Item:
    uid: str
    source: str
    kind: str
    title: str
    url: str
    published_at: datetime
    summary: str = ""
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["published_at"] = self.published_at.astimezone(timezone.utc).isoformat()
        return value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Item":
        value = dict(data)
        value["published_at"] = datetime.fromisoformat(value["published_at"].replace("Z", "+00:00"))
        return cls(**value)


@dataclass(slots=True)
class Trend:
    cluster_id: int
    label: str
    score: float
    velocity: float
    count_7d: int
    count_30d: int
    source_count: int
    items: list[Item]
    new_count: int = 0
    summary: str = ""
    why_it_matters: str = ""


@dataclass(slots=True)
class SpeechScript:
    title: str
    content: str
    estimated_minutes: int = 15
    provider: str = "heuristic"
