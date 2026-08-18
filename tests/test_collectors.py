from datetime import datetime, timezone

from ai_trend_radar.collectors import deduplicate
from ai_trend_radar.models import Item


def test_deduplicate_merges_arxiv_and_huggingface_signals():
    now = datetime.now(timezone.utc)
    items = [
        Item("arxiv:2608.12345", "arXiv", "paper", "A Useful Paper", "https://arxiv.org/abs/2608.12345", now),
        Item("hf:2608.12345", "Hugging Face Papers", "paper", "A Useful Paper", "https://huggingface.co/papers/2608.12345", now, metrics={"upvotes": 42}),
    ]
    result = deduplicate(items)
    assert len(result) == 1
    assert result[0].metrics["upvotes"] == 42
    assert result[0].metadata["signals"] == ["Hugging Face Papers", "arXiv"]

