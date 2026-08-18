from pathlib import Path

import numpy as np

from ai_trend_radar.providers import CachedEmbeddingProvider, EmbeddingProvider
from ai_trend_radar.storage import Store


class CountingProvider(EmbeddingProvider):
    model = "test-model"
    dimensions = 3
    batch_size = 2

    def __init__(self):
        self.calls = 0
        self.inputs = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.inputs += len(texts)
        return np.asarray([[len(text), 1, 0] for text in texts], dtype=np.float32)


def test_embedding_cache_avoids_recomputing_history(tmp_path: Path):
    store = Store(tmp_path / "radar.db")
    try:
        first_delegate = CountingProvider()
        first = CachedEmbeddingProvider(first_delegate, store)
        expected = first.embed(["alpha", "beta", "gamma"])
        assert first_delegate.calls == 2
        assert first.misses == 3
        assert store.embedding_cache_count() == 3

        second_delegate = CountingProvider()
        second = CachedEmbeddingProvider(second_delegate, store)
        actual = second.embed(["alpha", "beta", "gamma", "delta"])
        assert second_delegate.calls == 1
        assert second_delegate.inputs == 1
        assert second.hits == 3
        assert second.misses == 1
        np.testing.assert_array_equal(actual[:3], expected)
    finally:
        store.close()

