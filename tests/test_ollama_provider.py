import httpx
import numpy as np

from ai_trend_radar.providers import OllamaEmbeddingProvider


def test_ollama_provider_batches_and_normalizes():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        calls.append(payload)
        return httpx.Response(
            200,
            json={"embeddings": [[3.0, 4.0] for _ in payload["input"]]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OllamaEmbeddingProvider(
        model="test-embed",
        dimensions=2,
        batch_size=2,
        client=client,
    )
    vectors = provider.embed(["a", "b", "c"])
    assert len(calls) == 2
    assert calls[0]["dimensions"] == 2
    np.testing.assert_allclose(vectors, [[0.6, 0.8]] * 3)

