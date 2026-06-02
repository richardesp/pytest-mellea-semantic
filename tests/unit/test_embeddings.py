import math

from pytest_mellea_semantic import EmbeddingEncoder


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.vectors = {
            "redis": [3.0, 4.0],
            "cache": [6.0, 8.0],
            "orthogonal": [-4.0, 3.0],
        }

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vectors[text]


def test_encoder_normalizes_vectors() -> None:
    encoder = EmbeddingEncoder(backend=FakeBackend())

    assert encoder.encode("redis") == [0.6, 0.8]


def test_encoder_caches_by_text() -> None:
    backend = FakeBackend()
    encoder = EmbeddingEncoder(backend=backend)

    encoder.encode("redis")
    encoder.encode("redis")

    assert backend.calls == ["redis"]
    assert encoder.cache_size == 1


def test_similarity_uses_cosine_dot_product() -> None:
    encoder = EmbeddingEncoder(backend=FakeBackend())

    assert math.isclose(encoder.similarity("redis", "cache"), 1.0)
    assert math.isclose(encoder.similarity("redis", "orthogonal"), 0.0)
