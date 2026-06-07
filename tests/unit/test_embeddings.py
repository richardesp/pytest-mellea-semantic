import math

import pytest

from pytest_mellea_semantic import EmbeddingEncoder


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.vectors = {
            "redis": [3.0, 4.0],
            "Redis": [3.0, 4.0],
            "redis ": [3.0, 4.0],
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


def test_encoder_uses_exact_text_cache_keys() -> None:
    backend = FakeBackend()
    encoder = EmbeddingEncoder(backend=backend)

    encoder.encode("redis")
    encoder.encode("Redis")
    encoder.encode("redis ")

    assert backend.calls == ["redis", "Redis", "redis "]
    assert encoder.cache_size == 3


def test_encoder_evicts_least_recently_used_entry() -> None:
    backend = FakeBackend()
    encoder = EmbeddingEncoder(backend=backend, max_cache_size=2)

    encoder.encode("redis")
    encoder.encode("cache")
    encoder.encode("redis")
    encoder.encode("orthogonal")
    encoder.encode("cache")

    assert backend.calls == ["redis", "cache", "orthogonal", "cache"]
    assert encoder.cache_size == 2


def test_encoder_zero_capacity_disables_caching() -> None:
    backend = FakeBackend()
    encoder = EmbeddingEncoder(backend=backend, max_cache_size=0)

    encoder.encode("redis")
    encoder.encode("redis")

    assert backend.calls == ["redis", "redis"]
    assert encoder.cache_size == 0


def test_encoder_clear_cache_forces_reembedding() -> None:
    backend = FakeBackend()
    encoder = EmbeddingEncoder(backend=backend)

    encoder.encode("redis")
    encoder.clear_cache()
    encoder.encode("redis")

    assert backend.calls == ["redis", "redis"]
    assert encoder.cache_size == 1


def test_encoder_rejects_negative_cache_capacity() -> None:
    with pytest.raises(
        ValueError,
        match="max_cache_size must be greater than or equal to zero",
    ):
        EmbeddingEncoder(backend=FakeBackend(), max_cache_size=-1)


def test_similarity_uses_cosine_dot_product() -> None:
    encoder = EmbeddingEncoder(backend=FakeBackend())

    assert math.isclose(encoder.similarity("redis", "cache"), 1.0)
    assert math.isclose(
        encoder.similarity("redis", "orthogonal"),
        0.0,
        abs_tol=1e-12,
    )
