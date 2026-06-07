import pytest

import pytest_mellea_semantic._runtime as runtime
from pytest_mellea_semantic._constants import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_ENCODER_MODEL,
    DEFAULT_THRESHOLD,
)
from pytest_mellea_semantic._runtime import (
    configure,
    get_config,
    get_encoder,
    reset_runtime,
)


def setup_function() -> None:
    reset_runtime()


def test_runtime_defaults() -> None:
    config = get_config()

    assert config.threshold == DEFAULT_THRESHOLD
    assert config.encoder_model == DEFAULT_ENCODER_MODEL
    assert config.cache_size == DEFAULT_CACHE_SIZE


def test_configure_updates_values() -> None:
    config = configure(
        threshold=0.82,
        encoder_model="custom-embed",
        cache_size=256,
    )

    assert config.threshold == 0.82
    assert config.encoder_model == "custom-embed"
    assert config.cache_size == 256


def test_cache_size_change_recreates_shared_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOllamaBackend:
        def __init__(self, model_id: str, host: str | None) -> None:
            pass

        def embed(self, text: str) -> list[float]:
            return [1.0]

    monkeypatch.setattr(runtime, "OllamaEmbeddingBackend", FakeOllamaBackend)
    original = get_encoder()

    configure(cache_size=2)
    configured = get_encoder()
    configured.encode("first")
    configured.encode("second")
    configured.encode("third")

    assert configured is not original
    assert configured.cache_size == 2


def test_runtime_rejects_negative_cache_size() -> None:
    with pytest.raises(
        ValueError,
        match="cache_size must be greater than or equal to zero",
    ):
        configure(cache_size=-1)
