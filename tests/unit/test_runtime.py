import pytest

import pytest_mellea._runtime as runtime
from pytest_mellea._constants import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_ENCODER_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_THRESHOLD,
)
from pytest_mellea._runtime import (
    configure,
    get_config,
    get_encoder,
    get_judge_session,
    reset_runtime,
)


def setup_function() -> None:
    reset_runtime()


def test_runtime_defaults() -> None:
    config = get_config()

    assert DEFAULT_THRESHOLD == 0.65
    assert DEFAULT_ENCODER_MODEL == "granite-embedding:278m"
    assert DEFAULT_JUDGE_MODEL == "granite4.1:3b"
    assert config.threshold == DEFAULT_THRESHOLD
    assert config.encoder_model == DEFAULT_ENCODER_MODEL
    assert config.cache_size == DEFAULT_CACHE_SIZE
    assert config.judge_model == DEFAULT_JUDGE_MODEL


def test_configure_updates_values() -> None:
    config = configure(
        threshold=0.82,
        encoder_model="custom-embed",
        cache_size=256,
    )

    assert config.threshold == 0.82
    assert config.encoder_model == "custom-embed"
    assert config.cache_size == 256


def test_judge_backend_is_passed_to_mellea(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mellea

    received: dict[str, object] = {}
    expected_session = object()

    def fake_start_session(**kwargs: object) -> object:
        received.update(kwargs)
        return expected_session

    monkeypatch.setattr(mellea, "start_session", fake_start_session)
    configure(judge_backend="future-backend")

    assert get_judge_session() is expected_session
    assert received["backend_name"] == "future-backend"


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
