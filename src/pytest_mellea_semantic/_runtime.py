"""Runtime configuration and lazy singleton helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from pytest_mellea_semantic._embeddings import EmbeddingEncoder, OllamaEmbeddingBackend

DEFAULT_THRESHOLD = 0.70
DEFAULT_ENCODER_MODEL = "nomic-embed-text:v1.5"
DEFAULT_JUDGE_BACKEND = "ollama"
DEFAULT_JUDGE_MODEL = "gemma4:e2b"
DEFAULT_JUDGE_MODEL_OPTIONS: dict[str, Any] = {"temperature": 0}
BackendName = Literal["ollama", "hf", "openai", "watsonx", "litellm"]


@dataclass(frozen=True)
class SemanticConfig:
    """Runtime defaults for semantic assertions.

    Args:
        threshold: Default cosine similarity threshold for `Content`.
        encoder_model: Ollama embedding model for `Content`.
        ollama_host: Optional Ollama host for the embedding client.
        judge_backend: Mellea backend name for `Behaviour`.
        judge_model: Mellea model id for `Behaviour`.
    """

    threshold: float = DEFAULT_THRESHOLD
    encoder_model: str = DEFAULT_ENCODER_MODEL
    ollama_host: str | None = None
    judge_backend: str = DEFAULT_JUDGE_BACKEND
    judge_model: str = DEFAULT_JUDGE_MODEL


_config = SemanticConfig()
_encoder: EmbeddingEncoder | None = None
_judge_session: Any | None = None


def get_config() -> SemanticConfig:
    """Return the active runtime configuration.

    Returns:
        Active semantic assertion configuration.
    """
    return _config


def configure(**overrides: Any) -> SemanticConfig:
    """Update runtime configuration and reset affected lazy singletons.

    Args:
        **overrides: Field overrides accepted by `SemanticConfig`.

    Returns:
        Updated configuration.
    """
    global _config, _encoder, _judge_session

    previous = _config
    _config = replace(_config, **overrides)

    if (
        previous.encoder_model != _config.encoder_model
        or previous.ollama_host != _config.ollama_host
    ):
        _encoder = None

    if (
        previous.judge_backend != _config.judge_backend
        or previous.judge_model != _config.judge_model
    ):
        _judge_session = None

    return _config


def reset_runtime() -> None:
    """Reset runtime configuration and lazy singletons to package defaults."""
    global _config, _encoder, _judge_session
    _config = SemanticConfig()
    _encoder = None
    _judge_session = None


def get_encoder() -> EmbeddingEncoder:
    """Return the lazily-created default embedding encoder.

    Returns:
        Shared embedding encoder for the active configuration.
    """
    global _encoder
    if _encoder is None:
        config = get_config()
        backend = OllamaEmbeddingBackend(
            model_id=config.encoder_model, host=config.ollama_host
        )
        _encoder = EmbeddingEncoder(backend=backend)
    return _encoder


def get_judge_session() -> Any:
    """Return the lazily-created default Mellea judge session.

    Returns:
        Shared Mellea session for behaviour assertions.
    """
    global _judge_session
    if _judge_session is None:
        from mellea import start_session

        config = get_config()
        backend_name = cast(BackendName, config.judge_backend)
        _judge_session = start_session(
            backend_name=backend_name,
            model_id=config.judge_model,
            model_options=DEFAULT_JUDGE_MODEL_OPTIONS,
        )
    return _judge_session
