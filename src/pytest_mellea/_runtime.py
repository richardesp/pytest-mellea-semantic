"""Runtime configuration and lazy singleton helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pytest_mellea._constants import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_ENCODER_MODEL,
    DEFAULT_JUDGE_BACKEND,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_MODEL_OPTIONS,
    DEFAULT_THRESHOLD,
)
from pytest_mellea._embeddings import EmbeddingEncoder, OllamaEmbeddingBackend


@dataclass(frozen=True)
class SemanticConfig:
    """Runtime defaults for semantic assertions.

    Args:
        threshold: Default cosine similarity threshold for `Content`.
        encoder_model: Ollama embedding model for `Content`.
        cache_size: Maximum embeddings held by the shared encoder. Zero disables
            caching.
        ollama_host: Optional Ollama host for the embedding client.
        judge_backend: Mellea backend name for `Behavior`.
        judge_model: Mellea model id for `Behavior`.

    Raises:
        ValueError: If `cache_size` is negative.
    """

    threshold: float = DEFAULT_THRESHOLD
    encoder_model: str = DEFAULT_ENCODER_MODEL
    cache_size: int = DEFAULT_CACHE_SIZE
    ollama_host: str | None = None
    judge_backend: str = DEFAULT_JUDGE_BACKEND
    judge_model: str = DEFAULT_JUDGE_MODEL

    def __post_init__(self) -> None:
        """Validate runtime configuration values."""
        if self.cache_size < 0:
            raise ValueError("cache_size must be greater than or equal to zero")


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
        or previous.cache_size != _config.cache_size
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
        _encoder = EmbeddingEncoder(
            backend=backend,
            max_cache_size=config.cache_size,
        )
    return _encoder


def get_judge_session() -> Any:
    """Return the lazily-created default Mellea judge session.

    Returns:
        Shared Mellea session for behavior assertions.
    """
    global _judge_session
    if _judge_session is None:
        from mellea import start_session

        config = get_config()
        # Mellea owns backend validation; its closed annotation may lag new backends.
        _judge_session = start_session(
            backend_name=config.judge_backend,  # type: ignore[arg-type]
            model_id=config.judge_model,
            model_options=DEFAULT_JUDGE_MODEL_OPTIONS,
        )
    return _judge_session
