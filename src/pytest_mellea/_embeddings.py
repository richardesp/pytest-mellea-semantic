"""Embedding backends and cosine similarity helpers for content assertions.

Normalized vectors are cached by their exact input text. The default cache holds
up to 1024 entries, while a capacity of zero disables caching.
"""

from __future__ import annotations

import abc
from collections import OrderedDict
from typing import Protocol, cast

from pytest_mellea._constants import DEFAULT_CACHE_SIZE, DEFAULT_ENCODER_MODEL
from pytest_mellea._exceptions import SemanticAssertionRuntimeError


class EmbeddingBackend(abc.ABC):
    """Interface for text embedding providers.

    Args:
        No constructor arguments are required by the abstract interface.
    """

    @abc.abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a raw embedding vector for text.

        Args:
            text: Text to embed.

        Returns:
            Raw embedding vector. The vector does not need to be normalized.
        """


class SupportsEmbed(Protocol):
    """Protocol for objects compatible with :class:`EmbeddingEncoder`.

    Args:
        No constructor arguments are required by this protocol.
    """

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for text.

        Args:
            text: Text to embed.

        Returns:
            Raw embedding vector.
        """


class OllamaEmbeddingBackend(EmbeddingBackend):
    """Embedding backend backed by Ollama's `/api/embed` endpoint.

    Args:
        model_id: Ollama embedding model name.
        host: Optional Ollama host. `None` lets the Ollama client use
            `OLLAMA_HOST` or its default host.
    """

    def __init__(
        self, model_id: str = DEFAULT_ENCODER_MODEL, host: str | None = None
    ) -> None:
        """Initialize the Ollama embedding backend."""
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise SemanticAssertionRuntimeError(
                "Content assertions require the `ollama` package. Install the "
                "project dependencies with `uv sync --dev`."
            ) from exc

        self._client = ollama.Client(host=host)
        self._model_id = model_id

    def embed(self, text: str) -> list[float]:
        """Return the raw Ollama embedding vector for text.

        Args:
            text: Text to embed.

        Returns:
            Raw embedding vector from Ollama.

        Raises:
            SemanticAssertionRuntimeError: If Ollama cannot serve the embedding.
        """
        try:
            response = self._client.embed(model=self._model_id, input=text)
            return list(response.embeddings[0])
        except Exception as exc:
            raise SemanticAssertionRuntimeError(
                "Could not compute an Ollama embedding. Ensure Ollama is running "
                f"and the model is available: `ollama pull {self._model_id}`."
            ) from exc


class EmbeddingEncoder:
    """LRU-caching encoder for normalized embeddings and cosine similarity.

    Cache keys preserve the exact input text, including whitespace and case.
    Cache hits refresh entry recency, and the least-recently-used entry is
    discarded when the configured capacity is exceeded.

    Args:
        backend: Embedding backend. Defaults to `OllamaEmbeddingBackend`.
        max_cache_size: Maximum cached embeddings. Defaults to 1024. Set to zero
            to disable caching.

    Raises:
        ValueError: If `max_cache_size` is negative.
    """

    def __init__(
        self,
        backend: SupportsEmbed | None = None,
        *,
        max_cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        """Initialize an embedding encoder with a bounded LRU cache.

        Args:
            backend: Embedding backend. Defaults to `OllamaEmbeddingBackend`.
            max_cache_size: Maximum cached embeddings. Zero disables caching.

        Raises:
            ValueError: If `max_cache_size` is negative.
        """
        if max_cache_size < 0:
            raise ValueError("max_cache_size must be greater than or equal to zero")

        self._backend = backend or OllamaEmbeddingBackend()
        self._max_cache_size = max_cache_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    @property
    def cache_size(self) -> int:
        """Return the current number of cached exact-text entries.

        Returns:
            Number of cached normalized embeddings.
        """
        return len(self._cache)

    def clear_cache(self) -> None:
        """Remove all cached embeddings without changing encoder configuration."""
        self._cache.clear()

    def encode(self, text: str) -> list[float]:
        """Return an L2-normalized vector, caching it by exact text when enabled.

        A cache hit refreshes the entry's LRU recency. When insertion exceeds
        `max_cache_size`, the least-recently-used entry is discarded.

        Args:
            text: Text to encode.

        Returns:
            L2-normalized embedding vector.
        """
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]

        import numpy as np

        raw = self._backend.embed(text)
        vector = np.array(raw, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        normalized = cast(
            list[float],
            (vector / norm).tolist() if norm > 0 else vector.tolist(),
        )

        if self._max_cache_size > 0:
            self._cache[text] = normalized
            if len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)

        return normalized

    def similarity(self, left: str, right: str) -> float:
        """Return cosine similarity between two texts.

        Args:
            left: First text.
            right: Second text.

        Returns:
            Cosine similarity in the embedding space.
        """
        import numpy as np

        left_vector = np.array(self.encode(left), dtype=np.float64)
        right_vector = np.array(self.encode(right), dtype=np.float64)
        return float(np.dot(left_vector, right_vector))
