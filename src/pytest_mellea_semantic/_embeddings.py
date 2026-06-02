"""Embedding backends and cosine similarity helpers for content assertions."""

from __future__ import annotations

import abc
from typing import Protocol

from pytest_mellea_semantic._exceptions import SemanticAssertionRuntimeError


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
        self, model_id: str = "nomic-embed-text:v1.5", host: str | None = None
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
    """Caching encoder that normalizes embeddings and computes cosine similarity.

    Args:
        backend: Embedding backend. Defaults to `OllamaEmbeddingBackend`.
    """

    def __init__(self, backend: SupportsEmbed | None = None) -> None:
        """Initialize the embedding encoder."""
        self._backend = backend or OllamaEmbeddingBackend()
        self._cache: dict[str, list[float]] = {}

    @property
    def cache_size(self) -> int:
        """Return the number of cached texts.

        Returns:
            Number of cached normalized embeddings.
        """
        return len(self._cache)

    def encode(self, text: str) -> list[float]:
        """Return a cached, L2-normalized vector for text.

        Args:
            text: Text to encode.

        Returns:
            L2-normalized embedding vector.
        """
        if text not in self._cache:
            import numpy as np

            raw = self._backend.embed(text)
            vector = np.array(raw, dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            normalized = (vector / norm).tolist() if norm > 0 else vector.tolist()
            self._cache[text] = normalized
        return self._cache[text]

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
