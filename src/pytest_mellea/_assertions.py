"""Assertion wrapper objects for semantic pytest assertions."""

from __future__ import annotations

from typing import Any

from pytest_mellea._constants import DEFAULT_JUDGE_MODEL_OPTIONS
from pytest_mellea._embeddings import EmbeddingEncoder
from pytest_mellea._runtime import (
    get_config,
    get_encoder,
    get_judge_session,
)


class Content:
    """Semantic content assertion wrapper.

    Args:
        response: LLM response text to validate.
        threshold: Optional per-assertion cosine similarity threshold.
        encoder: Optional encoder override, useful for tests or custom providers.
    """

    def __init__(
        self,
        response: str,
        *,
        threshold: float | None = None,
        encoder: EmbeddingEncoder | None = None,
    ) -> None:
        """Initialize the content assertion wrapper."""
        self.response = response
        self.threshold = threshold
        self.encoder = encoder
        self._last_similarity: float | None = None
        self._last_expected: str | None = None

    def __contains__(self, expected: str) -> bool:
        """Return whether the response semantically contains expected text.

        Args:
            expected: Expected semantic concept or content.

        Returns:
            `True` when similarity is greater than or equal to the active threshold.
        """
        self._last_expected = expected
        self._last_similarity = None
        encoder = self.encoder or get_encoder()
        threshold = self.active_threshold
        self._last_similarity = encoder.similarity(self.response, expected)
        return self._last_similarity >= threshold

    @property
    def active_threshold(self) -> float:
        """Return the threshold used by this assertion.

        Returns:
            Per-instance threshold or configured default.
        """
        return self.threshold if self.threshold is not None else get_config().threshold

    def __repr__(self) -> str:
        """Return a compact pytest-friendly representation.

        Returns:
            Short representation with a truncated response preview.
        """
        return f"Content({self._preview(self.response)!r})"

    @staticmethod
    def _preview(text: str, limit: int = 80) -> str:
        """Return a compact single-line preview.

        Args:
            text: Text to preview.
            limit: Maximum preview length.

        Returns:
            Truncated single-line preview.
        """
        clean = " ".join(text.split())
        return clean if len(clean) <= limit else f"{clean[: limit - 3]}..."


class Behavior:
    """Semantic behavior assertion wrapper using Mellea LLM-as-a-judge.

    Args:
        response: LLM response text to validate.
        session: Optional Mellea session override.
        model_options: Optional judge model options merged over the default
            `temperature=0` options.
    """

    def __init__(
        self,
        response: str,
        *,
        session: Any | None = None,
        model_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the behavior assertion wrapper."""
        self.response = response
        self.session = session
        self.model_options = {
            **DEFAULT_JUDGE_MODEL_OPTIONS,
            **(model_options or {}),
        }
        self._last_reason: str | None = None
        self._last_expected: str | None = None
        self._last_requirement: str | None = None

    def __contains__(self, expected: str) -> bool:
        """Return whether the response exhibits the expected behavior.

        Args:
            expected: Behavior phrase to judge.

        Returns:
            `True` when Mellea's judge requirement passes.
        """
        self._last_expected = expected
        self._last_requirement = None
        self._last_reason = None

        from mellea.core import ModelOutputThunk
        from mellea.stdlib.requirements import LLMaJRequirement

        session = self.session or get_judge_session()
        requirement_text = f'The response exhibits the behavior "{expected}".'
        self._last_requirement = requirement_text
        results = session.validate(
            [LLMaJRequirement(requirement_text)],
            output=ModelOutputThunk(self.response),
            model_options=self.model_options,
        )

        result = results[0]
        self._last_reason = result.reason
        return bool(result)

    def __repr__(self) -> str:
        """Return a compact pytest-friendly representation.

        Returns:
            Short representation with a truncated response preview.
        """
        return f"Behavior({Content._preview(self.response)!r})"
