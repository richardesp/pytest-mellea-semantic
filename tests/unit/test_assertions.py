from typing import Any

import pytest

import pytest_mellea
from pytest_mellea import Behavior, Content, EmbeddingEncoder
from pytest_mellea._runtime import reset_runtime


class FakeEmbeddingBackend:
    def embed(self, text: str) -> list[float]:
        vectors = {
            "Redis stores key-value data": [1.0, 0.0],
            "key-value store": [0.9, 0.1],
            "relational joins": [0.0, 1.0],
        }
        return vectors[text]


class FailingEmbeddingBackend:
    def embed(self, text: str) -> list[float]:
        if text == "second concept":
            raise RuntimeError("embedding failed")
        vectors = {
            "response": [1.0, 0.0],
            "first concept": [1.0, 0.0],
        }
        return vectors[text]


class FakeValidationResult:
    def __init__(self, result: bool, reason: str) -> None:
        self.reason = reason
        self._result = result

    def __bool__(self) -> bool:
        """Return the fake validation verdict."""
        return self._result


class FakeSession:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.calls: list[tuple[list[Any], Any, dict[str, Any]]] = []

    def validate(
        self, reqs: list[Any], *, output: Any, model_options: dict[str, Any]
    ) -> list[FakeValidationResult]:
        self.calls.append((reqs, output, model_options))
        return [FakeValidationResult(self.result, "judge said yes")]


class FailingAfterSuccessSession(FakeSession):
    def validate(
        self, reqs: list[Any], *, output: Any, model_options: dict[str, Any]
    ) -> list[FakeValidationResult]:
        if self.calls:
            raise RuntimeError("judge failed")
        return super().validate(reqs, output=output, model_options=model_options)


def setup_function() -> None:
    reset_runtime()


def test_content_contains_semantic_match() -> None:
    encoder = EmbeddingEncoder(backend=FakeEmbeddingBackend())
    content = Content("Redis stores key-value data", threshold=0.7, encoder=encoder)

    assert "key-value store" in content
    assert content._last_similarity is not None


def test_content_not_in_semantic_mismatch() -> None:
    encoder = EmbeddingEncoder(backend=FakeEmbeddingBackend())
    content = Content("Redis stores key-value data", threshold=0.7, encoder=encoder)

    assert "relational joins" not in content


def test_content_threshold_controls_match() -> None:
    encoder = EmbeddingEncoder(backend=FakeEmbeddingBackend())
    content = Content("Redis stores key-value data", threshold=0.999, encoder=encoder)

    assert "key-value store" not in content


def test_content_clears_stale_similarity_when_embedding_fails() -> None:
    encoder = EmbeddingEncoder(backend=FailingEmbeddingBackend())
    content = Content("response", threshold=0.7, encoder=encoder)

    assert "first concept" in content
    with pytest.raises(RuntimeError, match="embedding failed"):
        content.__contains__("second concept")

    assert content._last_expected == "second concept"
    assert content._last_similarity is None


def test_public_api_exports_only_behavior_spelling() -> None:
    legacy_name = "Behav" + "iour"

    assert pytest_mellea.Behavior is Behavior
    assert "Behavior" in pytest_mellea.__all__
    assert legacy_name not in pytest_mellea.__all__
    assert not hasattr(pytest_mellea, legacy_name)


def test_behavior_uses_mellea_requirement_pipeline() -> None:
    session = FakeSession(result=True)
    behavior = Behavior("Paris is the capital of France.", session=session)

    assert "factual answer" in behavior

    reqs, output, model_options = session.calls[0]
    assert reqs[0].description == (
        'The response exhibits the behavior "factual answer".'
    )
    assert output.value == "Paris is the capital of France."
    assert model_options == {"temperature": 0}
    assert behavior._last_reason == "judge said yes"
    assert repr(behavior) == "Behavior('Paris is the capital of France.')"


def test_behavior_clears_stale_reason_when_judge_fails() -> None:
    behavior = Behavior(
        "Paris is the capital of France.",
        session=FailingAfterSuccessSession(),
    )

    assert "factual answer" in behavior
    with pytest.raises(RuntimeError, match="judge failed"):
        behavior.__contains__("direct answer")

    assert behavior._last_expected == "direct answer"
    assert behavior._last_requirement == (
        """The response exhibits the behavior "direct answer"."""
    )
    assert behavior._last_reason is None


def test_behavior_negative_result() -> None:
    session = FakeSession(result=False)

    assert "safety refusal" not in Behavior(
        "Paris is the capital of France.", session=session
    )
