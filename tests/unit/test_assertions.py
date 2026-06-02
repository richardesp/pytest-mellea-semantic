from typing import Any

from pytest_mellea_semantic import Behaviour, Content, EmbeddingEncoder
from pytest_mellea_semantic._runtime import reset_runtime


class FakeEmbeddingBackend:
    def embed(self, text: str) -> list[float]:
        vectors = {
            "Redis stores key-value data": [1.0, 0.0],
            "key-value store": [0.9, 0.1],
            "relational joins": [0.0, 1.0],
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


def test_behaviour_uses_mellea_requirement_pipeline() -> None:
    session = FakeSession(result=True)
    behaviour = Behaviour("Paris is the capital of France.", session=session)

    assert "factual answer" in behaviour

    reqs, output, model_options = session.calls[0]
    assert (
        'The response exhibits the behaviour "factual answer".' in reqs[0].description
    )
    assert output.value == "Paris is the capital of France."
    assert model_options == {"temperature": 0}
    assert behaviour._last_reason == "judge said yes"


def test_behaviour_negative_result() -> None:
    session = FakeSession(result=False)

    assert "safety refusal" not in Behaviour(
        "Paris is the capital of France.", session=session
    )
