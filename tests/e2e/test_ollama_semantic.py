import socket

import pytest

from pytest_mellea import Behavior, Content

pytestmark = [pytest.mark.e2e, pytest.mark.ollama, pytest.mark.semantic]


def _ollama_available() -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        return sock.connect_ex(("127.0.0.1", 11434)) == 0
    finally:
        sock.close()


def _model_available(fragment: str) -> bool:
    try:
        import ollama

        return any(fragment in (model.model or "") for model in ollama.list().models)
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(), reason="Ollama is not running on 127.0.0.1:11434"
)
requires_embed_model = pytest.mark.skipif(
    not _model_available("granite-embedding:278m"),
    reason="Embedding model unavailable. Run: ollama pull granite-embedding:278m",
)
requires_judge_model = pytest.mark.skipif(
    not _model_available("granite4.1:3b"),
    reason="Judge model unavailable. Run: ollama pull granite4.1:3b",
)


@requires_ollama
@requires_embed_model
def test_content_static_redis_example() -> None:
    response = "Redis is an in-memory key-value store used for caching."

    assert "key-value store" in Content(response)
    assert "relational joins and SQL queries" not in Content(response)


@requires_ollama
@requires_judge_model
def test_behavior_static_factual_answer() -> None:
    response = "The capital of France is Paris."

    assert "factual answer" in Behavior(response)
    assert "safety refusal or content policy rejection" not in Behavior(response)
