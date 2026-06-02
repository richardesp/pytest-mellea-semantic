import socket

import pytest

from pytest_mellea_semantic import Behaviour, Content

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
    not _model_available("nomic-embed-text"),
    reason="Embedding model unavailable. Run: ollama pull nomic-embed-text:v1.5",
)
requires_judge_model = pytest.mark.skipif(
    not _model_available("gemma4:e2b"),
    reason="Judge model unavailable. Run: ollama pull gemma4:e2b",
)


@requires_ollama
@requires_embed_model
def test_content_static_redis_example() -> None:
    response = "Redis is an in-memory key-value store used for caching."

    assert "key-value store" in Content(response)
    assert "relational joins and SQL queries" not in Content(response)


@requires_ollama
@requires_judge_model
def test_behaviour_static_factual_answer() -> None:
    response = "The capital of France is Paris."

    assert "factual answer" in Behaviour(response)
    assert "safety refusal or content policy rejection" not in Behaviour(response)
