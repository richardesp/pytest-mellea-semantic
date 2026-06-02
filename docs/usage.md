# Usage

Install the package in a project that already uses Mellea:

```bash
uv add pytest-mellea-semantic
```

Pull the default local models when using the default Ollama-backed runtime:

```bash
ollama pull nomic-embed-text:v1.5
ollama pull gemma4:e2b
```

Then write normal pytest assertions:

```python
from pytest_mellea_semantic import Content, Behaviour


def test_redis_response(session):
    response = session.instruct("What is Redis? Answer in one sentence.").value

    assert "key-value store" in Content(response, threshold=0.60)
    assert "machine learning" not in Content(response, threshold=0.60)
    assert "factual answer" in Behaviour(response)
    assert "safety refusal" not in Behaviour(response)
```

`Content` validates what the response says using embedding similarity.
`Behaviour` validates how the response behaves using Mellea's LLM-as-a-judge requirement pipeline.
