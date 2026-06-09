# Usage

Install the package in a project that already uses Mellea:

```bash
uv add pytest-mellea
```

Pull the default local models when using the default Ollama-backed runtime:

```bash
ollama pull granite-embedding:278m
ollama pull granite4.1:3b
```

The Granite defaults align the plugin with Mellea's default local model family,
avoiding an additional judge-model download for an existing Mellea setup.

Then write normal pytest assertions:

```python
from pytest_mellea import Behavior, Content


def test_redis_response(session):
    response = session.instruct("What is Redis? Answer in one sentence.").value

    assert "key-value store" in Content(response, threshold=0.60)
    assert "machine learning" not in Content(response, threshold=0.60)
    assert "factual answer" in Behavior(response)
    assert "safety refusal" not in Behavior(response)
```

`Content` validates what the response says using embedding similarity.
`Behavior` validates how the response behaves using Mellea's LLM-as-a-judge
requirement pipeline.
