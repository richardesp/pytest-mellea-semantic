# pytest-mellea-semantic

Pytest-native semantic assertions powered by [Mellea](https://github.com/generative-computing/mellea).

```python
from pytest_mellea_semantic import Content, Behaviour

assert "key-value store" in Content(response)
assert "safety refusal" not in Behaviour(response)
```

`Content` checks what a response says using embedding similarity. `Behaviour` checks how a response behaves using Mellea's LLM-as-a-judge requirement pipeline.

## Install

```bash
uv add pytest-mellea-semantic
```

For local development:

```bash
git clone https://github.com/generative-computing/pytest-mellea-semantic.git
cd pytest-mellea-semantic
mise run setup
```

## Default local models

The default runtime uses Ollama:

```bash
ollama pull nomic-embed-text:v1.5
ollama pull gemma4:e2b
```

## Configuration

Pytest ini options:

```ini
[pytest]
mellea_semantic_threshold = 0.70
mellea_semantic_encoder_model = nomic-embed-text:v1.5
mellea_semantic_cache_size = 1024
mellea_semantic_judge_backend = ollama
mellea_semantic_judge_model = gemma4:e2b
```

CLI options with the same names are available using hyphens, for example:

```bash
pytest --mellea-semantic-threshold=0.60
```

The shared encoder caches normalized embeddings by exact input text using
least-recently-used eviction. Its default capacity is 1024 entries. Set
`mellea_semantic_cache_size = 0` to disable caching. The equivalent CLI option
is `--mellea-semantic-cache-size`, and the environment variable is
`MELLEA_SEMANTIC_CACHE_SIZE`.

Shared runtime configuration precedence is CLI, environment, pytest ini, then
package defaults. A custom encoder can set its own capacity independently and
clear cached entries without changing its backend or configuration:

```python
from pytest_mellea_semantic import EmbeddingEncoder

encoder = EmbeddingEncoder(max_cache_size=256)
encoder.clear_cache()
```

Pass `max_cache_size=0` to a custom encoder to disable caching. Negative shared
or custom capacities raise `ValueError`.

## Development

The pre-commit and pre-push flow is handled directly by the tasks in
[mise.toml](mise.toml).

```bash
mise run check
mise run tests
mise run build
```

`mise run setup` installs native `.git/hooks/pre-commit` and `.git/hooks/pre-push` scripts.
