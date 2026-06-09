# pytest-mellea

Pytest-native semantic assertions powered by [Mellea](https://github.com/generative-computing/mellea).

```python
from pytest_mellea import Behavior, Content

assert "key-value store" in Content(response)
assert "safety refusal" not in Behavior(response)
```

`Content` checks what a response says using embedding similarity. `Behavior`
checks how a response behaves using Mellea's LLM-as-a-judge requirement
pipeline.

## Install

```bash
uv add pytest-mellea
```

For local development:

```bash
git clone https://github.com/generative-computing/pytest-mellea.git
cd pytest-mellea
mise run setup
```

## Default local models

The default runtime uses Ollama with IBM Granite models:

```bash
ollama pull granite-embedding:278m
ollama pull granite4.1:3b
```

These defaults keep the plugin aligned with Mellea's default local Granite
judge, so an existing Mellea setup does not need a second model family.

## Configuration

Pytest ini options:

```ini
[pytest]
mellea_semantic_threshold = 0.65
mellea_semantic_encoder_model = granite-embedding:278m
mellea_semantic_cache_size = 1024
mellea_semantic_judge_backend = ollama
mellea_semantic_judge_model = granite4.1:3b
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
package defaults. CLI flags therefore override matching `MELLEA_SEMANTIC_*`
environment variables. A custom encoder can set its own capacity independently
and clear cached entries without changing its backend or configuration:

```python
from pytest_mellea import EmbeddingEncoder

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
