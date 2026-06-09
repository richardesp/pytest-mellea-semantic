# Design

`pytest-mellea` is a thin pytest-native workflow layer over Mellea.
It is intentionally separate from Mellea's JSON-oriented `unit_test_eval` component:

- `unit_test_eval` is for bulk/offline evaluation cases.
- `pytest-mellea` is for inline assertions in ordinary Python test suites.

The package exposes two assertion dimensions:

| Wrapper | Mechanism | Validates |
| --- | --- | --- |
| `Content` | Embedding cosine similarity | What the response says |
| `Behavior` | Mellea LLM-as-a-judge requirements | How the response behaves |

The pytest plugin only configures defaults and improves assertion output. The assertion objects can also be used directly outside pytest.
