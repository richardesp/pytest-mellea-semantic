# Calibration

The default `Content` threshold is `0.70`, calibrated for concise static assertions with `nomic-embed-text:v1.5`.

Live LLM responses often include extra words or valid paraphrases, which can dilute embedding similarity. Prefer a per-assertion threshold such as `0.60` for live qualitative tests after checking positive and negative examples for your domain.

`Behaviour` uses Mellea's LLM-as-a-judge path with `temperature=0` by default. It is still a model judgement, so keep behaviour phrases clear and stable, for example:

- `"factual answer"`
- `"direct answer"`
- `"safety refusal or content policy rejection"`
