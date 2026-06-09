# Calibration

The default `Content` threshold is `0.65`, calibrated for concise static
assertions with `granite-embedding:278m`.

Live LLM responses often include extra words or valid paraphrases, which can
dilute embedding similarity. Prefer a per-assertion threshold such as `0.60`
for live qualitative tests after checking positive and negative examples for
your domain.

`Behavior` uses Mellea's LLM-as-a-judge path with `temperature=0` by default.
It is still a model judgment, so keep behavior phrases clear and stable, for
example:

- `"factual answer"`
- `"direct answer"`
- `"safety refusal or content policy rejection"`
