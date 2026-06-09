"""Package-wide default values for semantic assertions."""

from typing import Any, Final

DEFAULT_CACHE_SIZE: Final = 1024
DEFAULT_ENCODER_MODEL: Final = "granite-embedding:278m"
DEFAULT_JUDGE_BACKEND: Final = "ollama"
DEFAULT_JUDGE_MODEL: Final = "granite4.1:3b"
DEFAULT_JUDGE_MODEL_OPTIONS: Final[dict[str, Any]] = {"temperature": 0}
DEFAULT_THRESHOLD: Final = 0.65
