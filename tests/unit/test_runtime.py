from pytest_mellea_semantic._runtime import (
    DEFAULT_ENCODER_MODEL,
    DEFAULT_THRESHOLD,
    configure,
    get_config,
    reset_runtime,
)


def setup_function() -> None:
    reset_runtime()


def test_runtime_defaults() -> None:
    config = get_config()

    assert config.threshold == DEFAULT_THRESHOLD
    assert config.encoder_model == DEFAULT_ENCODER_MODEL


def test_configure_updates_values() -> None:
    config = configure(threshold=0.82, encoder_model="custom-embed")

    assert config.threshold == 0.82
    assert config.encoder_model == "custom-embed"
