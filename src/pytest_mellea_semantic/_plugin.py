"""Pytest plugin integration for semantic assertions."""

from __future__ import annotations

import os
from typing import Any

import pytest

from pytest_mellea_semantic._runtime import (
    DEFAULT_ENCODER_MODEL,
    DEFAULT_JUDGE_BACKEND,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_THRESHOLD,
    configure,
)

ENV_THRESHOLD = "MELLEA_SEMANTIC_THRESHOLD"
ENV_ENCODER_MODEL = "MELLEA_SEMANTIC_ENCODER_MODEL"
ENV_OLLAMA_HOST = "MELLEA_SEMANTIC_OLLAMA_HOST"
ENV_JUDGE_BACKEND = "MELLEA_SEMANTIC_JUDGE_BACKEND"
ENV_JUDGE_MODEL = "MELLEA_SEMANTIC_JUDGE_MODEL"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register semantic assertion CLI and ini configuration.

    Args:
        parser: Pytest parser used to register options.
    """
    group = parser.getgroup("mellea-semantic")
    group.addoption(
        "--mellea-semantic-threshold",
        action="store",
        type=float,
        default=None,
        help="Default cosine similarity threshold for Content assertions.",
    )
    group.addoption(
        "--mellea-semantic-encoder-model",
        action="store",
        default=None,
        help="Ollama embedding model used by Content assertions.",
    )
    group.addoption(
        "--mellea-semantic-ollama-host",
        action="store",
        default=None,
        help="Ollama host used by Content assertions.",
    )
    group.addoption(
        "--mellea-semantic-judge-backend",
        action="store",
        default=None,
        help="Mellea backend used by Behaviour assertions.",
    )
    group.addoption(
        "--mellea-semantic-judge-model",
        action="store",
        default=None,
        help="Mellea model id used by Behaviour assertions.",
    )

    parser.addini(
        "mellea_semantic_threshold",
        default=str(DEFAULT_THRESHOLD),
        help="Default cosine similarity threshold for Content assertions.",
    )
    parser.addini(
        "mellea_semantic_encoder_model",
        default=DEFAULT_ENCODER_MODEL,
        help="Ollama embedding model used by Content assertions.",
    )
    parser.addini(
        "mellea_semantic_ollama_host",
        default="",
        help="Ollama host used by Content assertions.",
    )
    parser.addini(
        "mellea_semantic_judge_backend",
        default=DEFAULT_JUDGE_BACKEND,
        help="Mellea backend used by Behaviour assertions.",
    )
    parser.addini(
        "mellea_semantic_judge_model",
        default=DEFAULT_JUDGE_MODEL,
        help="Mellea model id used by Behaviour assertions.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Apply pytest configuration and register markers.

    Args:
        config: Active pytest config object.
    """
    config.addinivalue_line(
        "markers", "semantic: tests using pytest-mellea-semantic assertions"
    )
    configure(**_config_from_pytest(config))


def pytest_assertrepr_compare(op: str, left: object, right: object) -> list[str] | None:
    """Provide focused assertion output for semantic `in` comparisons.

    Args:
        op: Comparison operator from pytest assertion rewriting.
        left: Left operand.
        right: Right operand.

    Returns:
        Custom failure lines, or `None` for unrelated assertions.
    """
    if op not in {"in", "not in"} or not isinstance(left, str):
        return None

    from pytest_mellea_semantic import Behaviour, Content

    if isinstance(right, Content):
        lines = ["Semantic Content assertion failed"]
        lines.append(f"  Expected content : {left!r}")
        lines.append(f"  Response preview : {Content._preview(right.response)!r}")
        if right._last_similarity is not None:
            relation = (
                "above" if right._last_similarity >= right.active_threshold else "below"
            )
            lines.append(f"  Similarity score : {right._last_similarity:.4f}")
            lines.append(f"  Threshold        : {right.active_threshold:.2f}")
            lines.append(
                f"  Result           : score is {relation} threshold for operator {op!r}"
            )
        return lines

    if isinstance(right, Behaviour):
        lines = ["Semantic Behaviour assertion failed"]
        lines.append(f"  Expected behaviour : {left!r}")
        lines.append(f"  Response preview   : {Content._preview(right.response)!r}")
        if right._last_requirement:
            lines.append(f"  Judge requirement  : {right._last_requirement}")
        if right._last_reason:
            lines.append(f"  Judge reason       : {right._last_reason}")
        return lines

    return None


def _config_from_pytest(config: pytest.Config) -> dict[str, Any]:
    """Resolve package configuration from CLI, environment, ini, and defaults.

    Args:
        config: Active pytest configuration.

    Returns:
        Keyword arguments accepted by `pytest_mellea_semantic._runtime.configure`.
    """
    return {
        "threshold": _resolve_float(
            config,
            cli_name="mellea_semantic_threshold",
            env_name=ENV_THRESHOLD,
            ini_name="mellea_semantic_threshold",
            default=DEFAULT_THRESHOLD,
        ),
        "encoder_model": _resolve_str(
            config,
            cli_name="mellea_semantic_encoder_model",
            env_name=ENV_ENCODER_MODEL,
            ini_name="mellea_semantic_encoder_model",
            default=DEFAULT_ENCODER_MODEL,
        ),
        "ollama_host": _resolve_optional_str(
            config,
            cli_name="mellea_semantic_ollama_host",
            env_name=ENV_OLLAMA_HOST,
            ini_name="mellea_semantic_ollama_host",
        ),
        "judge_backend": _resolve_str(
            config,
            cli_name="mellea_semantic_judge_backend",
            env_name=ENV_JUDGE_BACKEND,
            ini_name="mellea_semantic_judge_backend",
            default=DEFAULT_JUDGE_BACKEND,
        ),
        "judge_model": _resolve_str(
            config,
            cli_name="mellea_semantic_judge_model",
            env_name=ENV_JUDGE_MODEL,
            ini_name="mellea_semantic_judge_model",
            default=DEFAULT_JUDGE_MODEL,
        ),
    }


def _resolve_float(
    config: pytest.Config,
    *,
    cli_name: str,
    env_name: str,
    ini_name: str,
    default: float,
) -> float:
    """Resolve a float from CLI, environment, ini, or default.

    Args:
        config: Active pytest configuration.
        cli_name: Pytest option destination name.
        env_name: Environment variable name.
        ini_name: Pytest ini option name.
        default: Default value.

    Returns:
        Resolved float.
    """
    cli_value = config.getoption(cli_name, default=None)
    if cli_value is not None:
        return float(cli_value)
    env_value = os.environ.get(env_name)
    if env_value:
        return float(env_value)
    ini_value = config.getini(ini_name)
    return float(ini_value or default)


def _resolve_str(
    config: pytest.Config,
    *,
    cli_name: str,
    env_name: str,
    ini_name: str,
    default: str,
) -> str:
    """Resolve a string from CLI, environment, ini, or default.

    Args:
        config: Active pytest configuration.
        cli_name: Pytest option destination name.
        env_name: Environment variable name.
        ini_name: Pytest ini option name.
        default: Default value.

    Returns:
        Resolved string.
    """
    cli_value = config.getoption(cli_name, default=None)
    if cli_value:
        return str(cli_value)
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    ini_value = config.getini(ini_name)
    return str(ini_value or default)


def _resolve_optional_str(
    config: pytest.Config,
    *,
    cli_name: str,
    env_name: str,
    ini_name: str,
) -> str | None:
    """Resolve an optional string from CLI, environment, or ini.

    Args:
        config: Active pytest configuration.
        cli_name: Pytest option destination name.
        env_name: Environment variable name.
        ini_name: Pytest ini option name.

    Returns:
        Resolved string or `None`.
    """
    cli_value = config.getoption(cli_name, default=None)
    if cli_value:
        return str(cli_value)
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    ini_value = config.getini(ini_name)
    return str(ini_value) if ini_value else None
