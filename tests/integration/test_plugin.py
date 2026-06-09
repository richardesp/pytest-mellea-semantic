import pytest
from _pytest.pytester import Pytester

pytest_plugins = ["pytester"]


def make_child_ini(pytester: Pytester, extra: str = "") -> None:
    pytester.makeini(
        f"""
        [pytest]
        asyncio_default_fixture_loop_scope = function
        {extra}
        """
    )


def test_plugin_registers_marker(pytester: Pytester) -> None:
    make_child_ini(pytester)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.semantic
        def test_marked():
            assert True
        """
    )

    result = pytester.runpytest("--strict-markers")

    result.assert_outcomes(passed=1)


def test_plugin_applies_default_configuration(pytester: Pytester) -> None:
    make_child_ini(pytester)
    pytester.makepyfile(
        """
        from pytest_mellea._runtime import get_config

        def test_config():
            config = get_config()
            assert config.threshold == 0.65
            assert config.encoder_model == "granite-embedding:278m"
            assert config.judge_model == "granite4.1:3b"
        """
    )

    result = pytester.runpytest()

    result.assert_outcomes(passed=1)


def test_plugin_applies_ini_configuration(pytester: Pytester) -> None:
    make_child_ini(
        pytester,
        """
        mellea_semantic_threshold = 0.42
        mellea_semantic_cache_size = 64
        """,
    )
    pytester.makepyfile(
        """
        from pytest_mellea._runtime import get_config

        def test_config():
            assert get_config().threshold == 0.42
            assert get_config().cache_size == 64
        """
    )

    result = pytester.runpytest()

    result.assert_outcomes(passed=1)


def test_plugin_cli_overrides_ini_configuration(pytester: Pytester) -> None:
    make_child_ini(pytester, "mellea_semantic_threshold = 0.42")
    pytester.makepyfile(
        """
        from pytest_mellea._runtime import get_config

        def test_config():
            assert get_config().threshold == 0.91
        """
    )

    result = pytester.runpytest("--mellea-semantic-threshold=0.91")

    result.assert_outcomes(passed=1)


def test_plugin_environment_overrides_ini_cache_size(
    pytester: Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MELLEA_SEMANTIC_CACHE_SIZE", "32")
    make_child_ini(pytester, "mellea_semantic_cache_size = 64")
    pytester.makepyfile(
        """
        from pytest_mellea._runtime import get_config

        def test_config():
            assert get_config().cache_size == 32
        """
    )

    result = pytester.runpytest()

    result.assert_outcomes(passed=1)


def test_plugin_cli_overrides_environment_cache_size(
    pytester: Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MELLEA_SEMANTIC_CACHE_SIZE", "32")
    make_child_ini(pytester, "mellea_semantic_cache_size = 64")
    pytester.makepyfile(
        """
        from pytest_mellea._runtime import get_config

        def test_config():
            assert get_config().cache_size == 16
        """
    )

    result = pytester.runpytest("--mellea-semantic-cache-size=16")

    result.assert_outcomes(passed=1)


def test_plugin_assertion_output_for_content(pytester: Pytester) -> None:
    make_child_ini(pytester)
    pytester.makepyfile(
        """
        from pytest_mellea import Content, EmbeddingEncoder

        class Backend:
            def embed(self, text):
                return {"response": [1, 0], "expected": [0, 1]}[text]

        def test_failure():
            content = Content(
                "response", threshold=0.7, encoder=EmbeddingEncoder(Backend())
            )
            assert "expected" in content
        """
    )

    result = pytester.runpytest()

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Semantic Content assertion failed*"])


def test_plugin_assertion_output_for_content_not_in(pytester: Pytester) -> None:
    make_child_ini(pytester)
    pytester.makepyfile(
        """
        from pytest_mellea import Content, EmbeddingEncoder

        class Backend:
            def embed(self, text):
                return {"response": [1, 0], "expected": [1, 0]}[text]

        def test_failure():
            content = Content(
                "response", threshold=0.7, encoder=EmbeddingEncoder(Backend())
            )
            assert "expected" not in content
        """
    )

    result = pytester.runpytest()

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*Semantic Content assertion failed*",
            "*score is above threshold for operator 'not in'*",
        ]
    )


def test_plugin_assertion_output_for_behavior(pytester: Pytester) -> None:
    make_child_ini(pytester)
    pytester.makepyfile(
        """
        from pytest_mellea import Behavior

        class Result:
            reason = "judge said no"

            def __bool__(self):
                return False

        class Session:
            def validate(self, reqs, *, output, model_options):
                return [Result()]

        def test_failure():
            behavior = Behavior("response", session=Session())
            assert "direct answer" in behavior
        """
    )

    result = pytester.runpytest()

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*Semantic Behavior assertion failed*",
            "*Expected behavior : 'direct answer'*",
        ]
    )
