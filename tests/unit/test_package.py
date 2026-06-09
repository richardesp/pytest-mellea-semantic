from importlib import metadata, util

import pytest_mellea


def test_package_uses_new_import_name() -> None:
    legacy_package = "pytest_mellea_" + "semantic"

    assert pytest_mellea.__name__ == "pytest_mellea"
    assert util.find_spec(legacy_package) is None


def test_distribution_registers_renamed_pytest_plugin() -> None:
    distribution = metadata.distribution("pytest-mellea")
    pytest_plugins = [
        entry_point
        for entry_point in distribution.entry_points
        if entry_point.group == "pytest11"
    ]

    assert [
        (entry_point.name, entry_point.value) for entry_point in pytest_plugins
    ] == [("pytest-mellea", "pytest_mellea._plugin")]
