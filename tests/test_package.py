import importlib

import pytest


@pytest.mark.parametrize(
    "module", ["lainbow", "lainbow.core", "lainbow.server", "lainbow.db", "lainbow.engine"]
)
def test_packages_import(module: str) -> None:
    assert importlib.import_module(module).__doc__
