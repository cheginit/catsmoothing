"""Configuration for pytest."""

from __future__ import annotations

import builtins
import importlib
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(autouse=True)
def add_doctest_imports(doctest_namespace: dict[str, object]) -> None:
    """Add catsmoothing namespace for doctest."""
    import catsmoothing as cs

    doctest_namespace["catsmoothing"] = cs


@pytest.fixture
def block_optional_imports(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def _block(*names: str) -> None:
        original_import = builtins.__import__

        def mocked_import(name: str, *args: Any, **kwargs: Any):
            if name in names:
                raise ImportError(f"Import of '{name}' is blocked for testing")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)
        monkeypatch.setattr(importlib, "import_module", mocked_import)

    return _block
