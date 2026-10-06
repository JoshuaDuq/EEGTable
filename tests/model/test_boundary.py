from __future__ import annotations

import ast
import pathlib

import pytest

MODEL = pathlib.Path(__file__).parents[2] / "src" / "eegtable" / "model"


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", sorted(MODEL.glob("*.py")), ids=lambda p: p.name)
def test_no_model_module_imports_the_runner(path: pathlib.Path) -> None:
    assert not {n for n in _imported_modules(path) if n.startswith("eegtable.runner")}


@pytest.mark.parametrize("path", sorted(MODEL.glob("*.py")), ids=lambda p: p.name)
def test_no_model_module_reads_toml_or_writes_files(path: pathlib.Path) -> None:
    # The science half takes arrays and returns values. Reading a recipe or writing a
    # result is the runner's job, and keeping that true is what makes every module here
    # testable with arrays alone.
    forbidden = {"tomllib", "tomli", "logging"}
    assert not _imported_modules(path) & forbidden
    source = path.read_text(encoding="utf-8")
    assert "open(" not in source
    assert ".write_text(" not in source


def test_top_level_eegtable_does_not_import_model() -> None:
    # Importing eegtable must not cost scikit-learn.
    init = (MODEL.parent / "__init__.py").read_text(encoding="utf-8")
    assert "eegtable.model" not in init
