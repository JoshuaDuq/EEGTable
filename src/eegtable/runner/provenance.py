"""Recipe-scoped code and software identities for extraction reuse."""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

import eegtable.provenance as provenance
from eegtable.runner.measures import GRAPH, REQUIRES, SEGMENTATION, get
from eegtable.runner.recipe import Recipe

_ROOT = Path(provenance.__file__).parent
_REGISTRIES = {"eegtable", "eegtable.runner.measures"}
_DISTRIBUTIONS = {"sklearn": "scikit-learn", "mne_connectivity": "mne-connectivity"}
_IMPLEMENTATION_PACKAGES = {
    get(name).function: _DISTRIBUTIONS.get(module, module) for name, (module, _) in REQUIRES.items()
}


def _module_source(module: str) -> Path:
    path = _ROOT.joinpath(*module.split(".")[1:])
    return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")


def _local_imports(module: str, source: Path) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            package = module if source.name == "__init__.py" else module.rpartition(".")[0]
            imported = resolve_name(name, package)
            imports.add(imported)
            if node.module is None:
                imports.update(
                    f"{imported}.{alias.name}"
                    for alias in node.names
                    if _module_source(f"{imported}.{alias.name}").is_file()
                )
    return {name for name in imports if name == "eegtable" or name.startswith("eegtable.")}


def extraction_hash(recipe: Recipe) -> str:
    """Hash shared extraction modules and the selected measures' local dependencies.

    Registry modules are hashed without following their bulk imports. Their
    selected numerical functions seed the dependency graph explicitly. Module
    scope remains conservative: edits to shared modules invalidate their users.
    """
    functions = [get(spec.measure).function for spec in recipe.features]
    functions.extend(GRAPH[name] for spec in recipe.features for name in spec.graph)
    if any(get(spec.measure).kind == "microstates" for spec in recipe.features):
        functions.append(SEGMENTATION.function)
    pending = {"eegtable.runner.batch", *(function.__module__ for function in functions)}
    sources = {}
    while pending:
        module = pending.pop()
        if module in sources or not (module == "eegtable" or module.startswith("eegtable.")):
            continue
        source = _module_source(module)
        sources[module] = provenance.file_hash(source)
        if module not in _REGISTRIES:
            pending.update(_local_imports(module, source))
    return provenance.identity(sources)


def extraction_software(recipe: Recipe) -> dict[str, str]:
    """Read current versions of the recipe's numerical and provider dependencies."""
    packages = {"mne", "numpy", "scipy", "pandas"}
    for spec in recipe.features:
        measure = get(spec.measure)
        provider = measure.provider
        if provider is not None:
            packages.add(provider[0])
        if measure.kind == "microstates":
            packages.add("scikit-learn")
        if measure.function in _IMPLEMENTATION_PACKAGES:
            packages.add(_IMPLEMENTATION_PACKAGES[measure.function])
    return provenance.dependency_versions(packages)
