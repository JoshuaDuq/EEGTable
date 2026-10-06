import tomllib
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def test_packaging_only_advertises_implemented_features() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_dependencies = configuration["project"]["optional-dependencies"]

    capabilities = {
        "spectral-model": "specparam",
        "irasa": "neurodsp",
        "cycles": "bycycle",
        "complexity": "antropy",
        "pac": "tensorpac",
        "riemann": "pyriemann",
    }
    for extra, package in capabilities.items():
        assert any(package in requirement for requirement in optional_dependencies[extra])
        assert package in (ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert package in (ROOT / "docs" / "install.rst").read_text(encoding="utf-8").lower()


def test_package_description_represents_the_full_feature_scope() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert configuration["project"]["description"] == (
        "Labelled EEG feature extraction and modeling for MNE objects"
    )


def test_ci_covers_supported_endpoints_optional_integrations_and_the_wheel() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.11", "3.14"]' in workflow
    assert (
        ".[dev,model,connectivity,microstates,importance,references,preprocessing,preprocessing-auto,bids,spectral-model,irasa,cycles,complexity,pac,riemann]"
        in workflow
    )
    # Every cross-check against a third-party implementation is guarded by
    # find_spec, so an uninstalled reference skips silently. Importing them in CI
    # is what makes a missing one fail the build instead of quietly passing.
    assert (
        'python -c "import antropy, shap, mne_connectivity, sklearn, yaml, autoreject, pyprep, '
        'mne_bids, specparam, neurodsp, bycycle, tensorpac, pyriemann"' in workflow
    )
    for suite in (
        "tests/test_complexity.py",
        "tests/test_higuchi.py",
        "tests/model",
        "tests/test_spectral_model.py",
        "tests/test_irasa.py",
        "tests/test_cycles.py",
        "tests/test_bids.py",
    ):
        assert suite in workflow
    assert "python -m build" in workflow
    assert "pip install dist/*.whl" in workflow
    assert 'python -c "import eegtable"' in workflow
    assert 'MNE_DONTWRITE_HOME: "true"' in workflow


@pytest.mark.skipif(find_spec("sklearn") is not None, reason="scikit-learn is installed")
def test_the_model_subpackage_names_its_extra_when_scikit_learn_is_absent() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"pip install 'eegtable\[model\]'"):
        import eegtable.model  # noqa: F401
