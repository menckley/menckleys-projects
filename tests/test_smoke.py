"""Test de fumée : le notebook est valide et ses dépendances s'importent."""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "vanillaoption.ipynb"


def test_notebook_is_valid_json_with_code_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in nb["cells"])


def test_notebook_dependencies_import():
    import numpy
    from scipy.stats import norm

    assert numpy.isclose(norm.cdf(0.0), 0.5)
