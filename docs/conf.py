import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = "bns_remnant_classifier"
copyright = "2026, Anna Puecher"
author = "Anna Puecher"

try:
    from importlib.metadata import version as _version
    release = _version("bns_remnant_classifier")
except Exception:
    release = "1.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# Heavy scientific dependencies are mocked so the docs build doesn't need
# the full (and large) runtime dependency stack installed just to read
# docstrings and signatures.
autodoc_mock_imports = [
    "numpy", "scipy", "pandas", "joblib", "sklearn",
    "bilby", "astropy", "tqdm",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
