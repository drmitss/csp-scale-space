"""Sphinx configuration (Read the Docs)."""
project = "csp-scalespace"
author = "Dimitris M. Manias, Laurent Najman"
copyright = "2026, " + author
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
exclude_patterns = ["_build"]
html_theme = "furo"
html_static_path = ["_static"]
html_title = "csp-scalespace"
html_theme_options = {
    "source_repository": "https://github.com/drmitss/csp-scale-space/",
    "source_branch": "main",
    "source_directory": "docs/",
}
