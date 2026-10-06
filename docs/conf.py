from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "EEGTable"
author = "Joshua Duquette"
# The package's own version, so the docs cannot name another one.
release = next(
    line.split('"')[1]
    for line in (PROJECT_ROOT / "src" / "eegtable" / "__init__.py").read_text().splitlines()
    if line.startswith("__version__")
)
copyright = "2026, Joshua Duquette"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "notfound.extension",
    "sphinx_gallery.gen_gallery",
]

# ---------------------------------------------------------------------------
# Tutorials: run on every build, so their numbers and figures follow the code.
# ---------------------------------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../tutorials",
    "gallery_dirs": "auto_tutorials",
    "filename_pattern": r"/plot_",
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_signature": False,
    "write_computation_times": False,
}

# ---------------------------------------------------------------------------
# Source / build
# ---------------------------------------------------------------------------
templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = [
    "_build",
    "superpowers",
    "Thumbs.db",
    ".DS_Store",
]

# ---------------------------------------------------------------------------
# Napoleon (docstring style)
# ---------------------------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# ---------------------------------------------------------------------------
# Autodoc
# ---------------------------------------------------------------------------
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_mock_imports = ["mne_connectivity", "shap"]
# Numpydoc type words that name no class, so nitpicky mode reports only real
# dangling references.
nitpick_ignore = [
    ("py:class", n) for n in ("sequence", "mapping", "optional", "array-like", "BANDS_STANDARD")
]
nitpick_ignore_regex = [("py:class", r"\(?-?\d+(\.\d+)?\)?")]

# Keep the name written in the source (`BANDS_STANDARD`) instead of its repr.
autodoc_preserve_defaults = True
toc_object_entries_show_parents = "hide"

# ---------------------------------------------------------------------------
# MyST (Markdown support)
# ---------------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 4

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "mne": ("https://mne.tools/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# ---------------------------------------------------------------------------
# 404 page
# ---------------------------------------------------------------------------
# GitHub Pages serves 404.html for a missing path at any depth, so its links
# must be absolute from the site root. Build with
# `-D notfound_urls_prefix=/` to preview it from a local server.
notfound_urls_prefix = "/EEGTable/"

# ---------------------------------------------------------------------------
# HTML output — furo theme
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "EEGTable"
html_baseurl = "https://joshuaduq.github.io/EEGTable/"
html_static_path = ["_static", "../assets/branding"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js", "navigation.js"]
html_permalinks_icon = "#"
html_show_sphinx = False
pygments_style = "friendly"
pygments_dark_style = "monokai"
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
html_favicon = "_static/favicon.svg"

html_theme_options = {
    "light_logo": "eegtable-logo.svg",
    "dark_logo": "eegtable-logo-dark.svg",
    "dark_css_variables": {
        "color-background-primary": "#171c21",
        "color-background-secondary": "#20272d",
        "color-background-hover": "#293239",
        "color-background-border": "#3a454e",
        "color-foreground-primary": "#e5e9ec",
        "color-foreground-secondary": "#bec8cf",
        "color-foreground-muted": "#a4b0b9",
        "color-foreground-border": "#6e7d89",
        "color-brand-primary": "#8dd9bd",
        "color-brand-content": "#8dd9bd",
        "color-brand-visited": "#8dd9bd",
        "color-highlight-on-target": "#293239",
        "color-highlighted-background": "#315a45",
        "color-admonition-background": "#20272d",
        "color-api-name": "#8dd9bd",
        "color-api-pre-name": "#a4b0b9",
        "color-api-background": "#20272d",
        "color-api-background-hover": "#293239",
        "color-accent-surface": "#243e34",
        "color-on-accent": "#171c21",
    },
    "light_css_variables": {
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f6f7f8",
        "color-background-hover": "#edf1f2",
        "color-background-border": "#dce1e5",
        "color-foreground-primary": "#202b33",
        "color-foreground-secondary": "#4a5660",
        "color-foreground-muted": "#65717a",
        "color-foreground-border": "#a4afb8",
        "color-brand-primary": "#08745c",
        "color-brand-content": "#08745c",
        "color-brand-visited": "#08745c",
        "color-highlight-on-target": "#eaf3ee",
        "color-highlighted-background": "#d1eadb",
        "color-admonition-background": "#f6f7f8",
        "color-api-name": "#08745c",
        "color-api-pre-name": "#65717a",
        "color-api-background": "#f6f7f8",
        "color-api-background-hover": "#edf1f2",
        "color-accent-surface": "#eaf3ee",
        "color-on-accent": "#ffffff",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": False,
    "top_of_page_buttons": ["view"],
    "source_repository": "https://github.com/JoshuaDuq/EEGTable",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_context = {
    "brand_preview_url": html_baseurl + "_static/github-social-preview.png",
    "companion_pages": {
        **{
            f"methods/{name}": (f"api/{name}", "API reference")
            for name in ("spectral", "dynamics", "connectivity", "complexity")
        },
        **{
            f"api/{name}": (f"methods/{name}", "Method definitions")
            for name in ("spectral", "dynamics", "connectivity", "complexity")
        },
        "api/containers": ("concepts", "Data concepts"),
        "api/model": ("guides/modeling", "Modeling guide"),
        "api/preprocessing": ("guides/preprocessing", "Preprocessing guide"),
        "guides/modeling": ("api/model", "API reference"),
        "guides/preprocessing": ("api/preprocessing", "API reference"),
    }
}


# notfound rewrites the theme's asset and sidebar URLs but leaves links in the
# page body relative, so the 404 page's own links break under a nested path.
def _root_404_links(app, doctree, docname):
    if docname != app.config.notfound_pagename:
        return
    from docutils import nodes

    prefix = app.config.notfound_urls_prefix or "/"
    for node in doctree.findall(nodes.reference):
        uri = node.get("refuri", "")
        if uri and not uri.startswith(("#", "/")) and "://" not in uri:
            node["refuri"] = prefix + uri


def setup(app):
    app.connect("doctree-resolved", _root_404_links)
