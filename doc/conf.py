import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Orange3 NLP"
copyright = "2025, Chris Lee"
author = "Chris Lee"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build"]
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
htmlhelp_basename = "Orange3NLPdoc"
