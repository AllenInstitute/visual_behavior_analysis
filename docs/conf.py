import os
import sys

# Make the package importable during docs build
sys.path.insert(0, os.path.abspath('..'))

project = 'visual_behavior_analysis'
copyright = '2024, Allen Institute for Brain Science'
author = 'Marina Garrett, Doug Ollerenshaw, and contributors'
release = '0.13.3'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',    # NumPy/Google-style docstring support
    'sphinx.ext.viewcode',    # "View source" links in API docs
    'sphinx.ext.intersphinx', # Cross-links to numpy, pandas, etc.
    'myst_parser',            # Markdown support (for including README)
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# autodoc: document members in source order, include __init__ docstrings
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
}
autoclass_content = 'both'

# Napoleon settings (for NumPy-style docstrings used in this codebase)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx: link to external package docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}

# MyST parser options (for Markdown files)
myst_enable_extensions = ['colon_fence']
