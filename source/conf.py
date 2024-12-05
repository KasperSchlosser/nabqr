# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nabqr-RTD'
copyright = '2024, Bastian S. Jørgensen'
author = 'Bastian S. Jørgensen'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []

templates_path = ['_templates']
exclude_patterns = []

# Here we add the automatic documentation generation
import sys
import os
sys.path.insert(0,  os.path.abspath('../NABQR'))

#  add in the extension names to the empty list variable 'extensions'
extensions = [
      'sphinx.ext.autodoc', 
    #   'sphinx.ext.napoleon', 
      'autodocsumm', 
      'sphinx.ext.coverage'
]

# add in this line for the autosummary functionality
auto_doc_default_options = {'autosummary': True}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# simply add the extension to your list of extensions to use markdown files
extensions = ['myst_parser']

source_suffix = ['.rst', '.md']