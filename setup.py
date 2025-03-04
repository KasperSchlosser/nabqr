# !/usr/bin/env python

from setuptools import setup, find_namespace_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

# Documentation requirements
docs_require = [
    'sphinx>=4.0.0',
    'sphinx-rtd-theme',
    'sphinx-autodoc-typehints',
    'autodocsumm',
    'sphinx_mdinclude',
]

setup(
    name='nabqr',
    version='0.0.46',
    description='NABQR is a method for sequential error-corrections tailored for wind power forecast in Denmark',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Bastian S. Jørgensen',
    author_email='bassc@dtu.dk',
    url='https://github.com/bast0320/nabqr',
    license='MIT',
    keywords=['nabqr', 'energy', 'quantile', 'forecasting'],
    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src'),
    python_requires='>=3.10',
    install_requires=install_requires,
    extras_require={
        'docs': docs_require,
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development',
    ],
)
