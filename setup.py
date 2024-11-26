# !/usr/bin/env python

from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='nabqr',
    packages=[],
    version='0.0.6',
    description='NABQR is a method for sequential error-corrections tailored for wind power forecast in Denmark',
    author='Bastian S. Jørgensen',
    license='MIT',
    author_email='bassc@dtu.dk',
    url='https://github.com/bast0320/nabqr',
    keywords=['nabqr', 'energy', 'quantile', 'forecasting', ],
    package_dir={'': 'src'},
    py_modules=['nabqr', 'visualization', 'functions', 'helper_functions', 'functions_for_TAQR'],
    python_requires='>=3.10',
    long_description=long_description,
    long_description_content_type="text/markdown",
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
