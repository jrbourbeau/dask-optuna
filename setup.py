#!/usr/bin/env python

from os.path import exists
from setuptools import setup, find_packages

setup(
    name="dask-optuna",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    long_description=(open("README.md").read() if exists("README.md") else ""),
    long_description_content_type="text/markdown",
    install_requires=list(open("requirements.txt").read().strip().split("\n")),
)
