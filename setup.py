#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="dask-optuna",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
