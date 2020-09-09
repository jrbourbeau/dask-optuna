#!/usr/bin/env python

from os.path import exists
from setuptools import setup, find_packages

import versioneer

setup(
    name="dask-optuna",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    description="Scaling Optuna with Dask",
    url="https://github.com/jrbourbeau/dask-optuna",
    license="MIT",
    author="James Bourbeau",
    keywords="optuna,dask,distributed",
    include_package_data=True,
    zip_safe=False,
    long_description=(open("README.md").read() if exists("README.md") else ""),
    long_description_content_type="text/markdown",
    install_requires=list(open("requirements.txt").read().strip().split("\n")),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    project_urls={
        "Documentation": "https://jrbourbeau.github.io/dask-optuna",
        "Source Code": "https://github.com/jrbourbeau/dask-optuna",
        "Issue Tracker": "https://github.com/jrbourbeau/dask-optuna/issues",
    },
    python_requires=">=3.6",
)
