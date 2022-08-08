#!/usr/bin/env python
# coding: utf-8


from setuptools import setup

setup(
        name="Package",
        version="0.0.1",
        url="https://github.com/Esfahani98/Package.git",
        author="Zoraiz Ali",
        license="MIT",
        packages=["Functions", "Models"],
        include_package_data=True,
        install_requires=["numpy"]
)
