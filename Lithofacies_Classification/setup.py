#!/usr/bin/env python
# coding: utf-8


from setuptools import setup

setup(
        name="Lithofacies_Classification",
        version="0.0.2",
        url="https://github.com/Esfahani98/Lithofacies_Classification.git",
        author="Hamid Reza Mousavi",
        license="IUST",
        packages=["Functions", "Models"],
        include_package_data=True,
        install_requires=["numpy"]
)
