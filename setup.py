#!/usr/bin/env python
# coding: utf-8


from setuptools import setup

setup(
        name="Lithofacies-Classification",
        version="0.0.1",
        url="https://github.com/Esfahani98/Lithofacies-Classification-from-Well-Logs.git",
        author="Hamid Reza Mousavi",
        license="IUST",
        packages=["Functions", "Models"],
        include_package_data=True,
        install_requires=["numpy"]
)
