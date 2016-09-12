# -*- coding: utf-8 -*-
"""
The setup script for the entire project.
@author: Christoph Lassner
"""
from setuptools import setup, find_packages

VERSION = '0.1'

setup(
    name='clustertools',
    author='Christoph Lassner',
    author_email='mail@christophlassner.de',
    packages=find_packages(),
    install_requires=[
        'click',
    ],
    entry_points='''
    [console_scripts]

    ''',
    version=VERSION,
    license='MIT License',
)
