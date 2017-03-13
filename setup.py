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
    include_package_data=True,
    install_requires=[
        'numpy',
        'matplotlib',
        'click',
    ],
    entry_points='''
    [console_scripts]
    csubmit=clustertools.scripts.csubmit:cli
    encaged=clustertools.scripts.encaged:cli
    ''',
    version=VERSION,
    license='MIT License',
)
