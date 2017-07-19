# -*- coding: utf-8 -*-
"""
The setup script for the entire project.
@author: Christoph Lassner
"""
from setuptools import setup, find_packages

VERSION = '0.2'

setup(
    name='clustertools',
    author='Christoph Lassner',
    author_email='mail@christophlassner.de',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
        'aenum',  # python 3: enum34
        'click',
        'scikit-image',
        'pillow'
    ],
    entry_points='''
    [console_scripts]
    csubmit=clustertools.scripts.csubmit:cli
    encaged=clustertools.scripts.encaged:cli
    tfrcat=clustertools.scripts.tfrcat:cli
    tfrpack=clustertools.scripts.tfrpack:cli
    visualize_pose=clustertools.scripts.visualize_pose:cli
    ''',
    version=VERSION,
    license='MIT License',
)
