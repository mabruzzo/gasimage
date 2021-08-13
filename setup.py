#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='gasimage',
    version='0.0.1',
    description='Toolkit for creating mock HI images.',
    author='Matthew Abruzzo',
    author_email='matthewabruzzo@gmail.com',
    packages=find_packages(exclude = ['tests'])
)
