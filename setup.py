#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    # subclass setuptools extension builder to avoid importing numpy
    # at top level in setup.py. See http://stackoverflow.com/a/21621689/1382869
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process
        # see http://stackoverflow.com/a/21621493/1382869
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

ext_modules = [
    Extension('gasimage._ray_intersections_cy',
              ['gasimage/_ray_intersections_cy.pyx'])
]

# on some platforms, we need to apply the language level directive before setup
# (see https://stackoverflow.com/a/58116368)
for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(
    name='gasimage',
    version='0.0.1',
    description='Toolkit for creating mock HI images.',
    author='Matthew Abruzzo',
    author_email='matthewabruzzo@gmail.com',
    setup_requires = ['numpy', 'cython'],
    packages=find_packages(exclude = ['tests']),
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
