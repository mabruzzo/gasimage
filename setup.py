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
    Extension('gasimage.ray_traversal._ray_intersections_cy',
              ['gasimage/ray_traversal/_ray_intersections_cy.pyx'],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension('gasimage.ray_traversal._yt_grid_traversal_cy',
              ['gasimage/ray_traversal/_yt_grid_traversal_cy.pyx'],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              language = 'c++'
    ),
    Extension('gasimage._generate_spec_cy',
              ['gasimage/_generate_spec_cy.pyx'],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    #Extension('gasimage.utilts._ArrayDict_cy',
    #          ['gasimage/utils/_ArrayDict_cy.pyx']),
]

# on some platforms, we need to apply the language level directive before setup
# (see https://stackoverflow.com/a/58116368)
for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

# TODO:
# - fetch version number from __init__.py to make sure it remains consistent.
# - Maybe follow instructions from: https://packaging.python.org/en/latest/guides/single-sourcing-package-version/#single-sourcing-the-package-version
VERSION = "0.1.0"

setup(
    name='gasimage',
    version=VERSION,
    description='Toolkit for creating mock HI images.',
    author='Matthew Abruzzo',
    author_email='matthewabruzzo@gmail.com',
    setup_requires = ['numpy', 'cython'],
    packages=find_packages(exclude = ['tests']),
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
