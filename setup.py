#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext
import os

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

def _get_cpp_headers(dir_path):
    def _is_header(dir_entry):
        _, ext = os.path.splitext(dir_entry.name)
        valid_ext = (ext == '.h') or (ext == '.hpp')
        return dir_entry.is_file(follow_symlinks=True) and valid_ext

    with os.scandir(dir_path) as it:
        return sorted(dir_entry.path for dir_entry in filter(_is_header, it))

extra_kwargs = {}
ext_modules = [
    Extension('gasimage.ray_traversal._misc_cy',
              ['gasimage/ray_traversal/_misc_cy.pyx'],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              **extra_kwargs
    ),
    Extension('gasimage.ray_traversal._yt_grid_traversal_cy',
              ['gasimage/ray_traversal/_yt_grid_traversal_cy.pyx'],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
                             ("CYTHON_EXTERN_C", "extern \"C\"")],
              language = 'c++',
              **extra_kwargs
    ),
    Extension('gasimage._generate_spec_cy',
              sources = ['gasimage/_generate_spec_cy.pyx'],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
                             ("CYTHON_EXTERN_C", "extern \"C\"")],
              language = 'c++',
              extra_compile_args = ['--std=c++17'],
              depends = _get_cpp_headers(dir_path = 'gasimage/cpp'),
              **extra_kwargs
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
