.. _overview:

########
Overview
########

``gasimage`` is an experimental python package for post-processing hydrodynamic simulations to produce mock observations of gas.

This module was primarily designed to produce mock position-position-velocity cubes of 21 cm HI gas (under the assumption of optically thin radiative transfer).
The underlying design can be extended to other regimes.

############
Installation
############

The primary dependencies of this packages are:

- `cython <https://cython.org/>`_
- `unyt <https://github.com/yt-project/unyt>`_
- `yt <https://yt-project.org/>`_

Optional dependencies include:

- `astropy <https://www.astropy.org/>`_ (for ``FITS``-file manipulation)
- `mpi4py <https://github.com/mpi4py/mpi4py/>`_ (for parallelization)
- `pytest <https://docs.pytest.org>`_ (for testing)
- `schwimmbad <https://github.com/adrn/schwimmbad>`_ (for parallelization)

After installing these dependencies and cloning ``gasimage`` from the git-repository, you can install ``gasimage`` by invoking the following snippet from the root of the ``gasimage`` directory with:

.. code-block::

   python setup.py develop

Alternatively, you can install it with ``pip``, by invoking the following (again from the root of the ``gasimage`` directory):

.. code-block::

   pip install -e .

************
A quick note
************

If you want to parallelize the ``gasimage`` operations on a cluster and you're using ``conda`` to manage your python installation some care needs to be taken.
Specifically, you should avoid using ``conda`` to install ``mpi4py`` (when you do this ``conda`` tends to install it's own versions of the MPI libraries and links ``mpi4py`` agains those versions).
Instead, you should use ``pip`` to install ``mpi4py`` (this will help ensure ``mpi4py`` is linked against the MPI-libraries provided by the cluster).
