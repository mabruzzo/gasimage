.. _testing:

#######
Testing
#######

A handful of unit-tests and answer-tests are used to minimize unexpected breakages between different versions of the code.

- Unit-tests try to test different areas of the code in isolation.

- we define "Answer-tests" in a similar manner to the `yt-project <https://yt-project.org/doc/developing/testing.html#answer-testing>`_, `trident <https://trident.readthedocs.io/en/latest/testing.html>`_. Essentially, this is a form of regression-testing where we compare results from calculations from the current version of the code-base (that makes use of real data) against previously computed results using a known good version of the code.

.. _prerequisites:

*************
Prerequisites
*************

1. Install the `pytest <https://docs.pytest.org>`_ package. 

2. Create a directory, where test input data will be stored.
   For example, I might call this directory `~/gasimage-test-data`.
   Download the test data from the provided link into the input data-directory (**TODO: provide a link -- for now, ask me**) and untar any of the files.

***********************
Running the basic tests
***********************

TO run the non-answer tests, you can execute

.. code-block:: bash

   pytest


from within the root ``gasimage`` directory.

There may be some non-answer tests that require data that you downloaded as part of step 2 of the :ref:`prerequisites`.
These tests will be skipped unless you invoke ``pytest`` with the following flag:

.. code-block:: bash

   pytest --indata-dir=<test-input-data>

where ``<test-input-data>`` is the name of the directory containing the downloaded data.



Some extra flags
================

``pytest`` provides some extra flags that can be very useful when debugging a failing test.
These flags can always be appended with the other flags listed here.

Whenever you are running ``pytest`` you can append some useful estra flags like the following:

- ``-s``: ordinarily, ``pytest`` captures output and error messages of tests as they're running.
  This flag tells ``pytest`` not to do that.

- ``-v``: this will cause ``pytest`` to produce a slightly more informative summary.
  It will explicitly list the names of any tests that failed or were skipped (and will provide the reason that the tests were skipped)

- ``-k``: this flag is followed by an expression. When present, ``pytest`` will only run tests matching the expression.


***************************
Generating the answer tests
***************************

From a known good version of the python module (and a known good-version of its dependencies), you can collect save answers to the answer-tests by invoking the following command from within the root ``gasimage`` directory:

.. code-block:: bash

    pytest --indata-dir=<test-input-data> --save-answer-dir=<my-test-answers>

In this command,

- ``<test-input-data>`` is the path to the directory that you created in step 2 of the prerequistes section

- ``<my-test-answers>`` is a path to a directory where the test-answers will be stored. (This directory doesn't need to exist yet)

**NOTE:** At the moment, we're being a little relaxed about what a good known version is. This should be addressed.

**NOTE:** the non-answer tests are currently run when the above command is invoked.
It might be useful to disable those non-answer tests in the future.

************************
Running the answer tests
************************

After you have generated the test answers, and you can now checkout and rebuild the latest version of ``gasimage``.
You can now invoke ``pytest``, again from the root of the ``gasimage`` directory.
If you previously saved the generated test-answers to the ``<my-test-answers>`` directory, you then you should invoke:

.. code-block:: bash

   pytest --indata-dir=<test-input-data> --ref-answer-dir=<my-test-answers>

**NOTE**: Once again, the non-answer tests will also be invoked.


************
Weird Quirks
************

Throughout this page, we discussed some long command line options (i.e. options that are prefixed with ``--``, rather that ``-``) that can be passed to ``pytest``.
All of these flags are used to specify configuration values.
While it should be possible to separate the option from its value with a space (e.g. ``--my-opt value``), some weird behavior appears to occur in these cases.
It's often more reliable to use an ``=`` instead of a space (e.g. ``--my-opt=value``).

It's not totally clear why this happens... (maybe it has to do with the fact that the options usually take directory names as arguments?)
