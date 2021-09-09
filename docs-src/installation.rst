Installation instructions
=========================

BioSimulators-AMICI is available as a command-line program and as a command-line program encapsulated into a Docker image.

Command-line program
--------------------

After installing `Python <https://www.python.org/downloads/>`_ (>= 3.7), `pip <https://pip.pypa.io/>`_, `libATLAS <http://math-atlas.sourceforge.net/>`_, g++, and `SWIG <https://www.swig.org/>`_
run the following command to install BioSimulators-AMICI:

.. code-block:: text

    pip install biosimulators-amici


Docker image with a command-line entrypoint
-------------------------------------------

After installing `Docker <https://docs.docker.com/get-docker/>`_, run the following command to install the Docker image for BioSimulators-AMICI:

.. code-block:: text

    docker pull ghcr.io/biosimulators/amici
