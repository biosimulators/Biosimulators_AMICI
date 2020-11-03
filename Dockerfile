# Base OS
FROM python:3.7.9-slim-buster

# metadata
LABEL base_image="python:3.7.9-slim-buster"
LABEL version="1.0.0"
LABEL software="AMICI"
LABEL software.version="0.11.8"
LABEL about.summary="AMICI provides an interface for the SUNDIALS solvers CVODES (for ordinary differential equations) and IDAS (for algebraic differential equations)."
LABEL about.home="https://github.com/AMICI-dev/AMICI"
LABEL about.documentation="https://amici.readthedocs.io/"
LABEL about.license_file="https://github.com/AMICI-dev/AMICI/blob/master/LICENSE.md"
LABEL about.license="BSD-3-Clause"
LABEL about.tags="BioSimulators,mathematical model,kinetic model,simulation,systems biology,computational biology,SBML,SED-ML,COMBINE,OMEX"
LABEL extra.identifiers.biotools="AMICI"
LABEL maintainer="BioSimulators Team <info@biosimulators.org>"

# Install requirements
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        g++ \
        libatlas-base-dev \
        swig \
    && pip3 install -U pip \
    && pip3 install -U setuptools \
    && apt-get remove -y \
        g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy code for command-line interface into image and install it
COPY . /root/biosimulators_amici
RUN pip3 install /root/biosimulators_amici

# Entrypoint
ENTRYPOINT ["amici"]
CMD []
