# Base OS
FROM python:3.11-slim-bookworm

ARG VERSION="0.1.22"
ARG SIMULATOR_VERSION="0.18.1"

# metadata
LABEL \
    org.opencontainers.image.title="AMICI" \
    org.opencontainers.image.version="${SIMULATOR_VERSION}" \
    org.opencontainers.image.description="Interface for the SUNDIALS solvers CVODES (for ordinary differential equations) and IDAS (for algebraic differential equations)." \
    org.opencontainers.image.url="https://github.com/AMICI-dev/AMICI" \
    org.opencontainers.image.documentation="https://amici.readthedocs.io/" \
    org.opencontainers.image.source="https://github.com/biosimulators/Biosimulators_AMICI" \
    org.opencontainers.image.authors="BioSimulators Team <info@biosimulators.org>" \
    org.opencontainers.image.vendor="BioSimulators Team" \
    org.opencontainers.image.licenses="BSD-3-Clause" \
    \
    base_image="python:3.9-slim-buster" \
    version="${VERSION}" \
    software="AMICI" \
    software.version="${SIMULATOR_VERSION}" \
    about.summary="Interface for the SUNDIALS solvers CVODES (for ordinary differential equations) and IDAS (for algebraic differential equations)." \
    about.home="https://github.com/AMICI-dev/AMICI" \
    about.documentation="https://amici.readthedocs.io/" \
    about.license_file="https://github.com/AMICI-dev/AMICI/blob/master/LICENSE.md" \
    about.license="SPDX:BSD-3-Clause" \
    about.tags="BioSimulators,mathematical model,kinetic model,simulation,systems biology,computational biology,SBML,SED-ML,COMBINE,OMEX" \
    extra.identifiers.biotools="AMICI" \
    maintainer="BioSimulators Team <info@biosimulators.org>"

# Install requirements
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        g++ \
        libatlas-base-dev \
        swig \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# fonts for matplotlib
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Copy code for command-line interface into image and install it
COPY . /root/Biosimulators_AMICI
RUN pip install pip==23.0.1
RUN pip install sympy /root/Biosimulators_AMICI \
    && rm -rf /root/Biosimulators_AMICI
#RUN pip install sympy /root/Biosimulators_AMICI amici==${SIMULATOR_VERSION} \
#    && rm -rf /root/Biosimulators_AMICI
ENV VERBOSE=0 \
    MPLBACKEND=PDF

# Entrypoint
ENTRYPOINT ["biosimulators-amici"]
CMD []
