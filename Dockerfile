FROM ubtuntu:20.04

RUN apt-get update \
    && apt install libatlas-base-dev swig python3 python3-pip \
    && pip3 install amici
