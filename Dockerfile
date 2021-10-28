FROM ubuntu:20.04

ENV VERSION=0.4
RUN apt-get update && \
    apt-get install -y git python3.8 python3-pip unzip wget && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN pip install 'ludwig[full]'==$VERSION petastorm