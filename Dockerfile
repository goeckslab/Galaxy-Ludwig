FROM python:3.8-slim

RUN apt-get -y update && apt-get install -y --no-install-recommends build-essential cmake git unzip

RUN git clone https://github.com/ludwig-ai/ludwig.git && cd ludwig && \
    pip install -U pip &&\
    pip install -e '.[full]' &&\
    pip cache purge

RUN apt-get purge -y build-essential cmake && apt-get -y autoremove && apt-get clean
