FROM python:3.8-slim

ARG VERSION=0.6.1

RUN apt-get -y update && apt-get install -y --no-install-recommends build-essential cmake git unzip

RUN export HOROVOD_WITH_PYTORCH=1 && \
    pip install -U pip && \
    pip install 'torch' 'git+https://github.com/goeckslab/model-unpickler.git' 'git+https://github.com/goeckslab/smart-report.git' && \
    pip install 'horovod[pytorch]' && \
    pip install 'ludwig[full]'==$VERSION && \
    pip cache purge

RUN apt-get purge -y build-essential cmake && apt-get -y autoremove && apt-get clean
