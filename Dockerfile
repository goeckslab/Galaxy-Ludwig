FROM python:3.8-slim

RUN apt-get -y update && apt-get install -y --no-install-recommends build-essential cmake git unzip

RUN export HOROVOD_WITH_PYTORCH=1 && \
    pip install -U pip && \
    pip install 'torch' 'git+https://github.com/goeckslab/model-unpickler.git' && \
    pip install 'horovod[pytorch]' 'ludwig[full]==0.4.1'&& \
    pip cache purge

RUN apt-get purge -y build-essential cmake && apt-get -y autoremove && apt-get clean
