FROM python:3.10-slim

ARG VERSION=0.10

RUN apt-get -y update && apt-get install -y --no-install-recommends build-essential cmake git unzip

RUN pip install -U pip && \
    pip install 'git+https://github.com/goeckslab/model-unpickler.git' && \
    pip install 'git+https://github.com/goeckslab/smart-report.git@17df590f3ceb065add099f37b4874c85bd275014' && \
    pip install 'ludwig[full]'==$VERSION && \
    pip cache purge

RUN apt-get purge -y build-essential cmake && apt-get -y autoremove && apt-get clean

