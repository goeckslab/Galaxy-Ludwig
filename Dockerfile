FROM tensorflow/tensorflow:2.6.1

RUN apt-get -y update && apt-get install -y git cmake zip unzip

RUN pip install 'ludwig[full]==0.4' 'tensorflow==2.6.1' --use-feature=2020-resolver
