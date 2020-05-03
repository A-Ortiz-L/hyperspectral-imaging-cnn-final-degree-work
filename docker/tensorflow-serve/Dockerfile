FROM tensorflow/serving:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    nano \
    libgtk2.0-dev \
    python3-pip \
    python3 \
    python3-dev \
    gcc \
    python-setuptools \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
WORKDIR /app/
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN echo "alias python=python3" >> /etc/bash.bashrc
RUN pip3 install --upgrade pip
RUN pip3 install -U setuptools
COPY /docker/tensorflow-serve/requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN mkdir data
COPY /model/tensorflow_serve/v1 /app/model/1
ENV PYTHONPATH=$PYTHONPATH:/app/
COPY ./config config
COPY ./src src

ENTRYPOINT bash