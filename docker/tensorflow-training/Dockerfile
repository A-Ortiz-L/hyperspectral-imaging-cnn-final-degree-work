FROM python:3.7-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk2.0-dev \
    gcc \
    nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/
COPY /docker/tensorflow-training/requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

RUN mkdir data
COPY /model /app/model
ENV PYTHONPATH=$PYTHONPATH:/app/
COPY ./config config
COPY ./src src

