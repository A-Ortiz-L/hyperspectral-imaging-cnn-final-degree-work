FROM python:3.6-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk2.0-dev

WORKDIR /app/
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
ENV PYTHONPATH=$PYTHONPATH:/app/
COPY ./src src

