FROM python:3.7-slim-buster

ENV WEBCAM_URL http://webcam.local:8080
ENV WEBCAM_CHECK_EVERY 1

ENV MQTT_SERVICE_HOST mosquitto.local
ENV MQTT_SERVICE_PORT 1883
ENV MQTT_SERVICE_TOPIC home/garage/door
ENV MQTT_CLIENT_ID tensorflow-classifier-mqtt-service

ENV PYTHONPATH /usr/lib/python3/dist-packages

RUN apt-get update && \
    apt-get install -y gfortran python3-opencv && \
    rm -rf /var/lib/apt/lists/*

RUN ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future pip3 install --upgrade numpy && \
    pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl

RUN pip3 install paho-mqtt

WORKDIR /opt

COPY ./model/model.tflite /opt/model.tflite
COPY ./model/model.labels /opt/model.labels
COPY classify.py /opt/classify.py

ENTRYPOINT ["python", "/opt/classify.py"]