from __future__ import absolute_import, division, print_function

import logging
import os
import time

import cv2
import numpy as np

import paho.mqtt.publish as publish
from tflite_runtime.interpreter import Interpreter

# Config from environment (see Dockerfile)
WEBCAM_URL = os.getenv('WEBCAM_URL', 'http://192.168.1.26:8081')
WEBCAM_CHECK_EVERY = float(os.getenv('WEBCAM_CHECK_EVERY', 1))
MQTT_SERVICE_HOST = os.getenv('MQTT_SERVICE_HOST', None)
MQTT_SERVICE_PORT = int(os.getenv('MQTT_SERVICE_PORT', 1883))
MQTT_SERVICE_TOPIC = os.getenv('MQTT_SERVICE_TOPIC', 'home/garage/door')
MQTT_CLIENT_ID = os.getenv('HOSTNAME', 'tflite-garagedoor-mqtt-service')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(name)s] %(levelname)8s %(message)s')
logger = logging.getLogger(MQTT_CLIENT_ID)

# Display config on startup
logger.debug("#" * 80)
logger.debug(f"# WEBCAM_URL={WEBCAM_URL}")
logger.debug(f"# WEBCAM_CHECK_EVERY={WEBCAM_CHECK_EVERY}")
logger.debug(f"# MQTT_SERVICE_HOST={MQTT_SERVICE_HOST}")
logger.debug(f"# MQTT_SERVICE_PORT={MQTT_SERVICE_PORT}")
logger.debug(f"# MQTT_SERVICE_TOPIC={MQTT_SERVICE_TOPIC}")
logger.debug(f"# MQTT_CLIENT_ID={MQTT_CLIENT_ID}")
logger.debug("#" * 80)

try:
    with open(os.path.join(os.path.dirname(__file__), "model", "model.labels")) as lf:
        labels = [lbl.strip() for lbl in lf.readlines()]

    logger.info(f"Found labels:  %s" % " | ".join(labels))

    model_path = os.path.join(os.path.dirname(__file__), "model", "model.tflite")
    logger.info(f"Opening model: %s" % model_path)
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    _, height, width, _ = input_details['shape']

    logger.info(f"Opening video: %s" % WEBCAM_URL)
    capture = cv2.VideoCapture(WEBCAM_URL)

    logger.info("Entering main loop...")

    while capture.isOpened():

        start_time = time.time()

        # Capture frame from webcam
        image = capture.read()[1]
        # Resize imput image
        img = cv2.resize(image, (height, width), interpolation=cv2.INTER_CUBIC)
        img = np.expand_dims(img, axis=0)
        img = np.float32(img)

        interpreter.set_tensor(input_details['index'], img)
        interpreter.invoke()
        output = np.squeeze(interpreter.get_tensor(output_details['index']))

        score, state = output[np.argmax(output)], labels[np.argmax(output)]
        score_pct = round(score * 100, 4)

        logger.info(f"[predicted_label={state}] [score=%.4f%%] [time=%.2fms] output={output}" % (score_pct, (time.time() - start_time) * 1000))

        if MQTT_SERVICE_HOST is not None:

            msgs = [(f"{MQTT_SERVICE_TOPIC}/state", state),
                    (f"{MQTT_SERVICE_TOPIC}/score", str(score_pct))]

            # Publish door state and score on given MQTT broker
            try:
                publish.multiple(msgs, hostname=MQTT_SERVICE_HOST, port=MQTT_SERVICE_PORT, client_id=MQTT_CLIENT_ID)
            except Exception:
                logger.error("An error occured while trying to publish:", exc_info=True)

        time.sleep(WEBCAM_CHECK_EVERY)

except Exception:
    logger.error("An error occured:", exc_info=True)
    capture.release()
