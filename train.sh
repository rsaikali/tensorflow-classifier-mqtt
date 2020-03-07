#!/bin/bash

find . -name ".DS_Store" -delete

mkdir -p ./model

python train.py

echo "################################################################################"
echo "Converting model to .tflite (Tensorflow Lite)..."
tflite_convert --output_file=./model/model.tflite --saved_model_dir=./model
echo "################################################################################"