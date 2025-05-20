#!/bin/bash

python3 client_el-mqtt.py --model_name squeezenet --optimizer adam --lr 0.0005   --epochs 5   --batch_size 64 --client_id 1
python3 client_el-mqtt.py --model_name resnet18 --optimizer adam --lr 0.0005   --epochs 5   --batch_size 64 --client_id 1
