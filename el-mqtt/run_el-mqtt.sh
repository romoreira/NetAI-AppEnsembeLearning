#!/bin/bash

# Obtém e mata todos os processos que contêm os nomes especificados
pkill -f "client_el-mqtt.py"
pkill -f "server_el-mqtt.py"
sleep 4


python3 client_el-mqtt.py --broker 10.103.117.206 --port 1883 --topic probs --model_name alexnet --optimizer sgd --lr 0.001   --epochs 1   --batch_size 64 --client_id 1 --ensemble_method pso_stacking &
python3 client_el-mqtt.py --broker 10.103.117.206 --port 1883 --topic probs --model_name resnet18 --optimizer sgd --lr 0.001   --epochs 1   --batch_size 64 --client_id 2 --ensemble_method pso_stacking & 
python3 client_el-mqtt.py --broker 10.103.117.206 --port 1883 --topic probs --model_name mobilenet_v2 --optimizer sgd --lr 0.001   --epochs 1   --batch_size 64 --client_id 3 --ensemble_method pso_stacking & 



python3 server_el-mqtt.py --broker 10.103.117.206 --port 1883 --topic probs --expected_clients 3 --ensemble_method pso_stacking &
sleep 2
wait
