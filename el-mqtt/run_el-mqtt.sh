#!/bin/bash

# Obtém e mata todos os processos que contêm os nomes especificados
pkill -f "client_el-mqtt.py"
pkill -f "server_el-mqtt.py"
sleep 4


python3 client_el-mqtt.py --broker 10.96.221.15 --port 1883 --topic probs --model_name alexnet --optimizer sgd --lr 0.0005   --epochs 1   --batch_size 64 --client_id 1 &
python3 client_el-mqtt.py --broker 10.96.221.15 --port 1883 --topic probs --model_name resnet18 --optimizer adam --lr 0.0005   --epochs 1   --batch_size 64 --client_id 2 & 



python3 server_el-mqtt.py --broker 10.96.221.15 --port 1883 --topic probs --expected_clients 2 &
sleep 2
wait
