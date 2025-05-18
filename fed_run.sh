#!/bin/bash

# Obtém e mata todos os processos que contêm os nomes especificados
pkill -f "server_stacking.py"
pkill -f "client_stacking.py"
sleep 4

python3 server_stacking.py --model_name alexnet --ip 127.0.0.1 --trainingUuid 001002 --numRounds 2 &

python3 client_stacking.py --epoch 2 --lr 0.001 --batch_size 32 --client_id 1 --dataset 1 --model_name alexnet &
python3 client_stacking.py --epoch 2 --lr 0.001 --batch_size 32 --client_id 2 --dataset 1 --model_name alexnet &
wait