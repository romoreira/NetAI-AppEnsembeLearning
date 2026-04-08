# Asynchronous Probability Ensembling for Federated Disaster Detection

This repository implements the distributed ensemble learning framework described in our paper "Asynchronous Probability Ensembling for Federated Disaster Detection". The system enables heterogeneous CNN models to collaborate asynchronously through probability-level aggregation over MQTT, achieving competitive accuracy with orders-of-magnitude lower communication costs compared to traditional Federated Learning.

## Repository Branches

This repository contains multiple branches with different implementations and configurations:

- **[master](https://github.com/romoreira/NetAI-AppEnsembeLearning/tree/master)** - Main branch with stable implementation
- **[mqtt](https://github.com/romoreira/NetAI-AppEnsembeLearning/tree/mqtt)** - MQTT-based probability aggregation implementation
- **[mqtt-updates](https://github.com/romoreira/NetAI-AppEnsembeLearning/tree/mqtt-updates)** - Enhanced MQTT implementation with additional features

Explore all branches to see different experimental configurations and implementations.

## Overview

Traditional Federated Learning approaches face significant challenges in disaster response scenarios: high communication overhead from exchanging model weights, rigid synchronization requirements unsuitable for intermittent connectivity, and limited support for heterogeneous model architectures.

Our approach addresses these limitations by exchanging lightweight class probability vectors instead of model parameters, supporting asynchronous client participation, and enabling diverse CNN architectures (EfficientNet, ResNet, MobileNet variants, SqueezeNet) to collaborate effectively.

## Key Features

- **Probability-Level Aggregation**: Clients publish softmax probability vectors via MQTT rather than model weights
- **Asynchronous Training**: No blocking on slow or disconnected clients
- **Multiple Aggregation Strategies**: Logistic Regression Stacking, Genetic Algorithm (GA), Particle Swarm Optimization (PSO)
- **Knowledge Distillation Feedback Loop**: Server broadcasts ensemble predictions back to clients for local refinement
- **Communication Efficiency**: Reduces network traffic by 3+ orders of magnitude compared to federated parameter exchange
- **Architectural Heterogeneity**: Different CNN backbones can participate in the same ensemble

## System Architecture

The framework operates in four phases:

1. **Local Training**: Clients train models independently using their own architectures
2. **Probability Publishing**: Clients publish softmax vectors to MQTT broker
3. **Server Aggregation**: Server collects probabilities and applies stacking/optimization methods
4. **Feedback Distribution**: Server returns ensemble probabilities for knowledge distillation


## Repository Structure

```
el-mqtt/
├── client_el-mqtt.py      # MQTT client (trains model + publishes probabilities)
├── server_el-mqtt.py      # MQTT server (subscribes + performs aggregation)
├── run_el-mqtt.sh         # Launch script for distributed training
├── data/                  # Dataset directory
└── README.md
```


## Environment Setup

Create the conda environment with required dependencies:

```bash
conda create -n torch python=3.12
conda activate torch
pip install torch torchvision scikit-learn pandas paho-mqtt tqdm matplotlib seaborn hyperopt
```

For GPU support, ensure you have the appropriate CUDA-enabled PyTorch build.

## Running the System

### 1. Start MQTT Broker

Using Docker:
```bash
docker run -it -p 1883:1883 eclipse-mosquitto
```

Or use your existing MQTT broker infrastructure.

### 2. Launch Distributed Training

```bash
chmod +x run_el-mqtt.sh
./run_el-mqtt.sh
```

This script orchestrates multiple client instances with different CNN architectures and a central MQTT server for probability aggregation.

### 3. Manual Client Launch

For custom configurations:

```bash
python3 client_el-mqtt.py \
  --broker <broker_ip> \
  --port 1883 \
  --topic probs \
  --model_name resnet34 \
  --optimizer adam \
  --lr 0.0005 \
  --epochs 50 \
  --batch_size 64 \
  --client_id 1
```

### 4. Manual Server Launch

```bash
python3 server_el-mqtt.py \
  --broker <broker_ip> \
  --port 1883 \
  --topic probs \
  --expected_clients 5
```

## Supported Models

The framework supports multiple CNN architectures:
- EfficientNet-B0
- ResNet-34
- MobileNetV2
- MobileNetV3-Small
- SqueezeNet1.0

Each model can be independently configured with different hyperparameters using Tree-structured Parzen Estimator (TPE) optimization.

## Aggregation Methods

### Stacking
Concatenates probability vectors from all models and trains a logistic regression meta-classifier.

### Genetic Algorithm (GA)
Evolves optimal combination weights through evolutionary operators:
- Population size: 40
- Generations: 100
- Mutation rate: 0.3
- Elitism: Top 5 preserved

### Particle Swarm Optimization (PSO)
Optimizes combination weights using swarm intelligence:
- Swarm size: 20
- Iterations: 100
- Inertia coefficient: 0.7
- Cognitive/social factors: 1.5

## Experimental Results

Our experiments on the AIDER disaster dataset demonstrate:
- **Accuracy**: Comparable or superior to traditional FL (up to 98.22%)
- **Communication Efficiency**: ~150 KB vs. 250+ MB per model in standard FL
- **Scalability**: Linear growth O(N·C) vs. O(N·P) in parameter-based approaches
- **Robustness**: Consistent performance improvements across all aggregation methods

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{martins2025asynchronous,
  title={Asynchronous Probability Ensembling for Federated Disaster Detection},
  author={Martins, Emanuel Teixeira and Moreira, Rodrigo and Moreira, Larissa Ferreira Rodrigues and Villa{\c{c}}a, Rodolfo S. and Neto, Augusto and Silva, Fl{\'a}vio de Oliveira},
  booktitle={IEEE Conference},
  year={2025}
}
```

## Authors

**Emanuel Teixeira Martins**, **Rodrigo Moreira**, **Larissa Ferreira Rodrigues Moreira**  
Federal University of Viçosa (UFV), Rio Paranaíba, MG, Brazil

**Rodolfo S. Villaça**  
Federal University of Espírito Santo (UFES), Vitória, ES, Brazil

**Augusto Neto**  
Federal University of Rio Grande do Norte (UFRN), Natal, RN, Brazil

**Flávio de Oliveira Silva**  
University of Minho (UMinho), Braga, Portugal

## Acknowledgments

This research was supported by CNPq (National Council for Scientific and Technological Development) under grant 421944/2021-8 (CNPq/MCTI/FNDCT 18/2021), FAPEMIG (Minas Gerais Research Foundation) grant APQ00923-24, FAPESP (São Paulo Research Foundation) MCTIC/CGI Research Project 2018/23097-3 (SFI2 - Slicing Future Internet Infrastructures), FCT (Fundação para a Ciência e Tecnologia) R&D Unit Project Scope UID/00319/Centro ALGORITMI (ALGORITMI/UM), FAPES (Espírito Santo Research Foundation) grant 2023-RWXSZ, and CAPES (Coordination for the Improvement of Higher Education Personnel).

## License

MIT © The Authors
