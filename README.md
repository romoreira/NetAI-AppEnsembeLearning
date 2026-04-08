# Asynchronous Probability Ensembling for Federated Disaster Detection

This repository implements the distributed ensemble learning framework described in our paper "Asynchronous Probability Ensembling for Federated Disaster Detection". The system enables heterogeneous CNN models to collaborate asynchronously through probability-level aggregation over MQTT, achieving competitive accuracy with orders-of-magnitude lower communication costs compared to traditional Federated Learning.

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
