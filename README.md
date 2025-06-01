# 🧠 Distributed Ensemble Learning over MQTT with PyTorch

This project implements a **distributed ensemble learning** architecture using **MQTT** as the communication protocol. Each client trains its own deep learning model on CIFAR-10 and publishes **class probabilities**, while a central **MQTT server** collects and aggregates them via **stacking (logistic regression)**.

---

## ⚙️ Overview

- Each **client** trains a separate model (`ResNet18`, `AlexNet`, etc.).
- After training, it publishes softmax **probabilities and labels** to a unique MQTT topic.
- The **server** listens on wildcard topics, waits for all expected clients, and performs **stacking** using logistic regression.

---

## 📦 Environment (Conda)

You can recreate the exact Python environment using:

```bash
conda create -n torch python=3.12
conda activate torch
pip install torch torchvision flwr scikit-learn pandas paho-mqtt tqdm matplotlib seaborn hyperopt
```

If you prefer, generate and use an `environment.yml` from this environment snapshot.

OBS: Emanuel added hperopt in the required dependencies.

---

## 🗂 Project Structure

```
el-mqtt/
├── client_el-mqtt.py      # MQTT client (training + publishing)
├── server_el-mqtt.py      # MQTT server (subscribes + aggregates)
├── run_el-mqtt.sh         # Script to launch everything
└── data/                  # CIFAR-10 will be downloaded here
```

---

## 🚀 Running the System

### 1. Start MQTT Broker

Run Mosquitto locally:

```bash
docker run -it -p 1883:1883 eclipse-mosquitto
```

Or, use your existing broker (e.g., in Kubernetes). Use the correct broker IP in `--broker`.

---

### 2. Launch Training & Aggregation

```bash
chmod +x run_el-mqtt.sh
./run_el-mqtt.sh
```

This script will:

- Kill any running server/client
- Start `client1` (e.g. AlexNet) and `client2` (e.g. ResNet18)
- Start the MQTT **server**
- Display all logs live

---

## 🧪 Test: Publish Manually

To verify MQTT communication:

```python
# publish_test.py
import paho.mqtt.client as mqtt
broker = "your_broker_ip"
client = mqtt.Client()
client.connect(broker, 1883, 60)
client.publish("client1/probs", "Hello MQTT!")
client.disconnect()
```

---

## 🧠 Key Concepts

- **MQTT Topics**: Each client publishes to `clientX/probs`.
- **Server Subscription**: The server subscribes to `+/probs` to receive from any client.
- **Aggregation**: Once all expected clients send data, stacking is done via `LogisticRegression`.
- **Disconnection**: Clients disconnect after publishing. Server remains active using `loop_start()`.

---

## 🔧 Manual Example

```bash
python3 client_el-mqtt.py \
  --broker 10.96.221.15 \
  --port 1883 \
  --topic probs \
  --model_name resnet18 \
  --optimizer adam \
  --lr 0.0005 \
  --epochs 1 \
  --batch_size 64 \
  --client_id 1
```

And start the server:

```bash
python3 server_el-mqtt.py \
  --broker 10.96.221.15 \
  --port 1883 \
  --topic probs \
  --expected_clients 2
```

---

## ✅ Features

- ✅ Lightweight messaging with MQTT
- ✅ Supports any number of clients
- ✅ Customizable models and hyperparameters
- ✅ Server-side stacking (centralized ensemble)

---

## 📄 License

MIT © Rodrigo Moreira
