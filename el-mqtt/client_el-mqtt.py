# client.py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import paho.mqtt.client as mqtt
import json
from tqdm import tqdm
import time

# Argumentos via CLI
parser = argparse.ArgumentParser()
parser.add_argument('--broker', type=str, default='localhost')
parser.add_argument('--port', type=int, default=1883)
parser.add_argument('--topic', type=str, default='#')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--client_id', type=int, required=True)
args = parser.parse_args()

# MQTT config
MQTT_BROKER = args.broker
MQTT_PORT = args.port
client_str = f"client{args.client_id}"
MQTT_TOPIC = f"{client_str}/probs"

print(f"[CONFIG] Broker: {MQTT_BROKER}, Porta: {MQTT_PORT}, Tópico: {MQTT_TOPIC}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Usando dispositivo: {device}")
num_classes = 10

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets
print("[DATA] Carregando dataset CIFAR10...")
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# Modelo
def get_model(name, num_classes):
    name = name.lower()
    print(f"[MODEL] Carregando modelo: {name}")
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "squeezenet":
        model = models.squeezenet1_0(weights=None)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    else:
        raise ValueError(f"Modelo '{name}' não suportado.")
    return model

model = get_model(args.model_name, num_classes).to(device)

# Otimizador
print(f"[TRAIN] Inicializando otimizador: {args.optimizer}")
if args.optimizer.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
else:
    raise ValueError(f"Otimizador '{args.optimizer}' não suportado.")

# Treinamento
criterion = nn.CrossEntropyLoss()
print("[TRAIN] Iniciando treinamento...")
model.train()
for epoch in range(args.epochs):
    for inputs, targets in tqdm(train_loader, desc=f"{args.client_id} - Epoch {epoch+1}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
print("[TRAIN] Treinamento concluído.")

# Avaliação
print("[EVAL] Extraindo probabilidades do conjunto de validação...")
model.eval()
all_probs = []
all_labels = []
with torch.no_grad():
    for inputs, targets in tqdm(val_loader, desc="Extracting probs"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1).cpu()
        all_probs.append(probs)
        all_labels.append(targets)

probs = torch.cat(all_probs).tolist()
labels = torch.cat(all_labels).tolist()
print(f"[EVAL] Total de amostras avaliadas: {len(probs)}")

# Publicação via MQTT
print(f"🚀 [MQTT] Conectando ao broker {MQTT_BROKER}:{MQTT_PORT}")
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

payload = json.dumps({"probs": probs, "labels": labels})
print(f"📡 Enviando para tópico {MQTT_TOPIC} com round_id='default'")
info = client.publish(MQTT_TOPIC, payload, qos=1)
print(f"📤 [DEBUG] Mensagem publicada. ID: {info.mid}")
info.wait_for_publish()
print(f"📤 [DEBUG] Publicação finalizada. Status: {info.rc}")
client.disconnect()
print("🔌 [DEBUG] Cliente desconectado do broker.")
