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

# Argumentos via CLI
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True, help='Modelo: resnet18, alexnet, vgg16...')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizador: adam, sgd')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--epochs', type=int, default=3, help='Número de épocas')
parser.add_argument('--batch_size', type=int, default=32, help='Tamanho do batch')
parser.add_argument('--client_id', type=int, help='ID do cliente MQTT')
args = parser.parse_args()

# MQTT config
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = f"{args.client_id}/probs"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Datasets
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# Modelo
def get_model(name, num_classes):
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Modelo '{name}' não suportado.")
    return model

model = get_model(args.model_name, num_classes).to(device)

# Otimizador
if args.optimizer.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
else:
    raise ValueError(f"Otimizador '{args.optimizer}' não suportado.")

# Treinamento
criterion = nn.CrossEntropyLoss()
model.train()
for epoch in range(args.epochs):
    for inputs, targets in tqdm(train_loader, desc=f"{args.client_id} - Epoch {epoch+1}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Avaliação
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

# Publicação via MQTT
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)
payload = json.dumps({"probs": probs, "labels": labels})
client.publish(MQTT_TOPIC, payload)
client.disconnect()
print(f"{args.client_id} publicou com modelo {args.model_name}, opt {args.optimizer}, lr {args.lr}.")
