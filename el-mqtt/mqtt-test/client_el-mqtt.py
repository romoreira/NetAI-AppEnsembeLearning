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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os

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
parser.add_argument('--ensemble_method', type=str, required=True)
parser.add_argument('--combo_name', type=str, required=True, help='Identifier for the model combination') # NOVO
args = parser.parse_args()

if args.ensemble_method == "baseline":
    print("[INFO] Executando baseline.py...")
    os.system(f"python3 baseline.py --model_name {args.model_name} --optimizer {args.optimizer} --lr {args.lr} --epochs {args.epochs} --batch_size {args.batch_size}")
    

# MQTT config
MQTT_BROKER = args.broker
MQTT_PORT = args.port
client_str = f"client{args.client_id}"
MQTT_TOPIC = f"{client_str}/probs"

print(f"[CONFIG] Broker: {MQTT_BROKER}, Porta: {MQTT_PORT}, Tópico: {MQTT_TOPIC}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Usando dispositivo: {device}")
num_classes = 5

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root="../AIDER_split/train", transform=train_transform)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder(root="../AIDER_split/val", transform=val_transform)

num_classes = train_dataset.classes.__len__()

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

def get_model(name, num_classes):
    name = name.lower()

    weights = 'IMAGENET1K_V1'

    print(f"[MODEL] Carregando modelo: {name}")
    if name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet34":
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "alexnet":
        model = models.alexnet(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg16":
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg19":
        model = models.vgg19(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "squeezenet":
        model = models.squeezenet1_0(weights=weights)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
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
train_losses = []
train_accuracies = []

for epoch in range(args.epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(train_loader, desc=f"{args.client_id} - Epoch {epoch+1}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / len(train_loader)
    acc = correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(acc)
    print(f"[TRAIN] Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

print("[TRAIN] Treinamento concluído.")

# Salvar gráficos
os.makedirs("results", exist_ok=True)
results_path = f"results/{args.combo_name}/{args.ensemble_method}"
os.makedirs(results_path, exist_ok=True)

plt.figure()
plt.plot(train_losses, marker='o', label="Loss")
plt.plot(train_accuracies, marker='x', label="Accuracy")
plt.title(f"Training Metrics - Client {args.client_id} ({args.model_name})")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
fig_path = f"{results_path}/metrics_client{args.client_id}_{args.model_name}.png"
plt.savefig(fig_path)
print(f"[LOG] Gráfico salvo em {fig_path}")

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
client.loop_start()
info = client.publish(MQTT_TOPIC, payload, qos=2)
print(f"📤 [DEBUG] Mensagem publicada. ID: {info.mid}")
info.wait_for_publish()
client.loop_stop()
print(f"📤 [DEBUG] Publicação finalizada. Status: {info.rc}")
client.disconnect()
print("🔌 [DEBUG] Cliente desconectado do broker.")