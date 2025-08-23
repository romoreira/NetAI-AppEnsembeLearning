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

# Argumentos via CLI (mantidos o mais próximo possível)
parser = argparse.ArgumentParser()
parser.add_argument('--broker', type=str, default='localhost')
parser.add_argument('--port', type=int, default=1883)
parser.add_argument('--topic', type=str, default='#')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--optimizer', type=str, default='adam')       # mantido (não usado)
parser.add_argument('--lr', type=float, default=1e-3)              # mantido (não usado)
parser.add_argument('--epochs', type=int, default=3)               # mantido (não usado)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--client_id', type=int, required=True)
parser.add_argument('--ensemble_method', type=str, required=True)  # mantido (não usado)
parser.add_argument('--combo_name', type=str, required=True, help='Identifier for the model combination') # mantido (não usado)
parser.add_argument('--pth_path', type=str, required=True, help='Caminho do arquivo .pth com os pesos do modelo')  # NOVO
args = parser.parse_args()

# NÃO executa baseline/treino — foco apenas em extrair probabilidades de um .pth

# MQTT config
MQTT_BROKER = args.broker
MQTT_PORT = args.port
client_str = f"client{args.client_id}"
MQTT_TOPIC = f"{client_str}/probs"

print(f"[CONFIG] Broker: {MQTT_BROKER}, Porta: {MQTT_PORT}, Tópico: {MQTT_TOPIC}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Usando dispositivo: {device}")

# Somente validação/inferência
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder(root="../AIDER_split/val", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
classes = list(val_dataset.classes)
num_classes = len(classes)
print(f"[DATA] Val: {len(val_dataset)} amostras | classes({num_classes}): {classes}")

def get_model(name, num_classes):
    name = name.lower()

    # Não precisamos de pesos do ImageNet; os pesos virão do .pth
    weights = None

    print(f"[MODEL] Carregando arquitetura: {name}")
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
    elif name == "squeezenet" or name == "squeezenet1_0":
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

def _strip_module_prefix(state_dict):
    if not any(k.startswith('module.') for k in state_dict.keys()):
        return state_dict
    return {k.replace('module.', '', 1): v for k, v in state_dict.items()}

def load_checkpoint_any(model, pth_path, map_location, strict=False):
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {pth_path}")
    ckpt = torch.load(pth_path, map_location=map_location)

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        try:
            state = ckpt.state_dict()  # caso tenham salvo o modelo inteiro
        except Exception as e:
            raise RuntimeError(f"Formato de checkpoint não suportado: {type(ckpt)}") from e

    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return missing, unexpected

model = get_model(args.model_name, num_classes).to(device)

# Carrega pesos do .pth
print(f"[LOAD] Carregando pesos de: {args.pth_path}")
try:
    missing, unexpected = load_checkpoint_any(model, args.pth_path, map_location=device, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict(strict=False): missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:   print(f"       missing (até 10): {missing[:10]}")
        if unexpected:print(f"       unexpected (até 10): {unexpected[:10]}")
except Exception as e:
    raise SystemExit(f"[ERROR] Falha ao carregar pesos: {e}")

# Avaliação: extrai probabilidades
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

# Publicação via MQTT (mesmo payload de antes)
print(f"🚀 [MQTT] Conectando ao broker {MQTT_BROKER}:{MQTT_PORT}")
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

payload = json.dumps({"probs": probs, "labels": labels})
print(f"📡 Enviando para tópico {MQTT_TOPIC}")
client.loop_start()
info = client.publish(MQTT_TOPIC, payload, qos=2)
print(f"📤 [DEBUG] Mensagem publicada. ID: {info.mid}")
info.wait_for_publish()
client.loop_stop()
print(f"📤 [DEBUG] Publicação finalizada. Status: {info.rc}")
client.disconnect()
print("🔌 [DEBUG] Cliente desconectado do broker.")
