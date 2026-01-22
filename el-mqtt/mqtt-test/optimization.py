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
import numpy as np
from sklearn.metrics import classification_report
from threading import Event

# -----------------------------
# Argumentos via CLI
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--broker', type=str, default='localhost')
parser.add_argument('--port', type=int, default=1883)
parser.add_argument('--topic', type=str, default='#')  # legado, mantido
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--optimizer', type=str, default='adam')  # mantido (não usado)
parser.add_argument('--lr', type=float, default=1e-3)        # mantido (não usado)
parser.add_argument('--epochs', type=int, default=3)         # mantido (não usado)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--client_id', type=int, required=True)
parser.add_argument('--ensemble_method', type=str, required=True)   # mantido (não usado)
parser.add_argument('--combo_name', type=str, required=True, help='Identifier for the model combination')
parser.add_argument('--pth_path', type=str, required=True, help='Caminho do arquivo .pth com os pesos do modelo')

# Novos (opcionais)
parser.add_argument('--results_dir', type=str, default='./resultados', help='Diretório base para salvar relatórios')
parser.add_argument('--topic_out', type=str, default=None, help='Tópico para publicar as probabilidades locais')
parser.add_argument('--topic_in', type=str, default=None, help='Tópico para receber as probabilidades agregadas')
parser.add_argument('--wait_timeout_sec', type=int, default=120, help='Timeout para aguardar resposta do MQTT (segundos)')

args = parser.parse_args()

# -----------------------------
# MQTT topics derivados do client_id (se não forem passados)
# -----------------------------
client_str = f"client{args.client_id}"
if args.topic_out is None:
    args.topic_out = f"{client_str}/probs"
if args.topic_in is None:
    args.topic_in = f"{client_str}/stacked"

# -----------------------------
# Config/paths
# -----------------------------
MQTT_BROKER = args.broker
MQTT_PORT = args.port
MQTT_TOPIC_OUT = args.topic_out
MQTT_TOPIC_IN = args.topic_in

print(f"[CONFIG] Broker: {MQTT_BROKER}, Porta: {MQTT_PORT}")
print(f"[CONFIG] Topic OUT: {MQTT_TOPIC_OUT} | Topic IN: {MQTT_TOPIC_IN}")

# results/<combo>/<client>/
results_dir = os.path.join(args.results_dir, args.combo_name, client_str)
os.makedirs(results_dir, exist_ok=True)
print(f"[RESULTS] Diretório de resultados: {results_dir}")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Usando dispositivo: {device}")

# -----------------------------
# Dataset (somente inferência)
# -----------------------------
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

# -----------------------------
# Modelo
# -----------------------------
def get_model(name, num_classes):
    name = name.lower()
    weights = None  # pesos virão do .pth

    print(f"[MODEL] Carregando arquitetura: {name}")
    if name == "resnet18":
        model = models.resnet18(weights=weights); model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet34":
        model = models.resnet34(weights=weights); model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=weights); model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "alexnet":
        model = models.alexnet(weights=weights); model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg16":
        model = models.vgg16(weights=weights); model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg19":
        model = models.vgg19(weights=weights); model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights); model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights); model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights); model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name in ("squeezenet", "squeezenet1_0"):
        model = models.squeezenet1_0(weights=weights); model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1, stride=1); model.num_classes = num_classes
    elif name == "densenet121":
        model = models.densenet121(weights=weights); model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights); model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
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
            state = ckpt.state_dict()
        except Exception as e:
            raise RuntimeError(f"Formato de checkpoint não suportado: {type(ckpt)}") from e

    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return missing, unexpected

model = get_model(args.model_name, num_classes).to(device)

print(f"[LOAD] Carregando pesos de: {args.pth_path}")
try:
    missing, unexpected = load_checkpoint_any(model, args.pth_path, map_location=device, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict(strict=False): missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:   print(f"       missing (até 10): {missing[:10]}")
        if unexpected:print(f"       unexpected (até 10): {unexpected[:10]}")
except Exception as e:
    raise SystemExit(f"[ERROR] Falha ao carregar pesos: {e}")

# -----------------------------
# Inferência (probabilidades locais)
# -----------------------------
print("[EVAL] Extraindo probabilidades do conjunto de validação...")
model.eval()
all_probs = []
all_labels = []
with torch.no_grad():
    for inputs, targets in tqdm(val_loader, desc="Extracting probs"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs_batch = F.softmax(outputs, dim=1).cpu()
        all_probs.append(probs_batch)
        all_labels.append(targets)

student_probs = torch.cat(all_probs)       # [N, C]
labels = torch.cat(all_labels)             # [N]
probs_list = student_probs.tolist()
labels_list = labels.tolist()
print(f"[EVAL] Total de amostras avaliadas: {len(probs_list)}")

# -----------------------------
# Funções auxiliares: relatórios
# -----------------------------
def save_classification_report(y_true, y_proba, class_names, out_txt_path, out_json_path):
    y_pred = np.argmax(y_proba, axis=1)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_str  = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    with open(out_txt_path, 'w', encoding='utf-8') as f:
        f.write(report_str + "\n")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    return y_pred, report_dict

# -----------------------------
# 1) Classification report local (antes do MQTT)
# -----------------------------
local_txt  = os.path.join(results_dir, "local_before_mqtt_report.txt")
local_json = os.path.join(results_dir, "local_before_mqtt_report.json")
_ = save_classification_report(
    y_true=np.array(labels_list),
    y_proba=np.array(probs_list),
    class_names=classes,
    out_txt_path=local_txt,
    out_json_path=local_json
)
# também salvamos os arrays:
np.save(os.path.join(results_dir, "labels.npy"), np.array(labels_list))
np.save(os.path.join(results_dir, "local_probs.npy"), np.array(probs_list))
print(f"[REPORT] Relatório local salvo em: {local_txt}")

# -----------------------------
# 2) Publicação via MQTT
# -----------------------------
print(f"🚀 [MQTT] Conectando ao broker {MQTT_BROKER}:{MQTT_PORT}")
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

payload = json.dumps({"probs": probs_list, "labels": labels_list})
print(f"📡 Enviando para tópico {MQTT_TOPIC_OUT}")
mqtt_client.loop_start()
pub_info = mqtt_client.publish(MQTT_TOPIC_OUT, payload, qos=2)
print(f"📤 [DEBUG] Mensagem publicada. ID: {pub_info.mid}")
pub_info.wait_for_publish()

# -----------------------------
# 3) Aguardar resposta do servidor com probs "teacher"
# -----------------------------
received_event = Event()
teacher_probs_np = None

def on_message(_client, _userdata, msg):
    nonlocal teacher_probs_np
    try:
        data = json.loads(msg.payload.decode('utf-8'))
        if "probs" in data:
            teacher_probs_np = np.array(data["probs"], dtype=np.float32)
            # sanity check shape
            if teacher_probs_np.ndim == 2 and teacher_probs_np.shape[0] == len(labels_list):
                print(f"[MQTT] Recebidas probs teacher: shape={teacher_probs_np.shape}")
                received_event.set()
            else:
                print(f"[WARN] Formato inesperado das probs recebidas: {teacher_probs_np.shape}")
        else:
            print("[WARN] Payload recebido não contém 'probs'.")
    except Exception as e:
        print(f"[ERROR] Falha ao parsear mensagem MQTT: {e}")

mqtt_client.subscribe(MQTT_TOPIC_IN, qos=2)
mqtt_client.on_message = on_message

print(f"📥 [MQTT] Aguardando resposta em '{MQTT_TOPIC_IN}' por até {args.wait_timeout_sec}s...")
ok = received_event.wait(timeout=args.wait_timeout_sec)

mqtt_client.loop_stop()
mqtt_client.disconnect()
print("🔌 [DEBUG] Cliente desconectado do broker.")

if not ok:
    print(f"[TIMEOUT] Não chegaram probabilidades do servidor em {args.wait_timeout_sec}s.")
    # Ainda assim encerramos com os artefatos locais já salvos.
    raise SystemExit(0)

# -----------------------------
# 4) KL interna (student vs teacher) + relatório teacher
# -----------------------------
# KL(student || teacher) = KLDivLoss(log(student), teacher)
kd_loss_fn = nn.KLDivLoss(reduction="batchmean")
student_probs_t = student_probs  # [N, C] torch
teacher_probs_t = torch.from_numpy(teacher_probs_np)  # [N, C]
# normaliza teacher (em caso de pequenas derivações numéricas)
teacher_probs_t = torch.clamp(teacher_probs_t, min=1e-8)
teacher_probs_t = teacher_probs_t / teacher_probs_t.sum(dim=1, keepdim=True)

with torch.no_grad():
    kl_value = kd_loss_fn(torch.log(torch.clamp(student_probs_t, min=1e-8)), teacher_probs_t).item()

with open(os.path.join(results_dir, "kl_loss.txt"), "w", encoding="utf-8") as f:
    f.write(f"{kl_value}\n")
np.save(os.path.join(results_dir, "teacher_probs.npy"), teacher_probs_np)
print(f"[KL] KL(batchmean) student||teacher = {kl_value:.6f} (salvo em kl_loss.txt)")

# Relatório com base no "teacher" (depois do MQTT)
teacher_txt  = os.path.join(results_dir, "teacher_after_mqtt_report.txt")
teacher_json = os.path.join(results_dir, "teacher_after_mqtt_report.json")
_ = save_classification_report(
    y_true=np.array(labels_list),
    y_proba=teacher_probs_np,
    class_names=classes,
    out_txt_path=teacher_txt,
    out_json_path=teacher_json
)
print(f"[REPORT] Relatório teacher (pós-MQTT) salvo em: {teacher_txt}")

print("[DONE] Pipeline finalizado.")
