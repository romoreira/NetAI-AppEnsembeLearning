# --- CLIENTE: mede tráfego MQTT (TX: probs publicadas, RX: teacher recebido) ---
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import paho.mqtt.client as mqtt
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import os
from threading import Event
import torch.nn.modules.batchnorm as bn

# ADIÇÃO: métricas e utilidades
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -------------------------------------------------
# Args
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--broker', type=str, default='localhost')
parser.add_argument('--port', type=int, default=1883)
parser.add_argument('--topic', type=str, default='#')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--optimizer', type=str, default='adam')       # legado
parser.add_argument('--lr', type=float, default=1e-3)              # legado
parser.add_argument('--epochs', type=int, default=3)               # legado
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--client_id', type=int, required=True)
parser.add_argument('--ensemble_method', type=str, required=True)
parser.add_argument('--combo_name', type=str, required=True)
parser.add_argument('--pth_path', type=str, required=True)
# Ciclo
parser.add_argument('--rounds', type=int, default=1)
parser.add_argument('--kd_epochs', type=int, default=1)
parser.add_argument('--wait_timeout_sec', type=int, default=300)
# KD principal
parser.add_argument('--kd_lr', type=float, default=1e-5)
parser.add_argument('--kd_weight_decay', type=float, default=0.0)
parser.add_argument('--kd_eps', type=float, default=1e-4)
parser.add_argument('--kd_alpha', type=float, default=0.0)   # CE local
parser.add_argument('--kd_beta', type=float, default=0.2)    # blending teacher/student
# Sinais extras
parser.add_argument('--kd_aug', action='store_true')
parser.add_argument('--kd_conf_thresh', type=float, default=0.0)
parser.add_argument('--kd_entropy_weight', action='store_true')
# NEW: regularização de consistência com o próprio round-1
parser.add_argument('--kd_self_consistency', type=float, default=0.1,
                    help='Peso da KL para manter aluno próximo das probs do round 1 (0 desliga)')
args = parser.parse_args()

# --- Caminho Base para Salvar Resultados ---
RESULTS_BASE_PATH = f"results/{args.combo_name}/{args.ensemble_method}"
ensure_dir(RESULTS_BASE_PATH)
CLIENT_RESULTS_PATH = os.path.join(RESULTS_BASE_PATH, f"client_{args.client_id}")
ensure_dir(CLIENT_RESULTS_PATH)

def _round_dir(r: int) -> str:
    d = os.path.join(CLIENT_RESULTS_PATH, f"round_{r}")
    ensure_dir(d)
    return d

def _save_mqtt_traffic(round_id: int, update: dict):
    """Salva/atualiza o arquivo de tráfego MQTT deste round."""
    path = os.path.join(_round_dir(round_id), "mqtt_traffic.json")
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.update(update or {})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# -------------------------------------------------
# MQTT
# -------------------------------------------------
MQTT_BROKER = args.broker
MQTT_PORT   = args.port
client_str  = f"client{args.client_id}"
MQTT_TOPIC_OUT = f"{client_str}/probs"  # publicamos aqui
MQTT_TOPIC_IN  = f"server/teacher/{args.combo_name}/{args.ensemble_method}"  # recebemos aqui

print(f"[CONFIG] Broker: {MQTT_BROKER}, Porta: {MQTT_PORT}")
print(f"[CONFIG] Tópico OUT (envio): {MQTT_TOPIC_OUT}")
print(f"[CONFIG] Tópico IN  (recebe teacher): {MQTT_TOPIC_IN}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Usando dispositivo: {device}")

# -------------------------------------------------
# Dataset
# -------------------------------------------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder(root="../AIDER_split/val", transform=val_transform)
val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

kd_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
kd_dataset = datasets.ImageFolder(root="../AIDER_split/val",
                                  transform=kd_transform if args.kd_aug else val_transform)
kd_loader  = DataLoader(kd_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

classes = list(val_dataset.classes)
num_classes = len(classes)
num_samples = len(val_dataset)
print(f"[DATA] Val: {num_samples} amostras | classes({num_classes}): {classes}")

# -------------------------------------------------
# Modelo
# -------------------------------------------------
def get_model(name, num_classes):
    name = name.lower()
    weights = None
    print(f"[MODEL] Carregando arquitetura: {name}")
    if name == "resnet18":
        m = models.resnet18(weights=weights); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet34":
        m = models.resnet34(weights=weights); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=weights); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "alexnet":
        m = models.alexnet(weights=weights);  m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif name == "vgg16":
        m = models.vgg16(weights=weights);    m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif name == "vgg19":
        m = models.vgg19(weights=weights);    m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=weights); m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=weights); m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=weights); m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name in ("squeezenet", "squeezenet1_0"):
        m = models.squeezenet1_0(weights=weights); m.classifier[1] = nn.Conv2d(512, num_classes, 1, 1); m.num_classes = num_classes
    elif name == "densenet121":
        m = models.densenet121(weights=weights); m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights); m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Modelo '{name}' não suportado.")
    return m

def get_classifier_params(model, name):
    name = name.lower()
    if name in ("resnet18", "resnet34", "resnet50"):
        return list(model.fc.parameters())
    if name in ("vgg16", "vgg19", "alexnet"):
        return [model.classifier[6].weight, model.classifier[6].bias]
    if name == "mobilenet_v2":
        return list(model.classifier[1].parameters())
    if name in ("mobilenet_v3_small", "mobilenet_v3_large"):
        return list(model.classifier[3].parameters())
    if name in ("squeezenet", "squeezenet1_0"):
        return list(model.classifier[1].parameters())
    if name == "densenet121":
        return list(model.classifier.parameters())
    if name == "efficientnet_b0":
        return list(model.classifier[1].parameters())
    raise ValueError(f"Modelo '{name}' não suportado para seleção de cabeça.")

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
        state = ckpt.state_dict()
    state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=strict)

model = get_model(args.model_name, num_classes).to(device)
print(f"[LOAD] Carregando pesos de: {args.pth_path}")
load_checkpoint_any(model, args.pth_path, map_location=device, strict=False)

# Congelar backbone e treinar só a cabeça
for p in model.parameters():
    p.requires_grad = False
head_params = get_classifier_params(model, args.model_name)
for p in head_params:
    p.requires_grad = True

# -------------------------------------------------
# Utils de avaliação/relatórios
# -------------------------------------------------
def extract_probs_current_model():
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Extracting probs", leave=False):
            inputs  = inputs.to(device)
            outputs = model(inputs)
            probs   = F.softmax(outputs, dim=1).cpu()
            all_probs.append(probs); all_labels.append(targets)
    probs  = torch.cat(all_probs)   # (N,C)
    labels = torch.cat(all_labels)  # (N,)
    return probs, labels

def _save_confusion_png(cm: np.ndarray, classes: list, out_png: str, title: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

def _save_classification_artifacts(probs: torch.Tensor,
                                   labels: torch.Tensor,
                                   classes: list,
                                   round_id: int,
                                   phase: str):  # "before" ou "after"
    """
    Salva classification_report em TXT/JSON e matriz de confusão em PNG.
    """
    # Dir do round
    round_dir = _round_dir(round_id)

    # Predições
    y_true = labels.cpu().numpy()
    y_pred = probs.argmax(dim=1).cpu().numpy()

    # Relatório
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0, digits=4)
    report_txt  = classification_report(y_true, y_pred, target_names=classes, zero_division=0, digits=4)
    acc = accuracy_score(y_true, y_pred)

    # Salva TXT
    with open(os.path.join(round_dir, f"classification_{phase}.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write(report_txt)

    # Salva JSON
    import json as _json
    payload = {"accuracy": acc, "report": report_dict}
    with open(os.path.join(round_dir, f"classification_{phase}.json"), "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, indent=2)

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    _save_confusion_png(cm, classes, os.path.join(round_dir, f"confusion_{phase}.png"),
                        title=f"Confusion ({phase}) - round {round_id}")

def publish_probs(mqttc, round_id, probs, labels):
    # Serializa uma vez para medir os bytes de TX
    payload_obj = {"round_id": round_id,
                   "probs": probs.detach().cpu().tolist(),
                   "labels": labels.detach().cpu().tolist()}
    payload_str = json.dumps(payload_obj, separators=(",", ":"))  # compacta para medir fielmente
    payload_bytes = payload_str.encode("utf-8")
    qos = 2
    info = mqttc.publish(MQTT_TOPIC_OUT, payload_bytes, qos=qos)
    info.wait_for_publish()
    print(f"[MQTT→] Round {round_id}: probs publicadas (N={len(labels)}) no tópico {MQTT_TOPIC_OUT} (mid={info.mid}).")

    # Salva métrica de TX deste round
    _save_mqtt_traffic(round_id, {
        "round": int(round_id),
        "topic_out": MQTT_TOPIC_OUT,
        "topic_in": MQTT_TOPIC_IN,
        "qos_out": int(qos),
        "num_samples": int(len(labels)),
        "num_classes": int(probs.shape[1]),
        "bytes_tx_probs": int(len(payload_bytes))
    })

teacher_event   = Event()
teacher_payload = {"round_id": None, "probs": None, "labels": None}

def on_connect(mqttc, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Conectado. Assinando {MQTT_TOPIC_IN}")
        mqttc.subscribe(MQTT_TOPIC_IN, qos=1)
    else:
        print(f"[MQTT] Falha na conexão (rc={rc})")

def on_message(mqttc, userdata, msg):
    global teacher_payload
    try:
        raw = msg.payload  # bytes recebidos
        data = json.loads(raw.decode())
        teacher_payload = {"round_id": data.get("round_id", None),
                           "probs":    data.get("probs", None),
                           "labels":   data.get("labels", None)}
        print(f"[MQTT←] Recebidas teacher probs (round {teacher_payload['round_id']}) de {msg.topic}")
        # Salva métrica de RX deste round (ou atualiza o arquivo existente)
        if teacher_payload["round_id"] is not None:
            _save_mqtt_traffic(int(teacher_payload["round_id"]), {
                "bytes_rx_teacher": int(len(raw)),
                "qos_in": int(msg.qos)
            })
        teacher_event.set()
    except Exception as e:
        print(f"[MQTT] Erro parse payload teacher: {e}")

optimizer = torch.optim.Adam(head_params, lr=args.kd_lr, weight_decay=args.kd_weight_decay)

def set_bn_eval(module):
    if isinstance(module, (bn.BatchNorm1d, bn.BatchNorm2d, bn.BatchNorm3d)):
        module.eval()

print(f"🚀 [MQTT] Conectando ao broker {MQTT_BROKER}:{MQTT_PORT}")
mqttc = mqtt.Client(client_id=f"{client_str}-worker")
mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.connect(MQTT_BROKER, MQTT_PORT, 60)
mqttc.loop_start()

# -------------------------------------------------
# Rounds
# -------------------------------------------------
current_round = 1
probs, labels = extract_probs_current_model()
print(f"[EVAL] Round {current_round} - amostras: {probs.shape[0]}, classes: {probs.shape[1]}")

# Salva relatório ANTES do MQTT (BEFORE do round 1)
_save_classification_artifacts(probs, labels, classes, round_id=current_round, phase="before")

publish_probs(mqttc, current_round, probs, labels)

# NEW: baseline do round-1 para consistência
baseline_probs = probs.to(device)  # (N,C)

total_rounds = max(1, args.rounds)
while current_round <= total_rounds:
    print(f"[WAIT] Aguardando teacher probs do round {current_round} em {MQTT_TOPIC_IN} (timeout={args.wait_timeout_sec}s)...")
    teacher_event.clear()
    if not teacher_event.wait(timeout=args.wait_timeout_sec):
        print(f"[WARN] Timeout aguardando teacher probs para round {current_round}. Encerrando ciclo.")
        break

    if teacher_payload["round_id"] != current_round:
        print(f"[INFO] Recebi teacher round {teacher_payload['round_id']}, mas espero {current_round}. Continuando espera...")
        continue

    teacher_probs = torch.tensor(teacher_payload["probs"], dtype=torch.float32, device=device)
    if teacher_probs.ndim != 2 or teacher_probs.shape[0] != num_samples or teacher_probs.shape[1] != num_classes:
        print(f"[ERROR] Shape inesperado de teacher_probs: {tuple(teacher_probs.shape)}; esperado ({num_samples}, {num_classes}). Abortando.")
        break

    print(f"[KD] Iniciando KD do round {current_round}: epochs={args.kd_epochs} (kd_lr={args.kd_lr})")
    model.train()
    model.apply(set_bn_eval)

    with torch.no_grad():
        K   = teacher_probs.shape[1]
        eps = args.kd_eps
        teacher_probs = teacher_probs * (1.0 - eps) + (eps / K)
        teacher_probs = torch.clamp(teacher_probs, min=1e-8)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True)

    use_loader = kd_loader if args.kd_aug else val_loader

    sample_base = 0
    for epoch in range(args.kd_epochs):
        sample_base = 0
        pbar = tqdm(use_loader, desc=f"KD Round {current_round} Epoch {epoch+1}/{args.kd_epochs}", leave=False)
        for inputs, y_local in pbar:
            bsz = inputs.size(0)
            inputs  = inputs.to(device)
            y_local = y_local.to(device)

            logits        = model(inputs)
            log_p_student = F.log_softmax(logits, dim=1)
            p_teacher     = teacher_probs[sample_base:sample_base+bsz, :]

            # blending teacher/student
            with torch.no_grad():
                p_student_now = F.softmax(logits, dim=1)
                if args.kd_beta > 0.0:
                    p_teacher = (1.0 - args.kd_beta) * p_teacher + args.kd_beta * p_student_now
                    p_teacher = p_teacher / p_teacher.sum(dim=1, keepdim=True)

            # KL por amostra com opções de peso/máscara
            kl_per_sample = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=1)

            if args.kd_entropy_weight:
                with torch.no_grad():
                    Kc  = p_teacher.size(1)
                    ent = -(p_teacher * p_teacher.clamp_min(1e-8).log()).sum(dim=1)
                    ent_norm = ent / torch.log(torch.tensor(Kc, device=ent.device, dtype=ent.dtype))
                    w = (1.0 - ent_norm).clamp(0.0, 1.0)
                kl_per_sample = kl_per_sample * w

            if args.kd_conf_thresh > 0.0:
                with torch.no_grad():
                    conf = p_teacher.max(dim=1).values
                    mask = (conf >= args.kd_conf_thresh).float()
                if mask.sum() < 1:
                    sample_base += bsz
                    continue
                kd_loss = (kl_per_sample * mask).sum() / mask.sum()
            else:
                kd_loss = kl_per_sample.mean()

            loss = kd_loss
            if args.kd_alpha > 0.0:
                ce = F.cross_entropy(logits, y_local)
                loss = args.kd_alpha * ce + (1.0 - args.kd_alpha) * kd_loss

            # NEW: consistência com as probs do round-1
            if args.kd_self_consistency > 0.0:
                p0_batch = baseline_probs[sample_base:sample_base+bsz, :].to(device)
                p0_batch = torch.clamp(p0_batch, min=1e-8)
                p0_batch = p0_batch / p0_batch.sum(dim=1, keepdim=True)
                kl_self  = F.kl_div(log_p_student, p0_batch, reduction="batchmean")
                loss = loss + args.kd_self_consistency * kl_self

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            optimizer.step()

            pbar.set_postfix({"kd_loss": float(kd_loss.detach().cpu())})
            sample_base += bsz

    # Após treinar com KD no round atual, roda avaliação e salva AFTER
    probs_after, labels_after = extract_probs_current_model()
    _save_classification_artifacts(probs_after, labels_after, classes, round_id=current_round, phase="after")

    current_round += 1
    if current_round <= total_rounds:
        probs, labels = extract_probs_current_model()
        print(f"[EVAL] Round {current_round} - amostras: {probs.shape[0]}, classes: {probs.shape[1]}")
        # BEFORE do próximo round
        _save_classification_artifacts(probs, labels, classes, round_id=current_round, phase="before")
        publish_probs(mqttc, current_round, probs, labels)
    else:
        print("[DONE] Número máximo de rounds alcançado.")

mqttc.loop_stop()
mqttc.disconnect()
print("🔌 [DEBUG] Cliente desconectado do broker.")
