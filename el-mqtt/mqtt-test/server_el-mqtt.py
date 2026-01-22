# --- SERVIDOR: mede tráfego MQTT (RX de clientes, TX do teacher) ---
import paho.mqtt.client as mqtt
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import argparse, time, os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

try:
    from torchvision import datasets
except Exception:
    datasets = None

from optimizations import (
    run_genetic_algorithm,
    evaluate_weighted_probs,          # <- mantém, agora com round_id opcional
    run_hybrid_ensemble_ga_stacking,
    run_pso_optimization,
    run_hybrid_ensemble_pso_stacking
)

def apply_acm_style():
    mpl.rcParams.update({
        "font.size": 16,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral", "TeX Gyre Termes"],
        "axes.titlesize": 16, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 14, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -------------------------------------------------
# Args
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--broker', type=str, default='localhost')
parser.add_argument('--port', type=int, default=1883)
parser.add_argument('--topic', type=str, default='probs')
parser.add_argument('--expected_clients', type=str, required=True)
parser.add_argument('--ensemble_method', type=str, required=True)
parser.add_argument('--combo_name', type=str, required=True)
# Suavização/EMA
parser.add_argument('--teacher_temp', type=float, default=2.0)
parser.add_argument('--teacher_eps', type=float, default=1e-4)
parser.add_argument('--teacher_ema', type=float, default=0.9)
# NEW: gating do teacher (aceitar só se não piorar muito)
parser.add_argument('--teacher_gate', action='store_true',
                    help='Ativa seleção best-so-far: rejeita teacher pior que o melhor por tol')
parser.add_argument('--teacher_tol', type=float, default=0.002,
                    help='Tolerância de queda de acurácia para aceitar novo teacher')
args = parser.parse_args()

print(f"[TEACHER] T={args.teacher_temp:.3f} | eps={args.teacher_eps:.1e} | EMA={args.teacher_ema:.2f} | gate={args.teacher_gate} tol={args.teacher_tol}")

RESULTS_BASE_PATH = f"results/{args.combo_name}/{args.ensemble_method}"
ensure_dir(RESULTS_BASE_PATH)
SERVER_RESULTS_PATH = os.path.join(RESULTS_BASE_PATH, "server")
ensure_dir(SERVER_RESULTS_PATH)

train_dataset = None
if datasets is not None:
    try:
        train_dataset = datasets.ImageFolder(root="../AIDER_split/train")
        print(f"[DATA] classes do ImageFolder: {len(train_dataset.classes)} detectadas.")
    except Exception as e:
        print(f"[WARN] Não foi possível carregar ImageFolder: {e}")

def get_class_names_from_train_dataset(n_expected: int):
    if train_dataset is not None and hasattr(train_dataset, "classes"):
        names = list(train_dataset.classes)
        if len(names) == n_expected:
            return [str(x) for x in names]
        if n_expected <= len(names):
            return [str(x) for x in names[:n_expected]]
    return None

def soften_and_smooth_probs(probs_matrix: np.ndarray, T: float, eps: float) -> np.ndarray:
    probs = np.asarray(probs_matrix, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    if T is None or T <= 1.0 + 1e-9:
        p_T = probs
    else:
        invT = 1.0 / T
        p_T = np.power(probs, invT)
        p_T /= np.sum(p_T, axis=1, keepdims=True)
    if eps and eps > 0.0:
        K = p_T.shape[1]
        p_T = (1.0 - eps) * p_T + (eps / K)
    p_T /= np.sum(p_T, axis=1, keepdims=True)
    p_T = np.clip(p_T, 1e-8, 1.0)
    p_T /= np.sum(p_T, axis=1, keepdims=True)
    return p_T.astype(np.float32)

# MQTT
MQTT_BROKER = args.broker
MQTT_PORT   = args.port
MQTT_TOPIC  = f"+/{args.topic}"
print(f"[CONFIG] Broker: {MQTT_BROKER}, Porta: {MQTT_PORT}, Tópico: {MQTT_TOPIC}")
expected_clients = {f"client{i+1}" for i in range(int(args.expected_clients))}
print(f"[CONFIG] Esperando mensagens de: {expected_clients}")

mqtt_client = None
MQTT_TEACHER_TOPIC = f"server/teacher/{args.combo_name}/{args.ensemble_method}"

received_probs   = {}
round_accuracies = []
round_durations  = []
message_counts   = []

# ---- NOVO: Telemetria de tráfego por rodada ----
# Structure:
#   traffic[round_id] = {
#       "rx_clients": {"client1": bytes_total, ...},
#       "tx_teacher": bytes_total
#   }
traffic = {}

def _traffic_add_rx(round_id, client_id, nbytes):
    r = traffic.setdefault(round_id, {"rx_clients": {}, "tx_teacher": 0})
    r["rx_clients"][client_id] = r["rx_clients"].get(client_id, 0) + int(nbytes)

def _traffic_set_tx(round_id, nbytes):
    r = traffic.setdefault(round_id, {"rx_clients": {}, "tx_teacher": 0})
    r["tx_teacher"] = int(nbytes)

def _traffic_flush_round(round_id, method: str, labels_len: int, classes: int):
    """Salva em JSON os totais por rodada no diretório do servidor."""
    r = traffic.get(round_id, {"rx_clients": {}, "tx_teacher": 0})
    per_client = [{"client": cid, "bytes_rx": int(v)} for cid, v in sorted(r["rx_clients"].items())]
    totals = {
        "bytes_rx_clients": int(sum(r["rx_clients"].values())),
        "bytes_tx_teacher": int(r["tx_teacher"])
    }
    out = {
        "round": int(round_id),
        "method": method,
        "labels_count": int(labels_len),
        "num_classes": int(classes),
        "totals": totals,
        "per_client": per_client
    }
    path = os.path.join(SERVER_RESULTS_PATH, f"traffic_round_{round_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[TRAFFIC] Salvo {path}: totals={totals}")

# Estado do teacher
last_teacher = None         # para EMA
best_teacher = None         # melhor teacher cheio (N,K)
best_acc     = -1.0         # melhor acc observada

def publish_teacher_probs(round_id, probs_matrix, labels, method):
    if mqtt_client is None:
        print("[WARN] mqtt_client ainda não inicializado; não foi possível publicar teacher_probs.")
        return
    payload_obj = {"round_id": round_id, "method": method,
                   "labels": np.asarray(labels).tolist(),
                   "probs":  np.asarray(probs_matrix, dtype=np.float32).tolist()}
    payload_str = json.dumps(payload_obj, separators=(",", ":"))
    payload_bytes = payload_str.encode("utf-8")
    mqtt_client.publish(MQTT_TEACHER_TOPIC, payload_bytes, qos=1)
    print(f"[PUBLISH] Teacher probs publicadas em '{MQTT_TEACHER_TOPIC}' (round {round_id}, method {method}).")
    # mede TX
    _traffic_set_tx(round_id, len(payload_bytes))
    # salva arquivo de tráfego deste round (com método e metadados)
    _traffic_flush_round(round_id, method, labels_len=len(labels), classes=probs_matrix.shape[1])

def apply_teacher_ema(y_teacher: np.ndarray) -> np.ndarray:
    global last_teacher
    if args.teacher_ema > 0.0 and last_teacher is not None and last_teacher.shape == y_teacher.shape:
        y_teacher = args.teacher_ema * last_teacher + (1.0 - args.teacher_ema) * y_teacher
        y_teacher = y_teacher / y_teacher.sum(axis=1, keepdims=True)
    last_teacher = y_teacher.copy()
    return y_teacher

def gate_teacher(y_teacher_new: np.ndarray, acc_new: float) -> np.ndarray:
    """Seleciona teacher com base no melhor até agora e tolerância."""
    global best_acc, best_teacher
    if not args.teacher_gate:
        if acc_new > best_acc:
            best_acc = acc_new
            best_teacher = y_teacher_new.copy()
        return y_teacher_new

    if best_acc < 0:
        best_acc = acc_new
        best_teacher = y_teacher_new.copy()
        print(f"[GATE] Primeira referência: acc_best={best_acc:.4f}")
        return y_teacher_new

    if acc_new < best_acc - args.teacher_tol:
        print(f"[GATE] Rejeitando teacher novo (acc={acc_new:.4f} < best={best_acc:.4f} - tol={args.teacher_tol:.4f}). Usando best-so-far.")
        return best_teacher.copy()
    else:
        if acc_new > best_acc:
            print(f"[GATE] Novo best teacher: {best_acc:.4f} → {acc_new:.4f}")
            best_acc = acc_new
            best_teacher = y_teacher_new.copy()
        else:
            print(f"[GATE] Aceito (dentro da tolerância). Best permanece {best_acc:.4f}.")
        return y_teacher_new

# ----------------- Métricas/plots utilitários (iguais) -----------------
def plot_round_series_pdf(values, ylabel, filename, marker='o'):
    apply_acm_style()
    fig, ax = plt.subplots(figsize=(6.9, 3.2), constrained_layout=True)
    x = np.arange(1, len(values) + 1)
    ax.plot(x, values, marker=marker, markersize=4.5, linewidth=2.0)
    ax.set_xlabel("Round", labelpad=8); ax.set_ylabel(ylabel, labelpad=8)
    ax.tick_params(axis='both', which='major', pad=6)
    fig.savefig(os.path.join(RESULTS_BASE_PATH, filename), bbox_inches="tight"); plt.close(fig)

def plot_avg_probs_pdf(avg_probs, class_names, filename):
    apply_acm_style()
    n = len(avg_probs)
    fig, ax = plt.subplots(figsize=(6.9, 3.2), constrained_layout=True)
    idx = np.arange(n)
    ax.bar(idx, avg_probs, width=0.8, edgecolor="black", linewidth=0.6, color="0.35")
    ax.set_xlabel("Class", labelpad=8); ax.set_ylabel("Average weighted probability", labelpad=8)
    ax.set_xticks(idx); ax.set_xticklabels(class_names if n <= 12 else class_names, rotation=0 if n <= 12 else 45, ha="right")
    ax.tick_params(axis='both', which='major', pad=6)
    fig.savefig(os.path.join(RESULTS_BASE_PATH, filename), bbox_inches="tight"); plt.close(fig)

def save_plots_and_reports(X_stack, y_stack, y_test, y_pred, y_proba, round_id):
    ensure_dir(RESULTS_BASE_PATH); apply_acm_style()
    report = classification_report(y_test, y_pred, digits=4)
    with open(f"{RESULTS_BASE_PATH}/classification_report_round{round_id}.txt", "w") as f:
        f.write(f"Classification Report - Round {round_id}\n\n{report}")

# ----------------- Agregadores -----------------
def process_stacking_aggregation(round_id, X_stack, y_stack, start_time):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_stack, y_stack, test_size=0.2, random_state=42, stratify=y_stack
    )
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_train, y_train)
    y_pred  = meta.predict(X_test)
    y_proba = meta.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    # (poderia registrar tempo/duração por round aqui)
    save_plots_and_reports(X_stack, y_stack, y_test, y_pred, y_proba, round_id)
    meta_full = LogisticRegression(max_iter=1000)
    meta_full.fit(X_stack, y_stack)
    y_proba_full = meta_full.predict_proba(X_stack)
    return acc, y_proba, y_proba_full

def aggregate_with_stacking(round_id):
    clients_received = received_probs.get(round_id, {})
    if not all(len(clients_received.get(c, [])) >= 1 for c in expected_clients):
        return
    print(f"[Stacking] ✅ Todos os clientes da rodada {round_id} enviaram. Realizando stacking...")
    t0 = time.time()
    probs_list = [np.array(clients_received[c][0]["probs"]) for c in sorted(expected_clients)]
    y_stack    = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])
    X_stack    = np.concatenate(probs_list, axis=1)
    acc, _, y_full = process_stacking_aggregation(round_id, X_stack, y_stack, t0)
    print(f"🎯 Acurácia rodada Stacking {round_id} (teste): {acc:.4f}")
    y_teacher = soften_and_smooth_probs(y_full, T=args.teacher_temp, eps=args.teacher_eps)
    y_teacher = gate_teacher(y_teacher, acc)
    y_teacher = apply_teacher_ema(y_teacher)
    publish_teacher_probs(round_id, y_teacher, y_stack, method="stacking")
    del received_probs[round_id]

def aggregate_with_GA(round_id):
    print(f"[GA] Verificando rodada {round_id}...")
    clients_received = received_probs.get(round_id, {})
    if not all(len(clients_received.get(c, [])) >= 1 for c in expected_clients):
        print(f"[GA] ⏳ Aguardando mensagens..."); return
    probs_list = [np.array(clients_received[c][0]["probs"]) for c in sorted(expected_clients)]
    labels     = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])
    t0 = time.time()
    best_weights = run_genetic_algorithm(probs_list, labels, args)
    acc = evaluate_weighted_probs(probs_list, best_weights, labels, args, args.ensemble_method, round_id)
    print(f"[GA - RESULT] Time={time.time()-t0:.3f}s | Acc={acc:.4f}")
    combined = np.zeros_like(probs_list[0])
    for i, p in enumerate(probs_list): combined += best_weights[i] * np.array(p)
    combined /= np.maximum(combined.sum(axis=1, keepdims=True), 1e-12)
    y_teacher = soften_and_smooth_probs(combined, T=args.teacher_temp, eps=args.teacher_eps)
    y_teacher = gate_teacher(y_teacher, acc)
    y_teacher = apply_teacher_ema(y_teacher)
    publish_teacher_probs(round_id, y_teacher, labels, method="ga")
    del received_probs[round_id]

def aggregate_with_PSO(round_id):
    print(f"[PSO] Verificando rodada {round_id}...")
    clients_received = received_probs.get(round_id, {})
    if not all(len(clients_received.get(c, [])) >= 1 for c in expected_clients):
        print(f"[PSO] ⏳ Aguardando mensagens..."); return
    probs_list = [np.array(clients_received[c][0]["probs"]) for c in sorted(expected_clients)]
    y_stack    = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])
    t0 = time.time()
    best_weights = run_pso_optimization(probs_list, y_stack)
    acc = evaluate_weighted_probs(probs_list, best_weights, y_stack, args, args.ensemble_method, round_id)
    print(f"[PSO - RESULT] Time={time.time()-t0:.3f}s | Acc={acc:.4f}")
    combined = np.zeros_like(probs_list[0])
    for i, p in enumerate(probs_list): combined += best_weights[i] * np.array(p)
    combined /= np.maximum(combined.sum(axis=1, keepdims=True), 1e-12)
    y_teacher = soften_and_smooth_probs(combined, T=args.teacher_temp, eps=args.teacher_eps)
    y_teacher = gate_teacher(y_teacher, acc)
    y_teacher = apply_teacher_ema(y_teacher)
    publish_teacher_probs(round_id, y_teacher, y_stack, method="pso")
    del received_probs[round_id]

def aggregate_with_GA_and_Stacking(round_id):
    clients_received = received_probs.get(round_id, {})
    if not all(len(clients_received.get(c, [])) >= 1 for c in expected_clients): return
    print(f"[GA+Stacking] ✅ Rodada {round_id}")
    probs_list = [np.array(clients_received[c][0]["probs"]) for c in sorted(expected_clients)]
    labels     = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])
    t0 = time.time()
    ga_weights = run_genetic_algorithm(probs_list, labels, args)
    _, X_stack, y_stack = run_hybrid_ensemble_ga_stacking(probs_list, labels, ga_weights)
    acc, _, y_full = process_stacking_aggregation(round_id, X_stack, y_stack, t0)
    print(f"🎯 Acurácia GA+Stacking {round_id}: {acc:.4f}")
    y_teacher = soften_and_smooth_probs(y_full, T=args.teacher_temp, eps=args.teacher_eps)
    y_teacher = gate_teacher(y_teacher, acc)
    y_teacher = apply_teacher_ema(y_teacher)
    publish_teacher_probs(round_id, y_teacher, y_stack, method="ga_stacking")
    del received_probs[round_id]

def aggregate_with_PSO_and_Stacking(round_id):
    clients_received = received_probs.get(round_id, {})
    if not all(len(clients_received.get(c, [])) >= 1 for c in expected_clients): return
    print(f"[PSO+Stacking] ✅ Rodada {round_id}")
    probs_list = [np.array(clients_received[c][0]["probs"]) for c in sorted(expected_clients)]
    labels     = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])
    t0 = time.time()
    pso_weights = run_pso_optimization(probs_list, labels)
    _, X_stack, y_stack = run_hybrid_ensemble_pso_stacking(probs_list, labels, pso_weights)
    acc, _, y_full = process_stacking_aggregation(round_id, X_stack, y_stack, t0)
    print(f"🎯 Acurácia PSO+Stacking {round_id}: {acc:.4f}")
    y_teacher = soften_and_smooth_probs(y_full, T=args.teacher_temp, eps=args.teacher_eps)
    y_teacher = gate_teacher(y_teacher, acc)
    y_teacher = apply_teacher_ema(y_teacher)
    publish_teacher_probs(round_id, y_teacher, y_stack, method="pso_stacking")
    del received_probs[round_id]

# -------------------------------------------------
# MQTT callbacks
# -------------------------------------------------
def on_message(client, userdata, msg):
    print(f"\n📩 Mensagem recebida de {msg.topic}")
    try:
        raw = msg.payload
        payload   = json.loads(raw.decode())
        client_id = msg.topic.split('/')[0]
        round_id  = payload.get("round_id", "default")

        # Telemetria de RX por cliente/rodada
        _traffic_add_rx(round_id, client_id, len(raw))

        if round_id not in received_probs: received_probs[round_id] = {}
        received_probs[round_id][client_id] = [{"probs": payload["probs"], "labels": payload["labels"]}]

        fn = {
            "stacking":     aggregate_with_stacking,
            "ga":           aggregate_with_GA,
            "ga_stacking":  aggregate_with_GA_and_Stacking,
            "pso":          aggregate_with_PSO,
            "pso_stacking": aggregate_with_PSO_and_Stacking,
        }.get(args.ensemble_method)

        if fn: fn(round_id)
        else:  print(f"❌ Método desconhecido: {args.ensemble_method}")
    except Exception as e:
        print(f"❌ Erro ao processar mensagem: {e}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"🔗 Conectado ao broker, inscrevendo-se em {MQTT_TOPIC}")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"❌ Falha na conexão com o broker (rc={rc})")

def main():
    global mqtt_client, last_teacher, best_teacher, best_acc
    print(f"[INIT] Conectando ao broker {MQTT_BROKER}:{MQTT_PORT}...")
    c = mqtt.Client()
    c.on_connect = on_connect
    c.on_message = on_message
    c.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client  = c
    last_teacher = None
    best_teacher = None
    best_acc     = -1.0
    try:
        print("🚀 Servidor aguardando mensagens...")
        c.loop_forever()
    except KeyboardInterrupt:
        print("\n🛑 Encerrando servidor MQTT.")
        c.disconnect()

if __name__ == "__main__":
    main()
