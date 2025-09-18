import paho.mqtt.client as mqtt
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import argparse
import time
import os
import matplotlib as mpl
mpl.use('Agg')  # backend off-screen para servidores/headless
import matplotlib.pyplot as plt

# === try/except para torchvision (labels do ImageFolder) ===
try:
    from torchvision import datasets
    from torchvision import transforms  # opcional; não é requerido para classes
except Exception:
    datasets = None
    transforms = None

from optimization import (
    run_genetic_algorithm,
    evaluate_weighted_probs,
    run_hybrid_ensemble_ga_stacking,
    run_pso_optimization,
    run_hybrid_ensemble_pso_stacking
)

# ========= Estilo ACM-like reutilizável =========
def apply_acm_style():
    mpl.rcParams.update({
        "font.size": 16,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral", "TeX Gyre Termes"],
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,  # texto selecionável no PDF
        "ps.fonttype": 42,
    })

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ========= CLI =========
parser = argparse.ArgumentParser()
parser.add_argument('--broker', type=str, default='localhost', help='Broker IP or hostname')
parser.add_argument('--port', type=int, default=1883, help='Broker port')
parser.add_argument('--topic', type=str, default='probs', help='MQTT topic to subscribe')
parser.add_argument('--expected_clients', type=str, required=True, help='Number of expected clients')
parser.add_argument('--ensemble_method', type=str, help='Ensemble method to use', required=True)
parser.add_argument('--combo_name', type=str, required=True, help='Identifier for the model combination')
args = parser.parse_args()

# --- Caminho Base para Salvar Resultados ---
RESULTS_BASE_PATH = f"results/{args.combo_name}/{args.ensemble_method}"
ensure_dir(RESULTS_BASE_PATH)

# ===== Carregar train_dataset para obter nomes de classes =====
train_dataset = None
if datasets is not None:
    try:
        # Você pode definir um transform se quiser; para obter .classes não é necessário
        train_dataset = datasets.ImageFolder(root="../AIDER_split/train")
        print(f"[DATA] classes do ImageFolder: {len(train_dataset.classes)} detectadas.")
    except Exception as e:
        print(f"[WARN] Não foi possível carregar ImageFolder para labels: {e}")

def get_class_names_from_train_dataset(n_expected: int):
    """
    Pega nomes de classes de train_dataset.classes se existir e tiver tamanho compatível.
    Caso contrário, retorna None para acionar o fallback.
    """
    if train_dataset is not None and hasattr(train_dataset, "classes"):
        names = list(train_dataset.classes)
        # Se o número esperado bate com o do dataset, retornamos
        if len(names) == n_expected:
            return [str(x) for x in names]
        # Se não bater, ainda podemos retornar (melhor que nada) — mas só se n_expected <= len(names)
        if n_expected <= len(names):
            return [str(x) for x in names[:n_expected]]
    return None

def fallback_class_names_from_labels(y_true, y_pred):
    """
    Fallback robusto: constrói nomes a partir dos rótulos observados.
    """
    uniq = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    return [str(c) for c in uniq], uniq  # nomes, ordem única

def map_labels_dense(y_true, y_pred, possible_labels=None):
    """
    Mapeia rótulos quaisquer para índices densos 0..K-1.
    possible_labels (opcional) pode determinar a ordem/quantidade.
    Retorna: y_true_idx, y_pred_idx, uniq_labels (na ordem usada)
    """
    if possible_labels is None:
        uniq = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    else:
        uniq = np.asarray(possible_labels)
    lab2idx = {lab: i for i, lab in enumerate(uniq)}
    y_true_idx = np.array([lab2idx[lab] for lab in y_true])
    y_pred_idx = np.array([lab2idx[lab] for lab in y_pred])
    return y_true_idx, y_pred_idx, uniq

def get_class_names_and_indices(n_classes: int, y_true=None, y_pred=None):
    """
    Política:
    1) Tentar usar train_dataset.classes se existir e bater n_classes.
    2) Caso contrário, se y_true/y_pred existirem, inferir nomes a partir deles.
    3) Se nada disso, cair para ["Class 0", ..., "Class n-1"].
    Retorna: class_names(list[str]), uniq_labels(np.ndarray com valores base), label_indices(range(K))
    """
    # 1) train_dataset
    names = get_class_names_from_train_dataset(n_classes)
    if names is not None:
        return names, np.arange(n_classes), list(range(n_classes))

    # 2) fallback de y_true/y_pred
    if y_true is not None and y_pred is not None:
        names_from_y, uniq = fallback_class_names_from_labels(y_true, y_pred)
        if len(names_from_y) == n_classes:
            return names_from_y, uniq, list(range(n_classes))
        else:
            # K é len(uniq); retornamos K nomes
            return names_from_y, uniq, list(range(len(uniq)))

    # 3) último recurso
    return [f"Class {i}" for i in range(n_classes)], np.arange(n_classes), list(range(n_classes))

# Configuração MQTT
MQTT_BROKER = args.broker
MQTT_PORT = args.port
MQTT_TOPIC = f"+/{args.topic}"

print(f"[CONFIG] Broker: {MQTT_BROKER}, Porta: {MQTT_PORT}, Tópico: {MQTT_TOPIC}")
expected_clients = {f"client{i+1}" for i in range(int(args.expected_clients))}
print(f"[CONFIG] Esperando mensagens de: {expected_clients}")

received_probs = {}
round_accuracies = []
round_durations = []
message_counts = []

# ========= Helpers de plot =========
def plot_round_series_pdf(values, ylabel, filename, marker='o'):
    apply_acm_style()
    fig, ax = plt.subplots(figsize=(6.9, 3.2), constrained_layout=True)
    x = np.arange(1, len(values) + 1)
    ax.plot(x, values, marker=marker, markersize=4.5, linewidth=2.0)
    # ax.set_title(...)  # títulos ficam na caption do paper
    ax.set_xlabel("Round", labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.tick_params(axis='both', which='major', pad=6)
    fig.savefig(os.path.join(RESULTS_BASE_PATH, filename), bbox_inches="tight")
    plt.close(fig)

def plot_avg_probs_pdf(avg_probs, class_names, filename):
    apply_acm_style()
    n = len(avg_probs)
    fig, ax = plt.subplots(figsize=(6.9, 3.2), constrained_layout=True)
    idx = np.arange(n)
    ax.bar(idx, avg_probs, width=0.8, edgecolor="black", linewidth=0.6, color="0.35")
    # ax.set_title(...)
    ax.set_xlabel("Class", labelpad=8)
    ax.set_ylabel("Average weighted probability", labelpad=8)
    ax.set_xticks(idx)
    if n > 12:
        ax.set_xticklabels(class_names, rotation=45, ha="right")
    else:
        ax.set_xticklabels(class_names)
    ax.tick_params(axis='both', which='major', pad=6)
    fig.savefig(os.path.join(RESULTS_BASE_PATH, filename), bbox_inches="tight")
    plt.close(fig)

# ========= Agregadores =========
def aggregate_with_GA(round_id):
    print(f"[GA] Verificando se todos os clientes enviaram para a rodada {round_id}...")
    clients_received = received_probs.get(round_id, {})
    ready = all(len(clients_received.get(c, [])) >= 1 for c in expected_clients)
    if not ready:
        print(f"[GA] ⏳ Ainda aguardando mensagens...")
        return

    print(f"[GA] ✅ Todos os clientes da rodada {round_id} enviaram. Otimizando com GA...")
    t_start = time.time()

    probs_list = [np.array(clients_received[client][0]["probs"]) for client in sorted(expected_clients)]
    labels = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])

    best_weights = run_genetic_algorithm(probs_list, labels, args)
    acc = evaluate_weighted_probs(probs_list, best_weights, labels, args, args.ensemble_method)
    duration = time.time() - t_start
    print(f"[GA - RESULT] - Time Elapsed: {duration:.3f}s - Accuracy with GA: {acc:.4f}")
    # (sem plots adicionais nesta função)

def aggregate_with_PSO(round_id):
    print(f"[PSO] Verificando se todos os clientes enviaram para a rodada {round_id}...")
    clients_received = received_probs.get(round_id, {})
    ready = all(len(clients_received.get(c, [])) >= 1 for c in expected_clients)
    if not ready:
        print(f"[PSO] ⏳ Ainda aguardando mensagens...")
        return

    print(f"[PSO] ✅ Todos os clientes da rodada {round_id} enviaram. Otimizando com PSO...")
    t_start = time.time()

    probs_list = [np.array(clients_received[client][0]["probs"]) for client in sorted(expected_clients)]
    y_stack = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])

    best_weights = run_pso_optimization(probs_list, y_stack)
    acc = evaluate_weighted_probs(probs_list, best_weights, y_stack, args, args.ensemble_method)
    duration = time.time() - t_start
    print(f"[PSO - RESULT] - Time Elapsed: {duration:.3f}s - Accuracy with PSO: {acc:.4f}")

    round_accuracies.append(acc)
    round_durations.append(duration)
    message_counts.append({c: len(clients_received.get(c, [])) for c in expected_clients})

    # --- Gráficos ACM em PDF ---
    ensure_dir(RESULTS_BASE_PATH)
    plot_round_series_pdf(round_accuracies, "Accuracy", "accuracy_per_round.pdf", marker='o')
    plot_round_series_pdf(round_durations, "Aggregation time (s)", "aggregation_time.pdf", marker='x')

    # Probabilidades médias ponderadas por classe
    combined = np.zeros_like(probs_list[0])
    for i, probs in enumerate(probs_list):
        combined += best_weights[i] * np.array(probs)
    avg_probs = np.mean(combined, axis=0)  # (n_classes,)

    n_classes = avg_probs.shape[0]
    class_names, _, _ = get_class_names_and_indices(n_classes)
    plot_avg_probs_pdf(avg_probs, class_names, f"avg_probs_round{round_id}.pdf")

    del received_probs[round_id]
    print(f"[PSO] Dados da rodada {round_id} limpos.")

# ========= Relatórios e Confusão =========
def save_plots_and_reports(X_stack, y_stack, y_test, y_pred, y_proba, round_id):
    """Salva relatório e gráficos no padrão ACM (PDF), usando labels do train_dataset quando possível."""
    ensure_dir(RESULTS_BASE_PATH)
    apply_acm_style()

    # Relatório de Classificação (TXT)
    report = classification_report(y_test, y_pred, digits=4)
    with open(f"{RESULTS_BASE_PATH}/classification_report_round{round_id}.txt", "w") as f:
        f.write(f"Classification Report - Round {round_id}\n\n{report}")

    # ===== Matriz de Confusão =====
    # Descobrir K (número de classes observadas no conjunto de teste)
    uniq = np.unique(np.concatenate([np.asarray(y_test), np.asarray(y_pred)]))
    K = len(uniq)

    # Tentar usar nomes do train_dataset; senão, nomes a partir dos rótulos observados
    names_from_ds = get_class_names_from_train_dataset(K)
    if names_from_ds is not None:
        class_names = names_from_ds
        # Supõe-se que y_test/y_pred estejam em 0..K-1; se não, mapear para denso
        if (np.min(y_test) < 0) or (np.max(y_test) >= K) or (np.min(y_pred) < 0) or (np.max(y_pred) >= K):
            y_test_idx, y_pred_idx, uniq_ord = map_labels_dense(y_test, y_pred)
        else:
            y_test_idx, y_pred_idx, uniq_ord = np.asarray(y_test), np.asarray(y_pred), np.arange(K)
    else:
        # Fallback consistente: nomes e ordem vindos de uniq
        class_names, uniq_ord = fallback_class_names_from_labels(y_test, y_pred)
        y_test_idx, y_pred_idx, uniq_ord = map_labels_dense(y_test, y_pred, possible_labels=uniq_ord)

    label_indices = list(range(len(uniq_ord)))

    # Contagens
    cm_counts = confusion_matrix(y_test_idx, y_pred_idx, labels=label_indices)
    fig_c, ax_c = plt.subplots(figsize=(6.9, 6.9), constrained_layout=True)
    disp_c = ConfusionMatrixDisplay(confusion_matrix=cm_counts, display_labels=class_names)
    disp_c.plot(cmap="Greys", xticks_rotation=45, ax=ax_c, colorbar=False, values_format="d")
    ax_c.set_xlabel("Predicted label", labelpad=10)
    ax_c.set_ylabel("True label", labelpad=10)
    ax_c.tick_params(axis="x", which="both", pad=6)
    ax_c.tick_params(axis="y", which="both", pad=6)
    fig_c.savefig(f"{RESULTS_BASE_PATH}/confusion_matrix_counts_round{round_id}.pdf", bbox_inches="tight")
    plt.close(fig_c)

    # Normalizada por linha
    cm_norm = confusion_matrix(y_test_idx, y_pred_idx, labels=label_indices, normalize="true")
    fig_n, ax_n = plt.subplots(figsize=(6.9, 6.9), constrained_layout=True)
    disp_n = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp_n.plot(cmap="Greys", xticks_rotation=45, ax=ax_n, colorbar=True, values_format=".2f")
    ax_n.set_xlabel("Predicted label", labelpad=10)
    ax_n.set_ylabel("True label", labelpad=10)
    ax_n.tick_params(axis="x", which="both", pad=6)
    ax_n.tick_params(axis="y", which="both", pad=6)
    fig_n.savefig(f"{RESULTS_BASE_PATH}/confusion_matrix_norm_round{round_id}.pdf", bbox_inches="tight")
    plt.close(fig_n)

    # ===== Séries de rodada (PDF) =====
    plot_round_series_pdf(round_accuracies, "Accuracy", "accuracy_per_round.pdf", marker='o')
    plot_round_series_pdf(round_durations, "Aggregation time (s)", "aggregation_time.pdf", marker='x')

    # ===== Distribuições de confiança (PDF) =====
    if y_proba is not None:
        confidences = np.max(y_proba, axis=1)
        fig_h1, ax_h1 = plt.subplots(figsize=(6.9, 3.2), constrained_layout=True)
        ax_h1.hist(confidences, bins=20, edgecolor="black")
        ax_h1.set_xlabel("Predicted confidence", labelpad=8)
        ax_h1.set_ylabel("Count", labelpad=8)
        ax_h1.tick_params(axis='both', which='major', pad=6)
        fig_h1.savefig(f"{RESULTS_BASE_PATH}/confidence_distribution_round{round_id}.pdf", bbox_inches="tight")
        plt.close(fig_h1)

        # Probabilidade da classe verdadeira (usa índices densos)
        # Garantir mapeamento denso como acima
        if (np.array_equal(np.unique(y_test), np.arange(K)) and
            np.array_equal(np.unique(y_pred), np.arange(K))):
            y_test_idx2 = np.asarray(y_test)
        else:
            y_test_idx2, _, _ = map_labels_dense(y_test, y_pred)
        true_class_probs = y_proba[np.arange(len(y_test_idx2)), y_test_idx2]
        fig_h2, ax_h2 = plt.subplots(figsize=(6.9, 3.2), constrained_layout=True)
        ax_h2.hist(true_class_probs, bins=20, edgecolor="black")
        ax_h2.set_xlabel("True-class probability", labelpad=8)
        ax_h2.set_ylabel("Count", labelpad=8)
        ax_h2.tick_params(axis='both', which='major', pad=6)
        fig_h2.savefig(f"{RESULTS_BASE_PATH}/true_class_prob_distribution_round{round_id}.pdf", bbox_inches="tight")
        plt.close(fig_h2)

def process_stacking_aggregation(round_id, X_stack, y_stack, start_time):
    """Treina/prediz, atualiza métricas globais e salva gráficos/relatórios ACM."""
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_stack, y_stack, test_size=0.2, random_state=42, stratify=y_stack
    )
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_train, y_train)
    y_pred = meta_model.predict(X_test)
    y_proba = meta_model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)

    round_accuracies.append(acc)
    round_durations.append(time.time() - start_time)

    save_plots_and_reports(X_stack, y_stack, y_test, y_pred, y_proba, round_id)
    return acc

def aggregate_with_stacking(round_id):
    clients_received = received_probs.get(round_id, {})
    if not all(len(clients_received.get(c, [])) >= 1 for c in expected_clients):
        return

    print(f"[Stacking] ✅ Todos os clientes da rodada {round_id} enviaram. Realizando stacking...")
    t_start = time.time()
    probs_list = [np.array(clients_received[client][0]["probs"]) for client in sorted(expected_clients)]
    y_stack = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])
    X_stack = np.concatenate(probs_list, axis=1)

    acc = process_stacking_aggregation(round_id, X_stack, y_stack, t_start)
    print(f"🎯 Acurácia rodada Stacking {round_id} (teste): {acc:.4f}")
    del received_probs[round_id]

def aggregate_with_GA_and_Stacking(round_id):
    clients_received = received_probs.get(round_id, {})
    if not all(len(clients_received.get(c, [])) >= 1 for c in expected_clients):
        return

    print(f"[GA+Stacking] ✅ Todos os clientes da rodada {round_id} enviaram. Realizando ensemble híbrido...")
    t_start = time.time()
    probs_list = [np.array(clients_received[client][0]["probs"]) for client in sorted(expected_clients)]
    labels = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])

    ga_weights = run_genetic_algorithm(probs_list, labels, args)
    _, X_stack, y_stack = run_hybrid_ensemble_ga_stacking(probs_list, labels, ga_weights)

    acc = process_stacking_aggregation(round_id, X_stack, y_stack, t_start)
    print(f"🎯 Acurácia rodada GA+Stacking {round_id} (teste): {acc:.4f}")
    del received_probs[round_id]

def aggregate_with_PSO_and_Stacking(round_id):
    clients_received = received_probs.get(round_id, {})
    if not all(len(clients_received.get(c, [])) >= 1 for c in expected_clients):
        return

    print(f"[PSO+Stacking] ✅ Todos os clientes da rodada {round_id} enviaram. Realizando ensemble híbrido...")
    t_start = time.time()
    probs_list = [np.array(clients_received[client][0]["probs"]) for client in sorted(expected_clients)]
    labels = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])

    pso_weights = run_pso_optimization(probs_list, labels)
    _, X_stack, y_stack = run_hybrid_ensemble_pso_stacking(probs_list, labels, pso_weights)

    acc = process_stacking_aggregation(round_id, X_stack, y_stack, t_start)
    print(f"🎯 Acurácia rodada PSO+Stacking {round_id} (teste): {acc:.4f}")
    del received_probs[round_id]

# ========= MQTT =========
def on_message(client, userdata, msg):
    print(f"\n📩 Mensagem recebida de {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        client_id = msg.topic.split('/')[0]
        round_id = payload.get("round_id", "default")

        if round_id not in received_probs:
            received_probs[round_id] = {}
        # Armazena apenas probs/labels (class_names agora vem do train_dataset)
        received_probs[round_id][client_id] = [{"probs": payload["probs"], "labels": payload["labels"]}]

        # Mapeamento de métodos para funções
        aggregation_functions = {
            "stacking": aggregate_with_stacking,
            "ga": aggregate_with_GA,
            "ga_stacking": aggregate_with_GA_and_Stacking,
            "pso": aggregate_with_PSO,
            "pso_stacking": aggregate_with_PSO_and_Stacking,
        }

        # Chama a função de agregação correspondente
        if args.ensemble_method in aggregation_functions:
            aggregation_functions[args.ensemble_method](round_id)
        else:
            print(f"❌ Método de ensemble desconhecido ou não implementado: {args.ensemble_method}.")

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"❌ Erro ao processar mensagem: {e}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"🔗 Conectado ao broker, inscrevendo-se em {MQTT_TOPIC}")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"❌ Falha na conexão com o broker (código: {rc})")

def main():
    print(f"[INIT] Conectando ao broker {MQTT_BROKER}:{MQTT_PORT}...")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    try:
        print("🚀 Servidor aguardando mensagens...")
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n🛑 Encerrando servidor MQTT.")
        client.disconnect()

if __name__ == "__main__":
    main()