import paho.mqtt.client as mqtt
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import argparse
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from optimization import run_genetic_algorithm, evaluate_weighted_probs, run_hybrid_ensemble_ga_stacking, run_pso_optimization, run_hybrid_ensemble_pso_stacking

# Argumentos via CLI
parser = argparse.ArgumentParser()
parser.add_argument('--broker', type=str, default='localhost', help='Broker IP or hostname')
parser.add_argument('--port', type=int, default=1883, help='Broker port')
parser.add_argument('--topic', type=str, default='probs', help='MQTT topic to subscribe')
parser.add_argument('--expected_clients', type=str, required=True, help='Number of expected clients')
parser.add_argument('--ensemble_method', type=str, help='Ensemble method to use', required=True)
parser.add_argument('--combo_name', type=str, required=True, help='Identifier for the model combination')
args = parser.parse_args()

# --- Caminho Base para Salvar Resultados ---
# Esta única linha cria o diretório necessário, tornando o bloco if/elif obsoleto.
RESULTS_BASE_PATH = f"results/{args.combo_name}/{args.ensemble_method}"
os.makedirs(RESULTS_BASE_PATH, exist_ok=True)

# Configuração
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
start_times = {}

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
    
    best_weights = run_genetic_algorithm(probs_list, labels)
    acc = evaluate_weighted_probs(probs_list, best_weights, labels, args.ensemble_method)
    duration = time.time() - t_start
    print(f"[GA - RESULT] - Time Elapsed: {duration:.3f}s - Accuracy with GA: {acc:.4f}")

    # (Nota: Esta função não gera gráficos no código original)

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
    acc = evaluate_weighted_probs(probs_list, best_weights, y_stack, args.ensemble_method)
    duration = time.time() - t_start
    print(f"[PSO - RESULT] - Time Elapsed: {duration:.3f}s - Accuracy with PSO: {acc:.4f}")

    round_accuracies.append(acc)
    round_durations.append(duration)
    message_counts.append({c: len(clients_received.get(c, [])) for c in expected_clients})

    # --- Salvando Gráficos ---
    plt.figure()
    plt.plot(round_accuracies, marker='o')
    plt.title("Accuracy per Round (PSO)")
    plt.savefig(f"{RESULTS_BASE_PATH}/accuracy_per_round.png")

    plt.figure()
    plt.plot(round_durations, marker='x')
    plt.title("Aggregation Time per Round (PSO)")
    plt.savefig(f"{RESULTS_BASE_PATH}/aggregation_time.png")

    combined = np.zeros_like(probs_list[0])
    for i, probs in enumerate(probs_list):
        combined += best_weights[i] * np.array(probs)
    avg_probs = np.mean(combined, axis=0)

    plt.figure()
    plt.bar(np.arange(len(avg_probs)), avg_probs)
    plt.title("Average Weighted Probabilities per Class (PSO)")
    plt.savefig(f"{RESULTS_BASE_PATH}/avg_probs_round{round_id}.png")
    
    del received_probs[round_id]
    print(f"[PSO] Dados da rodada {round_id} limpos.")

def save_plots_and_reports(X_stack, y_stack, y_test, y_pred, y_proba, round_id):
    """Função auxiliar para salvar todos os gráficos e relatórios comuns."""
    
    # Relatório de Classificação
    report = classification_report(y_test, y_pred, digits=4)
    with open(f"{RESULTS_BASE_PATH}/classification_report_round{round_id}.txt", "w") as f:
        f.write(f"Classification Report - Round {round_id}\n\n{report}")

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Round {round_id}")
    plt.savefig(f"{RESULTS_BASE_PATH}/confusion_matrix_round{round_id}.png")

    # Acurácia por Rodada
    plt.figure()
    plt.plot(round_accuracies, marker='o')
    plt.title("Accuracy per Round")
    plt.savefig(f"{RESULTS_BASE_PATH}/accuracy_per_round.png")

    # Tempo de Agregação por Rodada
    plt.figure()
    plt.plot(round_durations, marker='x')
    plt.title("Aggregation Time per Round")
    plt.savefig(f"{RESULTS_BASE_PATH}/aggregation_time.png")

    # Distribuição de Confiança
    plt.figure()
    confidences = np.max(y_proba, axis=1)
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title("Prediction Confidence Distribution")
    plt.savefig(f"{RESULTS_BASE_PATH}/confidence_distribution_round{round_id}.png")

    # Probabilidade da Classe Verdadeira
    true_class_probs = y_proba[np.arange(len(y_test)), y_test]
    plt.figure()
    plt.hist(true_class_probs, bins=20, color='orange', edgecolor='black')
    plt.title("True Class Probability Distribution")
    plt.savefig(f"{RESULTS_BASE_PATH}/true_class_prob_distribution_round{round_id}.png")
    
    # Limpa as figuras para evitar sobreposição
    plt.close('all')

def process_stacking_aggregation(round_id, X_stack, y_stack, start_time):
    """Função auxiliar para processar a agregação de stacking (treino e predição)."""
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
    if not all(len(clients_received.get(c, [])) >= 1 for c in expected_clients): return

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
    if not all(len(clients_received.get(c, [])) >= 1 for c in expected_clients): return
    
    print(f"[GA+Stacking] ✅ Todos os clientes da rodada {round_id} enviaram. Realizando ensemble híbrido...")
    t_start = time.time()
    probs_list = [np.array(clients_received[client][0]["probs"]) for client in sorted(expected_clients)]
    labels = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])
    
    ga_weights = run_genetic_algorithm(probs_list, labels)
    _, X_stack, y_stack = run_hybrid_ensemble_ga_stacking(probs_list, labels, ga_weights)
    
    acc = process_stacking_aggregation(round_id, X_stack, y_stack, t_start)
    print(f"🎯 Acurácia rodada GA+Stacking {round_id} (teste): {acc:.4f}")
    del received_probs[round_id]

def aggregate_with_PSO_and_Stacking(round_id):
    clients_received = received_probs.get(round_id, {})
    if not all(len(clients_received.get(c, [])) >= 1 for c in expected_clients): return

    print(f"[PSO+Stacking] ✅ Todos os clientes da rodada {round_id} enviaram. Realizando ensemble híbrido...")
    t_start = time.time()
    probs_list = [np.array(clients_received[client][0]["probs"]) for client in sorted(expected_clients)]
    labels = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])
    
    pso_weights = run_pso_optimization(probs_list, labels)
    _, X_stack, y_stack = run_hybrid_ensemble_pso_stacking(probs_list, labels, pso_weights)

    acc = process_stacking_aggregation(round_id, X_stack, y_stack, t_start)
    print(f"🎯 Acurácia rodada PSO+Stacking {round_id} (teste): {acc:.4f}")
    del received_probs[round_id]

def on_message(client, userdata, msg):
    print(f"\n📩 Mensagem recebida de {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        client_id = msg.topic.split('/')[0]
        round_id = payload.get("round_id", "default")

        if round_id not in received_probs:
            received_probs[round_id] = {}
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

    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ Erro ao processar mensagem: {e}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"🔗 Conectado ao broker, inscrevendo-se em {MQTT_TOPIC}")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"❌ Falha na conexão com o broker (código: {rc})")

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[INIT] Conectando ao broker {MQTT_BROKER}:{MQTT_PORT}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    try:
        print("🚀 Servidor aguardando mensagens...")
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n🛑 Encerrando servidor MQTT.")
        client.disconnect()

if __name__ == "__main__":
    main()