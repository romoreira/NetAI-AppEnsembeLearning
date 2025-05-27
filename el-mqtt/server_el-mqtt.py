import paho.mqtt.client as mqtt
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import argparse
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from optimization import run_genetic_algorithm, evaluate_weighted_probs, run_hybrid_ensemble_ga_stacking, run_pso_optimization, run_hybrid_ensemble_pso_stacking
from sklearn.metrics import classification_report

# Argumentos via CLI
parser = argparse.ArgumentParser()
parser.add_argument('--broker', type=str, default='localhost', help='Broker IP or hostname')
parser.add_argument('--port', type=int, default=1883, help='Broker port')
parser.add_argument('--topic', type=str, default='probs', help='MQTT topic to subscribe')
parser.add_argument('--expected_clients', type=str, required=True, help='Lista de clientes esperados para cada rodada')
parser.add_argument('--ensemble_method', type=str, help='Método de ensemble a ser usado (ga, stacking, voting, baseline)', required=True)
args = parser.parse_args()

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

# Salvar gráficos
os.makedirs("results", exist_ok=True)
if args.ensemble_method == "ga":   # Se for GA, criar pasta específica
    os.makedirs("results/ga", exist_ok=True)
elif args.ensemble_method == "stacking":  # Se for Stacking, criar pasta específica
    os.makedirs("results/stacking", exist_ok=True)
elif args.ensemble_method == "voting":  # Se for Voting, criar pasta específica
    os.makedirs("results/voting", exist_ok=True)
elif args.ensemble_method == "ga_stacking":  # Se for Voting, criar pasta específica
    os.makedirs("results/ga_stacking", exist_ok=True)
elif args.ensemble_method == "pso":  # Se for Voting, criar pasta específica
    os.makedirs("results/pso_stacking", exist_ok=True)
elif args.ensemble_method == "pso_stacking":  # Se for Voting, criar pasta específica
    os.makedirs("results/pso", exist_ok=True)
else:
    os.makedirs("results/baseline", exist_ok=True)

start_times = {}




def aggregate_with_GA(round_id):
    print(f"[GA] Verificando se todos os clientes enviaram para a rodada {round_id}...")
    clients_received = received_probs.get(round_id, {})
    for c in expected_clients:
        print(f"[GA] - {c}: {len(clients_received.get(c, []))} mensagens")

    ready = all(len(clients_received.get(c, [])) >= 1 for c in expected_clients)
    if not ready:
        print(f"[GA] ⏳ Ainda aguardando mensagens...")
        return

    print(f"[GA] ✅ Todos os clientes da rodada {round_id} enviaram. Realizando stacking...")

    t_start = time.time()

    probs_list = []
    for client in sorted(expected_clients):
        client_data = clients_received[client][0]
        probs = np.array(client_data["probs"])
        probs_list.append(probs)

    X_stack = np.concatenate(probs_list, axis=1)
    y_stack = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])

    print(f"[GA] Forma da matriz X empilhada: {X_stack.shape}")
    print(f"[GA] Rótulos únicos: {set(y_stack)}")

    best_weights = run_genetic_algorithm(probs_list, clients_received[sorted(expected_clients)[0]][0]["labels"])
    acc = evaluate_weighted_probs(probs_list, best_weights, clients_received[sorted(expected_clients)[0]][0]["labels"], args.ensemble_method)
    duration  = time.time() - t_start
    print(f"[GA - RESULT] - Time Enlapsed: {duration:.3f} - Accuracy with GA: {acc:.4f}")

def aggregate_with_PSO(round_id):
    print(f"[PSO] Verificando se todos os clientes enviaram para a rodada {round_id}...")
    clients_received = received_probs.get(round_id, {})
    for c in expected_clients:
        print(f"[PSO] - {c}: {len(clients_received.get(c, []))} mensagens")

    ready = all(len(clients_received.get(c, [])) >= 1 for c in expected_clients)
    if not ready:
        print(f"[PSO] ⏳ Ainda aguardando mensagens...")
        return

    print(f"[PSO] ✅ Todos os clientes da rodada {round_id} enviaram. Realizando agregação com PSO...")

    t_start = time.time()

    probs_list = []
    for client in sorted(expected_clients):
        client_data = clients_received[client][0]
        probs = np.array(client_data["probs"])
        probs_list.append(probs)

    y_stack = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])

    print(f"[PSO] Forma da matriz de entrada: ({len(y_stack)}, {len(probs_list) * probs_list[0].shape[1]})")
    print(f"[PSO] Rótulos únicos: {set(y_stack)}")

    # Otimização com PSO
    best_weights = run_pso_optimization(probs_list, y_stack)

    # Avaliação
    acc = evaluate_weighted_probs(probs_list, best_weights, y_stack, args.ensemble_method)
    duration = time.time() - t_start
    print(f"[PSO - RESULT] - Time Elapsed: {duration:.3f}s - Accuracy with PSO: {acc:.4f}")

    round_accuracies.append(acc)
    round_durations.append(duration)
    message_counts.append({c: len(clients_received[c]) for c in expected_clients})




    # Gráfico de acurácia por rodada
    plt.figure()
    plt.plot(round_accuracies, marker='o')
    plt.title("Accuracy per Round (PSO)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/accuracy_per_round.png")

    # Gráfico de mensagens recebidas por cliente
    plt.figure()
    for c in expected_clients:
        plt.plot([m[c] for m in message_counts], label=c)
    plt.title("Messages Received per Client")
    plt.xlabel("Round")
    plt.ylabel("# Messages")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/messages_per_client.png")

    # Gráfico de tempo por rodada
    plt.figure()
    plt.plot(round_durations, marker='x')
    plt.title("Aggregation Time per Round (PSO)")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/aggregation_time.png")

    # Gráfico de distribuição dos rótulos
    plt.figure()
    unique, counts = np.unique(y_stack, return_counts=True)
    plt.bar(unique, counts)
    plt.title("Label Distribution (y_stack)")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/label_distribution_round{round_id}.png")

    # Gráfico de médias das probabilidades por classe
    combined = np.zeros_like(probs_list[0])
    for i, probs in enumerate(probs_list):
        combined += best_weights[i] * np.array(probs)
    avg_probs = np.mean(combined, axis=0)

    plt.figure()
    plt.bar(np.arange(len(avg_probs)), avg_probs)
    plt.title("Average Weighted Probabilities per Class (PSO)")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/avg_probs_round{round_id}.png")

    del received_probs[round_id]
    print(f"[PSO] Dados da rodada {round_id} limpos.")

def aggregate_with_GA_and_Stacking(round_id):
    print(f"[GA+Stacking] Verificando se todos os clientes enviaram para a rodada {round_id}...")
    clients_received = received_probs.get(round_id, {})
    for c in expected_clients:
        print(f"[GA+Stacking] - {c}: {len(clients_received.get(c, []))} mensagens")

    ready = all(len(clients_received.get(c, [])) >= 1 for c in expected_clients)
    if not ready:
        print(f"[GA+Stacking] ⏳ Ainda aguardando mensagens...")
        return

    print(f"[GA+Stacking] ✅ Todos os clientes da rodada {round_id} enviaram. Realizando stacking...")

    t_start = time.time()

    probs_list = []
    for client in sorted(expected_clients):
        client_data = clients_received[client][0]
        probs = np.array(client_data["probs"])
        probs_list.append(probs)

    # Call GA to optimize weights
    # Extrai os rótulos da rodada
    labels = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])

    # Executa o GA para otimizar pesos
    ga_weights = run_genetic_algorithm(probs_list, labels)

    # Executa ensemble híbrido (GA + Stacking)
    acc_hybrid, X_stack, y_stack = run_hybrid_ensemble_ga_stacking(probs_list, labels, ga_weights)


    print(f"[GA+Stacking] Forma da matriz X empilhada: {X_stack.shape}")
    print(f"[GA+Stacking] Rótulos únicos: {set(y_stack)}")

    # Split dos dados para avaliação justa
    X_train, X_test, y_train, y_test = train_test_split(
        X_stack, y_stack, test_size=0.2, random_state=42, stratify=y_stack
    )

    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_train, y_train)
    y_pred = meta_model.predict(X_test)
    y_proba = meta_model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Acurácia rodada GA+Stacking {round_id} (teste): {acc:.4f}")

    round_accuracies.append(acc)
    round_durations.append(time.time() - t_start)
    message_counts.append({c: len(clients_received[c]) for c in expected_clients})


    #Saving classification report
    report = classification_report(y_test, y_pred, digits=4)
    report_path = f"results/{args.ensemble_method}/classification_report_round{round_id}.txt"

    with open(report_path, "w") as f:
        f.write(f"Classification Report - Round {round_id}\n\n")
        f.write(report)

    # Gráfico de acurácia por rodada
    plt.figure()
    plt.plot(round_accuracies, marker='o')
    plt.title("Accuracy per Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/accuracy_per_round.png")

    # Gráfico de mensagens recebidas por cliente
    plt.figure()
    for c in expected_clients:
        plt.plot([m[c] for m in message_counts], label=c)
    plt.title("Messages Received per Client")
    plt.xlabel("Round")
    plt.ylabel("# Messages")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/messages_per_client.png")

    # Gráfico de tempo por rodada
    plt.figure()
    plt.plot(round_durations, marker='x')
    plt.title("Aggregation Time per Round")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/aggregation_time.png")

    # Gráfico de distribuição dos rótulos
    plt.figure()
    unique, counts = np.unique(y_stack, return_counts=True)
    plt.bar(unique, counts)
    plt.title("Label Distribution (y_stack)")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/label_distribution_round{round_id}.png")

    # Gráfico de médias das probabilidades por classe
    plt.figure()
    avg_probs = np.mean(X_stack, axis=0)
    plt.bar(np.arange(len(avg_probs)), avg_probs)
    plt.title("Average Stacked Probabilities per Class")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/avg_probs_round{round_id}.png")

    # Confusion matrix com dados de teste
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Round {round_id}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"results/{args.ensemble_method}/confusion_matrix_round{round_id}.png")

    # 🔹 Gráfico 1: Distribuição das confianças nas predições (probabilidade máxima)
    plt.figure()
    confidences = np.max(y_proba, axis=1)
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Max predicted probability")
    plt.ylabel("Number of samples")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/confidence_distribution_round{round_id}.png")

    # 🔹 Gráfico 2: Probabilidade da classe verdadeira
    true_class_probs = y_proba[np.arange(len(y_test)), y_test]
    plt.figure()
    plt.hist(true_class_probs, bins=20, color='orange', edgecolor='black')
    plt.title("True Class Probability Distribution")
    plt.xlabel("Predicted probability for true class")
    plt.ylabel("Number of samples")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/true_class_prob_distribution_round{round_id}.png")

    del received_probs[round_id]
    print(f"[AGG+Hybrid] Dados da rodada {round_id} limpos.")


def aggregate_with_PSO_and_Stacking(round_id):
    print(f"[AGG+PSO+Stacking] Verificando se todos os clientes enviaram para a rodada {round_id}...")
    clients_received = received_probs.get(round_id, {})
    for c in expected_clients:
        print(f"[AGG+PSO+Stacking] - {c}: {len(clients_received.get(c, []))} mensagens")

    ready = all(len(clients_received.get(c, [])) >= 1 for c in expected_clients)
    if not ready:
        print(f"[AGG+PSO+Stacking] ⏳ Ainda aguardando mensagens...")
        return

    print(f"[AGG+PSO+Stacking] ✅ Todos os clientes da rodada {round_id} enviaram. Realizando stacking...")

    t_start = time.time()

    probs_list = []
    for client in sorted(expected_clients):
        client_data = clients_received[client][0]
        probs = np.array(client_data["probs"])
        probs_list.append(probs)

    # Call PSO to optimize weights
    # Extrai os rótulos da rodada
    labels = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])

    # Executa o GA para otimizar pesos
    pso_weights = run_pso_optimization(probs_list, labels)


    # Executa ensemble híbrido (GA + Stacking)
    acc_hybrid, X_stack, y_stack = run_hybrid_ensemble_pso_stacking(probs_list, labels, pso_weights)


    print(f"[AGG+PSO+Stacking] Forma da matriz X empilhada: {X_stack.shape}")
    print(f"[AGG+PSO+Stacking] Rótulos únicos: {set(y_stack)}")

    # Split dos dados para avaliação justa
    X_train, X_test, y_train, y_test = train_test_split(
        X_stack, y_stack, test_size=0.2, random_state=42, stratify=y_stack
    )

    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_train, y_train)
    y_pred = meta_model.predict(X_test)
    y_proba = meta_model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Acurácia rodada PSO+Stacking {round_id} (teste): {acc:.4f}")

    round_accuracies.append(acc)
    round_durations.append(time.time() - t_start)
    message_counts.append({c: len(clients_received[c]) for c in expected_clients})


    #Saving classification report
    report = classification_report(y_test, y_pred, digits=4)
    report_path = f"results/{args.ensemble_method}/classification_report_round{round_id}.txt"

    with open(report_path, "w") as f:
        f.write(f"Classification Report - Round {round_id}\n\n")
        f.write(report)

    # Gráfico de acurácia por rodada
    plt.figure()
    plt.plot(round_accuracies, marker='o')
    plt.title("Accuracy per Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/accuracy_per_round.png")

    # Gráfico de mensagens recebidas por cliente
    plt.figure()
    for c in expected_clients:
        plt.plot([m[c] for m in message_counts], label=c)
    plt.title("Messages Received per Client")
    plt.xlabel("Round")
    plt.ylabel("# Messages")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/messages_per_client.png")

    # Gráfico de tempo por rodada
    plt.figure()
    plt.plot(round_durations, marker='x')
    plt.title("Aggregation Time per Round")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/aggregation_time.png")

    # Gráfico de distribuição dos rótulos
    plt.figure()
    unique, counts = np.unique(y_stack, return_counts=True)
    plt.bar(unique, counts)
    plt.title("Label Distribution (y_stack)")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/label_distribution_round{round_id}.png")

    # Gráfico de médias das probabilidades por classe
    plt.figure()
    avg_probs = np.mean(X_stack, axis=0)
    plt.bar(np.arange(len(avg_probs)), avg_probs)
    plt.title("Average Stacked Probabilities per Class")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/avg_probs_round{round_id}.png")

    # Confusion matrix com dados de teste
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Round {round_id}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"results/{args.ensemble_method}/confusion_matrix_round{round_id}.png")

    # 🔹 Gráfico 1: Distribuição das confianças nas predições (probabilidade máxima)
    plt.figure()
    confidences = np.max(y_proba, axis=1)
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Max predicted probability")
    plt.ylabel("Number of samples")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/confidence_distribution_round{round_id}.png")

    # 🔹 Gráfico 2: Probabilidade da classe verdadeira
    true_class_probs = y_proba[np.arange(len(y_test)), y_test]
    plt.figure()
    plt.hist(true_class_probs, bins=20, color='orange', edgecolor='black')
    plt.title("True Class Probability Distribution")
    plt.xlabel("Predicted probability for true class")
    plt.ylabel("Number of samples")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/true_class_prob_distribution_round{round_id}.png")

    del received_probs[round_id]
    print(f"[AGG+Hybrid] Dados da rodada {round_id} limpos.")

def aggregate_with_stacking(round_id):
    print(f"[Stacking] Verificando se todos os clientes enviaram para a rodada {round_id}...")
    clients_received = received_probs.get(round_id, {})
    for c in expected_clients:
        print(f"[Stacking] - {c}: {len(clients_received.get(c, []))} mensagens")

    ready = all(len(clients_received.get(c, [])) >= 1 for c in expected_clients)
    if not ready:
        print(f"[Stacking] ⏳ Ainda aguardando mensagens...")
        return

    print(f"[Stacking] ✅ Todos os clientes da rodada {round_id} enviaram. Realizando stacking...")

    t_start = time.time()

    probs_list = []
    for client in sorted(expected_clients):
        client_data = clients_received[client][0]
        probs = np.array(client_data["probs"])
        probs_list.append(probs)

    X_stack = np.concatenate(probs_list, axis=1)
    y_stack = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])

    print(f"[Stacking] Forma da matriz X empilhada: {X_stack.shape}")
    print(f"[Stacking] Rótulos únicos: {set(y_stack)}")

    # Split dos dados para avaliação justa
    X_train, X_test, y_train, y_test = train_test_split(
        X_stack, y_stack, test_size=0.2, random_state=42, stratify=y_stack
    )

    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_train, y_train)
    y_pred = meta_model.predict(X_test)
    y_proba = meta_model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Acurácia rodada Stacking {round_id} (teste): {acc:.4f}")

    round_accuracies.append(acc)
    round_durations.append(time.time() - t_start)
    message_counts.append({c: len(clients_received[c]) for c in expected_clients})

    #Saving classification report
    report = classification_report(y_test, y_pred, digits=4)
    report_path = f"results/{args.ensemble_method}/classification_report_round{round_id}.txt"

    with open(report_path, "w") as f:
        f.write(f"Classification Report - Round {round_id}\n\n")
        f.write(report)


    # Gráfico de acurácia por rodada
    plt.figure()
    plt.plot(round_accuracies, marker='o')
    plt.title("Accuracy per Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/accuracy_per_round.png")

    # Gráfico de mensagens recebidas por cliente
    plt.figure()
    for c in expected_clients:
        plt.plot([m[c] for m in message_counts], label=c)
    plt.title("Messages Received per Client")
    plt.xlabel("Round")
    plt.ylabel("# Messages")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/messages_per_client.png")

    # Gráfico de tempo por rodada
    plt.figure()
    plt.plot(round_durations, marker='x')
    plt.title("Aggregation Time per Round")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/aggregation_time.png")

    # Gráfico de distribuição dos rótulos
    plt.figure()
    unique, counts = np.unique(y_stack, return_counts=True)
    plt.bar(unique, counts)
    plt.title("Label Distribution (y_stack)")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/label_distribution_round{round_id}.png")

    # Gráfico de médias das probabilidades por classe
    plt.figure()
    avg_probs = np.mean(X_stack, axis=0)
    plt.bar(np.arange(len(avg_probs)), avg_probs)
    plt.title("Average Stacked Probabilities per Class")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/avg_probs_round{round_id}.png")

    # Confusion matrix com dados de teste
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Round {round_id}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"results/{args.ensemble_method}/confusion_matrix_round{round_id}.png")

    # 🔹 Gráfico 1: Distribuição das confianças nas predições (probabilidade máxima)
    plt.figure()
    confidences = np.max(y_proba, axis=1)
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Max predicted probability")
    plt.ylabel("Number of samples")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/confidence_distribution_round{round_id}.png")

    # 🔹 Gráfico 2: Probabilidade da classe verdadeira
    true_class_probs = y_proba[np.arange(len(y_test)), y_test]
    plt.figure()
    plt.hist(true_class_probs, bins=20, color='orange', edgecolor='black')
    plt.title("True Class Probability Distribution")
    plt.xlabel("Predicted probability for true class")
    plt.ylabel("Number of samples")
    plt.grid(True)
    plt.savefig(f"results/{args.ensemble_method}/true_class_prob_distribution_round{round_id}.png")

    del received_probs[round_id]
    print(f"[AGG] Dados da rodada {round_id} limpos.")

def on_message(client, userdata, msg):
    print(f"\n📩 Mensagem recebida de {msg.topic}")
    print(f"[MSG] Payload bruto: {msg.payload[:80]}...")

    if not msg.payload or msg.payload.strip() == b'':
        print(f"⚠️ Mensagem vazia recebida de {msg.topic}. Ignorando.")
        return

    try:
        payload = json.loads(msg.payload.decode())
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao decodificar JSON: {e}")
        return

    client_id = msg.topic.split('/')[0]
    probs = payload["probs"]
    labels = payload["labels"]
    round_id = payload.get("round_id", "default")

    print(f"[MSG] client_id={client_id}, round_id={round_id}, probs_len={len(probs)}")

    if round_id not in received_probs:
        received_probs[round_id] = {}
    if client_id not in received_probs[round_id]:
        received_probs[round_id][client_id] = []

    received_probs[round_id][client_id].append({"probs": probs, "labels": labels})
    print(f"[MSG] Total de mensagens de {client_id} na rodada {round_id}: {len(received_probs[round_id][client_id])}")


    if args.ensemble_method == "stacking":
        aggregate_with_stacking(round_id)
    elif args.ensemble_method == "voting":
        print(f"[VOTING] Método de ensemble 'voting' não implementado ainda.")
    elif args.ensemble_method == "baseline":
        print(f"[BASELINE] Método de ensemble 'baseline' não implementado ainda.")
    elif args.ensemble_method == "ga":
        print(f"[GA] Método de ensemble 'genetic algorithm' selecionado.")
        aggregate_with_GA(round_id)
    elif args.ensemble_method == "ga_stacking":
        print(f"[GA+Stacking] Método de ensemble 'genetic algorithm' + 'stacking' selecionado.")
        aggregate_with_GA_and_Stacking(round_id)
    elif args.ensemble_method == "pso":
        print(f"[PSO] Método de ensemble 'PSO' selecionado.")
        aggregate_with_PSO(round_id)
    elif args.ensemble_method == "pso_stacking":
        print(f"[PSO+Stacking] Método de ensemble 'PSO+Stacking' selecionado.")
        aggregate_with_PSO_and_Stacking(round_id)
    else:
        print(f"❌ Método de ensemble desconhecido: {args.ensemble_method}.")


def on_connect(client, userdata, flags, rc):
    print(f"🔗 Conectado ao broker (rc={rc})")
    if rc == 0:
        print("[MQTT] Conexão estabelecida com sucesso.")
    else:
        print(f"[MQTT] Falha na conexão. Código de retorno: {rc}")
    client.subscribe(MQTT_TOPIC)
    print(f"📡 Inscrito no tópico: {MQTT_TOPIC}")

def on_subscribe(client, userdata, mid, granted_qos):
    print(f"📥 Subscreveu com sucesso (MID={mid}, QoS={granted_qos})")

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_subscribe = on_subscribe

    print(f"[INIT] Conectando ao broker {MQTT_BROKER}:{MQTT_PORT}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    print("🚀 Servidor aguardando mensagens (loop_start)...")
    client.loop_start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Encerrando servidor MQTT")
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()