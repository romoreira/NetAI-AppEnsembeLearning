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

# Argumentos via CLI
parser = argparse.ArgumentParser()
parser.add_argument('--broker', type=str, default='localhost', help='Broker IP or hostname')
parser.add_argument('--port', type=int, default=1883, help='Broker port')
parser.add_argument('--topic', type=str, default='probs', help='MQTT topic to subscribe')
parser.add_argument('--expected_clients', type=str, required=True, help='Lista de clientes esperados para cada rodada')
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

os.makedirs("results", exist_ok=True)

start_times = {}

def try_aggregate(round_id):
    print(f"[AGG] Verificando se todos os clientes enviaram para a rodada {round_id}...")
    clients_received = received_probs.get(round_id, {})
    for c in expected_clients:
        print(f"[AGG] - {c}: {len(clients_received.get(c, []))} mensagens")

    ready = all(len(clients_received.get(c, [])) >= 1 for c in expected_clients)
    if not ready:
        print(f"[AGG] ⏳ Ainda aguardando mensagens...")
        return

    print(f"[AGG] ✅ Todos os clientes da rodada {round_id} enviaram. Realizando stacking...")

    t_start = time.time()

    probs_list = []
    for client in sorted(expected_clients):
        client_data = clients_received[client][0]
        probs = np.array(client_data["probs"])
        probs_list.append(probs)

    X_stack = np.concatenate(probs_list, axis=1)
    y_stack = np.array(clients_received[sorted(expected_clients)[0]][0]["labels"])

    print(f"[AGG] Forma da matriz X empilhada: {X_stack.shape}")
    print(f"[AGG] Rótulos únicos: {set(y_stack)}")

    # Split dos dados para avaliação justa
    X_train, X_test, y_train, y_test = train_test_split(
        X_stack, y_stack, test_size=0.2, random_state=42, stratify=y_stack
    )

    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_train, y_train)
    y_pred = meta_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Acurácia rodada {round_id} (teste): {acc:.4f}")

    round_accuracies.append(acc)
    round_durations.append(time.time() - t_start)
    message_counts.append({c: len(clients_received[c]) for c in expected_clients})

    # Gráfico de acurácia por rodada
    plt.figure()
    plt.plot(round_accuracies, marker='o')
    plt.title("Accuracy per Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("results/accuracy_per_round.png")

    # Gráfico de mensagens recebidas por cliente
    plt.figure()
    for c in expected_clients:
        plt.plot([m[c] for m in message_counts], label=c)
    plt.title("Messages Received per Client")
    plt.xlabel("Round")
    plt.ylabel("# Messages")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/messages_per_client.png")

    # Gráfico de tempo por rodada
    plt.figure()
    plt.plot(round_durations, marker='x')
    plt.title("Aggregation Time per Round")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.savefig("results/aggregation_time.png")

    # Gráfico de distribuição dos rótulos
    plt.figure()
    unique, counts = np.unique(y_stack, return_counts=True)
    plt.bar(unique, counts)
    plt.title("Label Distribution (y_stack)")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"results/label_distribution_round{round_id}.png")

    # Gráfico de médias das probabilidades por classe
    plt.figure()
    avg_probs = np.mean(X_stack, axis=0)
    plt.bar(np.arange(len(avg_probs)), avg_probs)
    plt.title("Average Stacked Probabilities per Class")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.savefig(f"results/avg_probs_round{round_id}.png")

    # Confusion matrix com dados de teste
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Round {round_id}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"results/confusion_matrix_round{round_id}.png")

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

    try_aggregate(round_id)

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