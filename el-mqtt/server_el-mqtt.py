import paho.mqtt.client as mqtt
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Configuração
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "#"

# Espera por esses clientes em cada rodada
expected_clients = {"client1", "client2"}  # ou use set() e atualize dinamicamente
received_probs = {}
y_true = None

def try_aggregate(round_id):
    global received_probs, y_true

    clients_received = received_probs.get(round_id, {})
    if expected_clients.issubset(clients_received.keys()):
        print(f"✅ Todos os clientes da rodada {round_id} enviados. Realizando stacking...")

        # Monta matriz empilhada
        X_stack = np.concatenate([clients_received[c] for c in sorted(clients_received)], axis=1)

        # Treina metamodelo
        meta_model = LogisticRegression(max_iter=1000)
        meta_model.fit(X_stack, y_true)

        # Avalia
        y_pred = meta_model.predict(X_stack)
        acc = accuracy_score(y_true, y_pred)
        print(f"🎯 Acurácia rodada {round_id}: {acc:.4f}")

        # Limpa só essa rodada
        del received_probs[round_id]

def on_message(client, userdata, msg):
    global y_true

    topic = msg.topic  # Ex: "client1/probs"
    payload = json.loads(msg.payload.decode())

    client_id = topic.split('/')[0]
    probs = np.array(payload["probs"])

    round_id = payload.get("round_id", "default")

    if y_true is None:
        y_true = np.array(payload["labels"])

    # Inicializa entrada da rodada se não existir
    if round_id not in received_probs:
        received_probs[round_id] = {}

    received_probs[round_id][client_id] = probs
    print(f"📩 Recebido de {client_id} (rodada {round_id})")

    try_aggregate(round_id)

def main():
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(MQTT_TOPIC)
    print("🚀 Servidor aguardando mensagens...")
    client.loop_forever()

if __name__ == "__main__":
    main()