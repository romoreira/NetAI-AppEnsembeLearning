# Datasheet dos Testes

Cada pasta contém os dados e resultados de um tipo de experimento realizado. Modelos utilizados: efficientnet_b0, mobilenet_v2, mobilenet_v3_small, resnet34, squeezenet.

| Pasta                          | Descrição                                                                                   | Combinações | Épocas | Retorno servidor→clients | Log de rede |
|--------------------------------|---------------------------------------------------------------------------------------------|-------------|--------|---------------------------|-------------|
| **BASELINE**                   | Treino individual dos modelos                                                               | —           | 20     | ✗                         | ✗           |
| **STACKING**                   | Stacking puro (sem MQTT). 10 combinações de 3 modelos.                                      | 10          | —      | ✗                         | ✗           |
| **MQTT-OLD**                   | Primeiro teste MQTT, **sem retorno** do servidor → clients. 10 combinações de 3 modelos.    | 10          | —      | ✗                         | ✗           |
| **MQTT-COM-RETORNO**           | MQTT **com retorno** do servidor → clients. 10 combinações de 3 modelos.                    | 10          | —      | ✓                         | ✗           |
| **MQTT-COM-RETORNO-E-LOG-REDE**| MQTT com retorno **e monitoramento de bytes transferidos**. 10 combinações de 3 modelos.    | 10          | —      | ✓                         | ✓ (`server/traffic_round_1.json`) |
| **FEDERATED**                  | Federated Learning padrão. 5 combinações, cada uma com 3 clients por modelo. Seed não fixo. | 5           | 20     | ✓                         | ✓ (`server/traffic_round_1.json`) |

A estrutura  pastas é a seguinte:
  MQTT-TEST/
  ├─ <teste específico>/
  │   ├─ <combo>/<metodo>/
  │   │   ├─ client_<id>/
  │   │   │   └─ round_<n>/
  │   │   │       ├─ classification_before.txt / .json
  │   │   │       ├─ classification_after.txt  / .json
  │   │   │       ├─ confusion_before.png / confusion_after.png
  │   │   │       └─ mqtt_traffic.json                 # (quando há MQTT no cliente)
  │   │   └─ server/
  │   │       ├─ global_report_round_<n>.txt           # (quando aplicável)
  │   │       ├─ global_cm_counts_round_<n>.pdf
  │   │       ├─ global_cm_norm_round_<n>.pdf
  │   │       └─ traffic_round_<n>.json                # (logs de bytes no servidor)
  └─ README.md (este arquivo)
