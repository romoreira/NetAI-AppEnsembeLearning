# server.py (agrega tráfego por rodada e salva em server/traffic_round_R.json)
import flwr as fl
import argparse
import os
from typing import List, Tuple, Dict, Optional, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, Scalar
import numpy as np
import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import json
from torchvision import datasets
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

matplotlib.use('Agg')


def save_server_artifacts(y_true, y_pred, labels, out_dir, server_round):
    """Gera e salva os artefatos de avaliação globais do servidor."""
    server_out_dir = os.path.join(out_dir, "server")
    os.makedirs(server_out_dir, exist_ok=True)

    # 1) Relatório global
    report = classification_report(y_true, y_pred, digits=5, target_names=labels, zero_division=0)
    report_path = os.path.join(server_out_dir, f"global_report_round_{server_round}.txt")
    with open(report_path, 'w') as f:
        f.write(f"Relatório de Classificação Global - Rodada {server_round}\n\n")
        f.write(report)
    print(f"  -> Relatório global salvo em: {report_path}")

    # 2) Matrizes de confusão
    mpl.rcParams.update({
        "font.size": 16, "font.family": "serif", "font.serif": ["Times New Roman"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9
    })
    label_indices = list(range(len(labels)))

    # Contagens
    cm_counts = confusion_matrix(y_true, y_pred, labels=label_indices)
    fig_c, ax_c = plt.subplots(figsize=(6.9, 6.9), constrained_layout=True)
    disp_c = ConfusionMatrixDisplay(confusion_matrix=cm_counts, display_labels=labels)
    disp_c.plot(cmap="Greys", xticks_rotation=45, ax=ax_c, colorbar=False, values_format="d")
    ax_c.set_xlabel("Predicted label", labelpad=10);
    ax_c.set_ylabel("True label", labelpad=10)
    fig_c.savefig(os.path.join(server_out_dir, f"global_cm_counts_round_{server_round}.pdf"), bbox_inches="tight")
    plt.close(fig_c)

    # Normalizada
    cm_norm = confusion_matrix(y_true, y_pred, labels=label_indices, normalize="true")
    fig_n, ax_n = plt.subplots(figsize=(6.9, 6.9), constrained_layout=True)
    disp_n = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp_n.plot(cmap="Greys", xticks_rotation=45, ax=ax_n, colorbar=True, values_format=".2f")
    ax_n.set_xlabel("Predicted label", labelpad=10);
    ax_n.set_ylabel("True label", labelpad=10)
    fig_n.savefig(os.path.join(server_out_dir, f"global_cm_norm_round_{server_round}.pdf"), bbox_inches="tight")
    plt.close(fig_n)
    print(f"  -> Matrizes de confusão globais salvas no diretório: {server_out_dir}")


class AggregateMetricsStrategy(fl.server.strategy.FedAvg):
    def __init__(self, combo_name, **kwargs):
        self.combo_name = combo_name
        super().__init__(**kwargs)

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        if not results:
            return aggregated_loss, aggregated_metrics

        # Coletar predições e rótulos de todos os clientes (para artefatos globais)
        all_global_preds = []
        all_global_targets = []
        d = datasets.ImageFolder(root="../AIDER_split/val")
        class_names = d.classes

        for _, evaluate_res in results:
            if "predictions" in evaluate_res.metrics and "targets" in evaluate_res.metrics:
                preds_list = json.loads(evaluate_res.metrics["predictions"])
                targets_list = json.loads(evaluate_res.metrics["targets"])
                all_global_preds.extend(preds_list)
                all_global_targets.extend(targets_list)

        if all_global_preds:
            output_dir = f"results/{self.combo_name}"
            print(f"Agregando artefatos de {len(results)} clientes para a Rodada {server_round}...")
            save_server_artifacts(
                y_true=all_global_targets,
                y_pred=all_global_preds,
                labels=class_names,
                out_dir=output_dir,
                server_round=server_round
            )

        # Acurácia agregada (ponderada por num_examples)
        weighted_accuracies = [res.num_examples * res.metrics.get("accuracy", 0.0) for _, res in results]
        total_examples = sum([res.num_examples for _, res in results]) or 1
        aggregated_accuracy = sum(weighted_accuracies) / total_examples
        print(f"Round {server_round} - Acurácia Agregada: {aggregated_accuracy:.4f}\n")

        if aggregated_metrics is None:
            aggregated_metrics = {}
        aggregated_metrics["accuracy"] = aggregated_accuracy

        # ---------- NOVO: Agregar tráfego reportado pelos clientes e salvar ----------
        traffic_list = []
        total_down_fit = total_up_fit = total_down_eval = total_up_metrics_eval = 0.0

        for cp, evaluate_res in results:
            m = evaluate_res.metrics
            entry = {
                "client": getattr(cp, "cid", None),
                "bytes_down_params_fit": float(m.get("bytes_down_params_fit", 0.0)),
                "bytes_up_params_fit": float(m.get("bytes_up_params_fit", 0.0)),
                "bytes_down_params_eval": float(m.get("bytes_down_params_eval", 0.0)),
                "bytes_up_metrics_eval": float(m.get("bytes_up_metrics_eval", 0.0)),
                "num_examples": int(evaluate_res.num_examples),
                "accuracy": float(m.get("accuracy", 0.0)),
            }
            traffic_list.append(entry)
            total_down_fit += entry["bytes_down_params_fit"]
            total_up_fit += entry["bytes_up_params_fit"]
            total_down_eval += entry["bytes_down_params_eval"]
            total_up_metrics_eval += entry["bytes_up_metrics_eval"]

        server_out_dir = os.path.join(f"results/{self.combo_name}", "server")
        os.makedirs(server_out_dir, exist_ok=True)
        traffic_round = {
            "round": int(server_round),
            "num_clients": len(results),
            "totals": {
                "bytes_down_params_fit": float(total_down_fit),
                "bytes_up_params_fit": float(total_up_fit),
                "bytes_down_params_eval": float(total_down_eval),
                "bytes_up_metrics_eval": float(total_up_metrics_eval),
            },
            "per_client": traffic_list,
            "aggregated_accuracy": float(aggregated_accuracy),
        }
        with open(os.path.join(server_out_dir, f"traffic_round_{server_round}.json"), "w") as f:
            json.dump(traffic_round, f, indent=2)
        print(f"  -> Tráfego agregado salvo em server/traffic_round_{server_round}.json")

        return aggregated_loss, aggregated_metrics


def main():
    parser = argparse.ArgumentParser(description="Servidor Flower")
    parser.add_argument("--combo_name", type=str, required=True, help="Nome da combinação para salvar resultados")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min_clients", type=int, default=2)
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    args = parser.parse_args()

    strategy = AggregateMetricsStrategy(
        combo_name=args.combo_name,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )

    print(f"Iniciando servidor para a combinação: {args.combo_name}")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
