import subprocess
import time
import os
import signal
from itertools import combinations


def cleanup_processes():
    """Encontra e termina quaisquer processos de servidor ou cliente em execução."""
    print("=" * 80)
    print("🧹 Limpando quaisquer processos antigos em execução...")
    scripts_to_kill = ["server_el-mqtt.py", "client_el-mqtt.py"]
    for script_name in scripts_to_kill:
        try:
            command = f"pkill -f {script_name}"
            print(f"   -> Executando: {command}")
            subprocess.run(command, shell=True, check=False)
        except Exception as e:
            print(f"      Erro ao tentar encerrar {script_name}: {e}")
    print("✅ Limpeza concluída.")
    print("=" * 80)
    time.sleep(2)  # Pequena pausa para garantir que os processos foram encerrados


def run_ensemble_experiment(ensemble_method, client_models, combo_id, combo_name, epochs=20,
                            rounds=3, kd_epochs=1, wait_timeout_sec=300):
    """
    Runs a complete ensemble federated learning experiment for a given method
    and a specific combination of client models.
    """
    model_names = [m['model_name'] for m in client_models]
    print("=" * 80)
    print(f"🚀 STARTING EXPERIMENT (Combo {combo_id}): {', '.join(model_names)}")
    print(f"➡️ Ensemble Method: {ensemble_method.upper()}")
    print("=" * 80)

    server_command = [
        "python", "server_el-mqtt.py",
        "--broker", "localhost",
        "--port", "1883",
        "--topic", "probs",
        f"--expected_clients={len(client_models)}",
        f"--ensemble_method={ensemble_method}",
        f"--combo_name={combo_name}"
    ]
    print(f"🔥 Starting Server: {' '.join(server_command)}")
    server_process = subprocess.Popen(server_command, preexec_fn=os.setsid)
    time.sleep(5)

    client_processes = []
    try:
        for i, model_params in enumerate(client_models, 1):
            client_command = [
                "python", "client_el-mqtt.py",
                "--broker", "localhost",
                "--port", "1883",
                "--topic", "probs",
                f"--model_name={model_params['model_name']}",
                f"--optimizer={model_params['optimizer']}",
                f"--lr={model_params['lr']}",
                f"--epochs={epochs}",
                f"--batch_size={model_params['batch_size']}",
                f"--client_id={i}",
                f"--ensemble_method={ensemble_method}",
                f"--combo_name={combo_name}",
                f"--pth_path={model_params['pth_path']}",
                # === Novos parâmetros para o ciclo de distilação/KL ===
                f"--rounds={rounds}",
                f"--kd_epochs={kd_epochs}",
                f"--wait_timeout_sec={wait_timeout_sec}",
            ]
            print(f"🤖 Starting Client {i} ({model_params['model_name']}): {' '.join(client_command)}")
            client_processes.append(subprocess.Popen(client_command))

        print("\n⏳ Waiting for all clients in this combo to finish training...")
        for proc in client_processes:
            proc.wait()
        print("✅ All clients have completed their tasks for this combo.")

    finally:
        print("🛑 Terminating server process...")
        try:
            os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
            print("   -> Server process group terminated with SIGKILL.")
        except ProcessLookupError:
            print("   -> Server process was already terminated.")
        print("   -> Waiting 2 seconds for resource cleanup...")
        time.sleep(2)
        print(f"🏁 EXPERIMENT FINISHED: Combo {combo_id} with {ensemble_method.upper()}\n")


if __name__ == "__main__":
    # 1. Limpa processos antigos antes de começar
    cleanup_processes()

    # Melhores modelos + caminho do .pth correspondente
    TOP_MODELS = [
        {"model_name": "efficientnet_b0",     "batch_size": 16, "lr": 0.0015838415048311233,  "optimizer": "sgd", "pth_path": "pth/efficientnet_b0.pth"},
        {"model_name": "mobilenet_v2",        "batch_size": 16, "lr": 0.00019587277064107375, "optimizer": "sgd", "pth_path": "pth/mobilenet_v2.pth"},
        {"model_name": "resnet34",            "batch_size": 16, "lr": 0.0011026300264498465,  "optimizer": "sgd", "pth_path": "pth/resnet34.pth"},
        {"model_name": "mobilenet_v3_small",  "batch_size": 32, "lr": 0.0053303233841732884,  "optimizer": "sgd", "pth_path": "pth/mobilenet_v3_small.pth"},
        {"model_name": "squeezenet",          "batch_size": 64, "lr": 0.00101013811310212088, "optimizer": "sgd", "pth_path": "pth/squeezenet.pth"},
    ]

    ENSEMBLE_METHODS_TO_RUN = [
        "stacking",
        "pso",
        "ga",
        "pso_stacking",
        "ga_stacking"
    ]

    NUM_EPOCHS = 20

    # Hiperparâmetros do ciclo KD/KL (ajuste aqui, sem tocar no resto)
    ROUNDS = 1
    KD_EPOCHS = 1
    WAIT_TIMEOUT_SEC = 300

    model_combinations = list(combinations(TOP_MODELS, 3))
    print(f"Found {len(model_combinations)} model combinations to test.")

    for i, combo in enumerate(model_combinations, 1):
        model_names_in_combo = sorted([m['model_name'] for m in combo])
        combo_name = '-'.join(model_names_in_combo)
        for method in ENSEMBLE_METHODS_TO_RUN:
            run_ensemble_experiment(
                ensemble_method=method,
                client_models=list(combo),
                combo_id=i,
                combo_name=combo_name,
                epochs=NUM_EPOCHS,
                rounds=ROUNDS,
                kd_epochs=KD_EPOCHS,
                wait_timeout_sec=WAIT_TIMEOUT_SEC
            )

    print("🎉 All experiments for all model combinations have been completed successfully!")
