# run_flower_experiments.py
import subprocess
import time
import os
import signal
from itertools import combinations

def cleanup_processes():
    """Encontra e termina quaisquer processos de servidor ou cliente Flower em execução."""
    print("="*80)
    print("🧹 Limpando quaisquer processos antigos do Flower em execução...")
    scripts_to_kill = ["server.py", "flower_client.py"]
    for script_name in scripts_to_kill:
        try:
            command = f"pkill -f 'python .*{script_name}'"
            print(f"   -> Executando: {command}")
            subprocess.run(command, shell=True, check=False)
        except Exception as e:
            print(f"       Erro ao tentar encerrar {script_name}: {e}")
    print("✅ Limpeza concluída.")
    print("="*80)
    time.sleep(2)

def run_flower_experiment(client_config, num_clients, experiment_id, experiment_name, rounds=5):
    """
    Executa um experimento de Aprendizagem Federada homogêneo para uma
    configuração de modelo específica.
    """
    print("="*80)
    print(f"🚀 INICIANDO EXPERIMENTO {experiment_id}: Modelo '{experiment_name}'")
    print(f"➡️  Clientes: {num_clients}, Rodadas: {rounds}")
    print("="*80)

    # Comando para iniciar o servidor
    server_command = [
        "python", "server.py",
        f"--rounds={rounds}",
        f"--min_clients={num_clients}",
        f"--combo_name={experiment_name}"
    ]
    print(f"🔥 Iniciando Servidor: {' '.join(server_command)}")
    server_process = subprocess.Popen(server_command, preexec_fn=os.setsid)
    print(f"   -> Servidor iniciado com PID do grupo: {os.getpgid(server_process.pid)}")
    time.sleep(5)

    client_processes = []
    try:
        # Inicia N clientes, todos com a mesma configuração
        for i in range(1, num_clients + 1):
            client_command = [
                "python", "flower_client.py",
                f"--client_id={i}",
                f"--model_name={client_config['model_name']}",
                f"--optimizer={client_config['optimizer']}",
                f"--lr={client_config['lr']}",
                f"--batch_size={client_config['batch_size']}",
                f"--combo_name={experiment_name}"
            ]
            print(f"🤖 Iniciando Cliente {i} ({client_config['model_name']}): {' '.join(client_command)}")
            client_processes.append(subprocess.Popen(client_command))

        print(f"\n⏳ Aguardando o servidor ({rounds} rodadas) e os clientes concluírem...")
        server_process.wait()
        print("✅ Servidor concluiu o treinamento.")

    finally:
        print("🛑 Encerrando processos...")
        try:
            print(f"   -> Enviando SIGKILL para o grupo de processos do servidor {os.getpgid(server_process.pid)}...")
            os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
            print("   -> Grupo de processos do servidor encerrado.")
        except ProcessLookupError:
            print("   -> O processo do servidor já havia sido encerrado.")
        
        for proc in client_processes:
            if proc.poll() is None:
                print(f"   -> Encerrando cliente com PID {proc.pid}...")
                proc.kill()
        
        print("   -> Aguardando 2 segundos para limpeza de recursos...")
        time.sleep(2)
        print(f"🏁 EXPERIMENTO CONCLUÍDO: {experiment_name}\n")

if __name__ == "__main__":
    cleanup_processes()

    TOP_MODELS = [
        {"model_name": "resnet34", "batch_size": 32, "lr": 0.000017236649528347996, "optimizer": "adam"},  # Elisa,
        {"model_name": "squeezenet", "batch_size": 64, "lr": 0.00007429891316456461, "optimizer": "sgd"},  # Elisa
        {"model_name": "efficientnet_b0",    "batch_size": 16, "lr": 0.0015838415048311233,   "optimizer": "sgd"},
        {"model_name": "mobilenet_v2",       "batch_size": 16, "lr": 0.00019587277064107375,  "optimizer": "sgd"},
        {"model_name": "mobilenet_v3_small", "batch_size": 32, "lr": 0.0053303233841732884,   "optimizer": "sgd"},
    ]

    NUM_ROUNDS_PER_EXPERIMENT = 1
    NUM_CLIENTS_PER_EXPERIMENT = 2
    
    print(f"Iniciando série de {len(TOP_MODELS)} experimentos com {NUM_CLIENTS_PER_EXPERIMENT} clientes cada.")

    for i, model_config in enumerate(TOP_MODELS, 1):
        experiment_name = model_config['model_name']
        
        run_flower_experiment(
            client_config=model_config,
            num_clients=NUM_CLIENTS_PER_EXPERIMENT,
            experiment_id=i,
            experiment_name=experiment_name,
            rounds=NUM_ROUNDS_PER_EXPERIMENT
        )

    print("🎉 Todos os experimentos foram concluídos com sucesso!")