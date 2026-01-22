import subprocess
import os

def run_baseline_model(model_name, optimizer, lr, epochs, batch_size):
    command = [
        "python",
        "baseline.py",
        f"--model_name={model_name}",
        f"--optimizer={optimizer}",
        f"--lr={lr}",
        f"--epochs={epochs}",
        f"--batch_size={batch_size}"
    ]
    print(f"--- Executando: {' '.join(command)} ---")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"Erro ao executar o modelo {model_name}. Código de saída: {process.returncode}")
    except FileNotFoundError:
        print("Erro: O arquivo 'baseline.py' não foi encontrado. Certifique-se de que está no diretório correto.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    print(f"--- Finalizado a execução para o modelo: {model_name} ---\n")

if __name__ == "__main__":
    models_to_run = [
        {"model_name": "efficientnet_b0", "batch_size": 16, "lr": 0.0015838415048311233, "optimizer": "sgd", "epochs": 20},
        {"model_name": "mobilenet_v2", "batch_size": 16, "lr": 0.00019587277064107375, "optimizer": "sgd", "epochs": 20},
        {"model_name": "resnet34", "batch_size": 16, "lr": 0.0011026300264498465, "optimizer": "sgd", "epochs": 20},
        {"model_name": "mobilenet_v3_small", "batch_size": 32, "lr": 0.0053303233841732884, "optimizer": "sgd", "epochs": 20},
        {"model_name": "squeezenet", "batch_size": 64, "lr": 0.00101013811310212088, "optimizer": "sgd", "epochs": 20},
    ]

    for model_params in models_to_run:
        run_baseline_model(**model_params)

    print("Todas as execuções dos modelos baseline foram concluídas.")