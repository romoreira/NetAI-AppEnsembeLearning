import datetime
import time
import json
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from typing import Optional
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import ImageFolder
import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import seaborn as sns
from sklearn import metrics
import csv
from datetime import datetime

START = time.time()
DATE_NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

OUTPUT = 0
CLASS_NUM = 0
LOGS = []
ROUND = 0

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IAAS_ENDPOINT_LOCAL="http://localhost:8080/api/v1/training"
IAAS_ENDPOINT="https://api.iaas.emanuelm.dev/api/v1/training"

parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=str, default='3002')
parser.add_argument('--model_name', type=str, help='Give the model name')
parser.add_argument('--trainingUuid', type=str, help='Training UUID')
parser.add_argument('--numRounds', type=str, help='Number of rounds')
parser.add_argument('--dataset_name', type=str, default='biglycan')
parser.add_argument("--dataset_id", type=int, default=1, help='ID do DataSet')

args = parser.parse_args()

if(args.trainingUuid is None):
    raise Exception("You must inform the training UUID!!")


# Função para adicionar entradas ao log
def add_log(event):
    global LOGS
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOGS.append([current_time, event])

def save_logs(filename):
    global LOGS

    # Save the plot as a PDF
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, args.model_name)
    output_dir = output_dir + '/resources'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{args.model_name}_SERVER_RESOURCE_LOGS.txt')
    with open(filename, 'w', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(['Time', 'Event'])  # Cabeçalhos do CSV
        log_writer.writerows(LOGS)

def resources_usage(START, END):

    # Convertendo as strings para objetos datetime
    train_start_time_dt = datetime.strptime(START, "%Y-%m-%d %H:%M:%S")
    train_end_time_dt = datetime.strptime(END, "%Y-%m-%d %H:%M:%S")

    # Convertendo objetos datetime para timestamps Unix
    TRAIN_START_TIME = int(time.mktime(train_start_time_dt.timetuple()))
    TRAIN_END_TIME = int(time.mktime(train_end_time_dt.timetuple()))

    

    # Substituindo os valores na URL
    cpu_url = f"http://200.17.78.38:19999/api/v1/data?chart=system.cpu&options=unaligned&group=sum&units=percentage&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=3600&format=csv"

    #Coleta de RAM da ultima Hora - Disponível
    available_ram_url = f"http://200.17.78.38:19999/api/v1/data?chart=mem.available&options=unaligned&group=sum&units=percentage&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=3600&format=csv"

    #Coleta de RAM da ultima Hora - Usada (Comprometida)
    used_ram_url = f"http://200.17.78.38:19999/api/v1/data?chart=mem.committed&options=unaligned&group=sum&units=percentage&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=3600&format=csv"

    #Coleta de GPU da ultima Hora (Comsuption)
    gpu_url = f"http://200.17.78.38:19999/api/v1/data?chart=nvidia_smi.gpu_gpu-7113699b-cad5-cef4-f5fd-9ea4d34c0728_gpu_utilization&options=unaligned&group=avg&units=%25&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=86400&format=csv"

    #GPU Maquina .36
    #http://200.17.78.36:19999/api/v1/data?chart=nvidia_smi.gpu_gpu-2a009a66-14c0-49c0-5fe9-f406c6fabeed_gpu_utilization&options=unaligned&group=avg&units=%25&after=-3600&before=0&points=86400&format=csv

    # Fazendo a requisição GET
    response_cpu = requests.get(cpu_url)

    # Verificando se a requisição foi bem-sucedida
    if response_cpu.status_code == 200:
        # Salvando o conteúdo da resposta em um arquivo CSV


        # Save the plot as a PDF
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, args.dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, args.model_name)
        output_dir = output_dir + '/resources'
        os.makedirs(output_dir, exist_ok=True)  
        cpu_file = os.path.join(output_dir, f'{args.model_name}_RESOURCE_LOGS_CPU.txt')
        with open(cpu_file, 'wb') as file:
            file.write(response_cpu.content)
        print("Resource experiments (CPU Used) saved!")
    else:
        print(f"Falha na requisição. Status code: {response_cpu.status_code}")


    # Fazendo a requisição GET
    response_ram_available = requests.get(available_ram_url)

    # Verificando se a requisição foi bem-sucedida
    if response_ram_available.status_code == 200:
        # Salvando o conteúdo da resposta em um arquivo CSV


        # Save the plot as a PDF
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, args.dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, args.model_name)
        output_dir = output_dir + '/resources'
        os.makedirs(output_dir, exist_ok=True)  
        ram_available_file = os.path.join(output_dir, f'{args.model_name}_RESOURCE_LOGS_RAM_AVAILABLE.txt')

        with open(ram_available_file, 'wb') as file:
            file.write(response_ram_available.content)
        print("Resource experiments (RAM Available) saved!")
    else:
        print(f"Falha na requisição. Status code: {response_ram_available.status_code}")


        # Fazendo a requisição GET
    response_ram_used = requests.get(used_ram_url)

    # Verificando se a requisição foi bem-sucedida
    if response_ram_used.status_code == 200:
        # Salvando o conteúdo da resposta em um arquivo CSV


        # Save the plot as a PDF
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, args.dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, args.model_name)
        output_dir = output_dir + '/resources'
        os.makedirs(output_dir, exist_ok=True)  
        ram_used_file = os.path.join(output_dir, f'{args.model_name}_RESOURCE_LOGS_RAM_USED.txt')

        with open(ram_used_file, 'wb') as file:
            file.write(response_ram_used.content)
        print("Resource experiments (RAM Used) saved!")
    else:
        print(f"Falha na requisição. Status code: {response_ram_used.status_code}")


        # Fazendo a requisição GET
    response_gpu_used = requests.get(gpu_url)

    # Verificando se a requisição foi bem-sucedida
    if response_gpu_used.status_code == 200:
        # Salvando o conteúdo da resposta em um arquivo CSV


        # Save the plot as a PDF
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, args.dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, args.model_name)
        output_dir = output_dir + '/resources'
        os.makedirs(output_dir, exist_ok=True)  
        gpu_file = os.path.join(output_dir, f'{args.model_name}_RESOURCE_LOGS_GPU.txt')

        with open(gpu_file, 'wb') as file:
            file.write(response_gpu_used.content)
        print("Resource experiments (GPU Used) saved!")
    else:
        print(f"Falha na requisição. Status code: {response_gpu_used.status_code}")


def save_confusion_matrix(y_true, y_pred, class_names, output_dir, accuracy, loss, elapsed_time, model_name):
    value = ''
    if int(args.dataset_id) == 1:
        value = 'binary'
    elif int(args.dataset_id) == 2:
        value = 'macro'
    #value = 'binary' #Apagar se usar a API completa

    print("## (Server) Saving Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}, Loss: {loss:.4f}, Time: {elapsed_time:.2f} seconds')

    # Save the plot as a PDF
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, model_name)
    output_dir = output_dir + '/test'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'{model_name}_confusion_matrix_ROUND'+str(ROUND)+'.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches='tight')
    

    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, average=value)
    recall = recall_score(y_true, y_pred, average=value)
    f1 = f1_score(y_true, y_pred, average=value)

    metrics_path = os.path.join(output_dir, f'{model_name}_metrics_ROUND_'+str(ROUND)+'.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Loss: {loss:.4f}\n")
        f.write(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print("### (Server) Confusion Matrix saved at "+str(output_path))

class Net(nn.Module):

    def __init__(self, model_name, class_num, output) -> None:
        super(Net, self).__init__()
        model_ft = None
        input_size = 0
        print("Server Chosing Model: "+str(model_name))
        if model_name == "alexnet":
            self.model = models.alexnet(weights='DEFAULT')
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, output)

        elif model_name == "resnet":
            self.model = models.resnet50(weights='DEFAULT')
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, output)

        elif model_name == "vgg":
            self.model = models.vgg11_bn(weights='DEFAULT')
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, output)

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            self.model = models.squeezenet1_0(weights='DEFAULT')
            self.model.classifier[1] = nn.Conv2d(512, output, kernel_size=(1,1), stride=(1,1))
            self.model.num_classes = class_num

        elif model_name == "densenet":
            """ Densenet
            """
            self.model = models.densenet121(weights='DEFAULT')
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, output)

        elif model_name == "mobilenet":
            """ MobileNet V2
            """
            self.model = models.mobilenet_v2(weights='DEFAULT')
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, output)
  
        else:
            raise ValueError(f"Modelo não suportado: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

    

def generate_classification_report(model, dataloader, class_names, model_name):
    # Define o modelo para o dispositivo correto (CPU ou GPU)
    model = model.to(DEVICE)
    model.eval()
    
    # Inicializa as variáveis de predições e rótulos verdadeiros
    all_preds = torch.tensor([], dtype=torch.long, device=DEVICE)
    all_labels = torch.tensor([], dtype=torch.long, device=DEVICE)

    # Realiza a predição para cada lote de dados no dataloader
    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # Adiciona as predições e rótulos verdadeiros às variáveis criadas anteriormente
        all_preds = torch.cat((all_preds, preds), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)
    
    # Gera o classification report com base nas predições e rótulos verdadeiros
    report = metrics.classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(),target_names=class_names,
                                           digits=4, zero_division=0)
    
    
    # Save the plot as a PDF
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, model_name)
    output_dir = output_dir + '/test'
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f'{model_name}_report_ROUND_'+str(ROUND)+'.txt')
    with open(metrics_path, 'w') as f:
        f.write(report)
        f.close()

    print("### (Server) Classification Report saved at "+str(metrics_path))
    return report


def test(net, testloader, output_dir):
    global ROUND
    ROUND = ROUND + 1
    add_log('Round: '+str(ROUND)+' Test Started')
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0.0
    batch_losses = []
    batch_accuracies = []
    true_labels = []
    predicted_labels = []
    y_true = []
    y_pred = []
    net.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(testloader), 1):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels).item()
            total_loss += loss * inputs.size(0)

            correct += (predicted == labels).sum().item()

            # Coletar perda e acurácia por batch
            batch_losses.append(loss)
            batch_accuracies.append((predicted == labels).sum().item() / inputs.size(0))

            # Collect true and predicted labels for confusion matrix
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            y_true += labels.tolist()
            y_pred += predicted.tolist()

    add_log('Round: '+str(ROUND)+' Test Ended')
    end_time = time.time()
    elapsed_time = end_time - START
    accuracy = correct / len(testloader.dataset)
    real_loss = total_loss / len(testloader.dataset)

    print(f"Test Loss: {real_loss:.4f}, Test Accuracy: {accuracy:.2%}")

    
    
    if args.dataset_id == 1:
        class_names = ["benign", "malignant"]
    elif args.dataset_id == 2:
        class_names = ["dysplasia", "oscc", "without-dysplasia"]
    #class_names = ["benign", "malignant"] # Apagar temporario
    #class_names = ["dysplasia", "oscc", "without-dysplasia"] # Apagar temporario
    
    save_confusion_matrix(y_true, y_pred, class_names, output_dir, accuracy, real_loss, elapsed_time, args.model_name)
    generate_classification_report(net, testloader, class_names, args.model_name)
    return real_loss, accuracy


def load_data(dataset_id, client_id):
    global OUTPUT
    global CLASS_NUM
    
    print("Loading Dataset ID: "+str(dataset_id))
    #dataset_id = 1 # Apagar pos é temporário
    
    if dataset_id == 1: # 1 == BiglyCan

        OUTPUT = 2
        CLASS_NUM = 2

        # Load the breast cancer dataset (modify the paths accordingly)
        input_size = 224
        data_transforms = {
            'transform': transforms.Compose([
                            transforms.Resize(size=[224, 224]),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomRotation(30),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ])
        }

        trainset = ImageFolder("./dataset/production/train", transform=data_transforms['transform'])
        testset = ImageFolder("./dataset/production/test", transform=data_transforms['transform'])
        return DataLoader(trainset, batch_size=16, shuffle=True), DataLoader(testset)
    elif dataset_id == 2:
        OUTPUT = 3
        CLASS_NUM = 3
        input_size = 224
        data_transforms = {
            'transform': transforms.Compose([
                            transforms.Resize(size=[224, 224]),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomRotation(30),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ])
        }

        trainset = ImageFolder("./dataset/production/ufes_iid/train", transform=data_transforms['transform'])
        testset = ImageFolder("./dataset/production/ufes_iid/test", transform=data_transforms['transform'])
        return DataLoader(trainset, batch_size=16, shuffle=True), DataLoader(testset)



def aggregate_probs(results: List[Dict[str, any]]) -> None:
    all_probs = []
    all_labels = []
    
    if not results:
        print("No results to aggregate.")
        return
    
    for result in results:
        client_data = result[1]  # Acessa o dicionário retornado pelo cliente
        if 'probs' in client_data and 'labels' in client_data:
            all_probs.append(client_data['probs'])
            all_labels.append(client_data['labels'])

    # Concatena as probabilidades e rótulos
    if all_probs and all_labels:
        aggregated_probs = np.concatenate(all_probs)
        aggregated_labels = np.concatenate(all_labels)

        # Exemplo de como armazenar ou processar
        print("Probabilidades agregadas:", aggregated_probs)
        print("Rótulos agregados:", aggregated_labels)

    # Aqui podemos implementar alguma lógica adicional de agregação ou salvar os dados





# The `evaluate` function will be by Flower called after every round
def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar], 
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = Net(model_name=args.model_name, class_num=CLASS_NUM, output=OUTPUT).to(DEVICE)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader, output_dir=".")
    torch.save(net.state_dict(), 'server_model_aggregated.pth')
    accuracy_percent = accuracy * 100  # Multiplica a precisão por 100 para obter o valor percentual
    print(f"\n### (Evaluate) Server-side evaluation loss {loss} / accuracy {accuracy_percent:.2f}% ###\n")
    with open('1_biglycan_alexnet_server_accuracy.txt', 'a') as f:
        f.write(f"{accuracy_percent:.2f})\n")
    return loss, {"accuracy": accuracy}

def simple_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Calculate the average accuracy of each client
    accuracies = [m["accuracy"] for _, m in metrics]
    average_accuracy = sum(accuracies) / len(accuracies)

    # Return the average accuracy as the evaluation result
    return {"accuracy": average_accuracy}

def default_evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    
    # Verificar se o config contém as probabilidades
    print("Config: ", config)
    if "metrics" in config:
        aggregate_probs(config["metrics"])
    
    net = Net(model_name=args.model_name, class_num=CLASS_NUM, output=OUTPUT).to(DEVICE)
    _, testloader = load_data(args.dataset_id, "")
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader,"./")
    print(f"Default Evaluate: Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    for num_examples, m in metrics:
        print(f"Weighted Average> Examples: {num_examples}, Accuracy: {m['accuracy']:.2%}")
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_fn=default_evaluate, evaluate_metrics_aggregation_fn=weighted_average)



_, testloader = load_data(args.dataset_id, "")
net = Net(model_name=args.model_name, class_num=CLASS_NUM, output=OUTPUT).to(DEVICE)


TRAIN_START_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
add_log('FL Server Train Started')
# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:"+args.port,
    config=fl.server.ServerConfig(num_rounds=int(args.numRounds)),
    strategy=strategy
)
TRAIN_END_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#resources_usage(TRAIN_START_TIME, TRAIN_END_TIME)
add_log('FL Server Train Ended')
save_logs('execution_log.csv')

#nvidia_smi.gpu_gpu-7113699b-cad5-cef4-f5fd-9ea4d34c0728_gpu_utilization