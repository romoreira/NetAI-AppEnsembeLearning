import warnings
from collections import OrderedDict
import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import time
import datetime
import random
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import ImageFolder
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns 
from sklearn import metrics
import csv
import requests
import json

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
START = time.time()
DATE_NOW = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT = 0
CLASS_NUM = 0
LOGS = []
ROUND = 0

parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--server_ip', type=str, default='127.0.0.1')
parser.add_argument('--dataset_name', type=str, default='biglycan')
parser.add_argument('--port', type=str, default='3002')
parser.add_argument('--world_size', type=int)
parser.add_argument('--rank', type=int)
parser.add_argument('--client_id', type=int)
parser.add_argument('--model_name', type=str, help='Give the model name')
parser.add_argument('--dataset', type=str, help='Nome do diretório do dataset')
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--dataset_id", type=int, default=1, help='ID do DataSet')
parser.add_argument("--batch_size", type=int, default=32, help='Batch Size do Dataset')
parser.add_argument("--optim", type=str, default='Adam', help='Optimizer to choose: Adam or SGD')
parser.add_argument("--dirArg", type=str, default='./dataset/production/train')
args = parser.parse_args()

def resources_usage(START, END):

    # Convertendo as strings para objetos datetime
    train_start_time_dt = datetime.datetime.strptime(START, "%Y-%m-%d %H:%M:%S")
    train_end_time_dt = datetime.datetime.strptime(END, "%Y-%m-%d %H:%M:%S")

    # Convertendo objetos datetime para timestamps Unix
    TRAIN_START_TIME = int(time.mktime(train_start_time_dt.timetuple()))
    TRAIN_END_TIME = int(time.mktime(train_end_time_dt.timetuple()))

    

    # Substituindo os valores na URL
    cpu_url = f"http://200.17.78.37:19999/api/v1/data?chart=system.cpu&options=unaligned&group=sum&units=percentage&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=3600&format=csv"

    #Coleta de RAM da ultima Hora - Disponível
    available_ram_url = f"http://200.17.78.37:19999/api/v1/data?chart=mem.available&options=unaligned&group=sum&units=percentage&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=3600&format=csv"

    #Coleta de RAM da ultima Hora - Usada (Comprometida)
    used_ram_url = f"http://200.17.78.37:19999/api/v1/data?chart=mem.committed&options=unaligned&group=sum&units=percentage&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=3600&format=csv"

    #Coleta de GPU da ultima Hora (Comsuption)
    gpu_url = f"http://200.17.78.37:19999/api/v1/data?chart=nvidia_smi.gpu_gpu-42091774-4461-f9da-a039-2814106b5a77_gpu_utilization&options=unaligned&group=avg&units=%25&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=86400&format=csv"

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
        cpu_file = os.path.join(output_dir, f'{args.model_name}_ClientID'+str(args.client_id)+'_RESOURCE_LOGS_CPU.txt')
        with open(cpu_file, 'wb') as file:
            file.write(response_cpu.content)
        print("Resource experiments (CPU Used) saved! Client ID_"+str(args.client_id))
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
        ram_available_file = os.path.join(output_dir, f'{args.model_name}_ClientID'+str(args.client_id)+'_RESOURCE_LOGS_RAM_AVAILABLE.txt')

        with open(ram_available_file, 'wb') as file:
            file.write(response_ram_available.content)
        print("Resource experiments (RAM Available) saved! Client ID_"+str(args.client_id))
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
        ram_used_file = os.path.join(output_dir, f'{args.model_name}_ClientID'+str(args.client_id)+'_RESOURCE_LOGS_RAM_USED.txt')

        with open(ram_used_file, 'wb') as file:
            file.write(response_ram_used.content)
        print("Resource experiments (RAM Used) saved! Client ID_"+str(args.client_id))
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
        gpu_file = os.path.join(output_dir, f'{args.model_name}_ClientID'+str(args.client_id)+'_RESOURCE_LOGS_GPU.txt')

        with open(gpu_file, 'wb') as file:
            file.write(response_gpu_used.content)
        print("Resource experiments (GPU Used) saved! Client ID_"+str(args.client_id))
    else:
        print(f"Falha na requisição. Status code: {response_gpu_used.status_code}")




# Função para adicionar entradas ao log
def add_log(event):
    global LOGS
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOGS.append([current_time, event])

# Função para salvar os logs em um arquivo CSV
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
    filename = os.path.join(output_dir, f'{args.model_name}_ClientID'+str(args.client_id)+'_RESOURCE_LOGS.txt')
    with open(filename, 'w', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(['Time', 'Event'])  # Cabeçalhos do CSV
        log_writer.writerows(LOGS)

def generate_classification_report(model, dataloader, class_names, model_name, client_id):
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
    metrics_path = os.path.join(output_dir, f'{model_name}_ClientID'+str(client_id)+'_ROUND_'+str(ROUND)+'_report.txt')
    with open(metrics_path, 'w') as f:
        f.write(report)
        f.close()

    print("### (Server) Classification Report saved at "+str(metrics_path))
    return report

def save_confusion_matrix(y_true, y_pred, class_names, output_dir, accuracy, loss, elapsed_time, model_name, client_id):
    value = ''
    if int(args.dataset_id) == 1:
        value = 'binary'
    elif int(args.dataset_id) == 2:
        value = 'macro'
    #value = 'binary' #Apagar pois e temporario

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
    
    output_path = os.path.join(output_dir, f'{model_name}_confusion_matrix_clientID_'+str(args.client_id)+'_ROUND_'+str(ROUND)+'.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches='tight')
    

    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, average=value)
    recall = recall_score(y_true, y_pred, average=value)
    f1 = f1_score(y_true, y_pred, average=value)

    metrics_path = os.path.join(output_dir, f'{model_name}_metrics_CLIENT_ID_'+str(args.client_id)+'_ROUND_'+str(ROUND)+'.txt')
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


def train(net, trainloader, epochs, output_dir, model_name):
    global ROUND
    ROUND = ROUND + 1
    add_log('Round: '+str(ROUND)+'_Client_'+str(args.client_id)+' Train Started.')
    
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = ''
    #optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) # Apagar temporario
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    
    # Lists to store loss and accuracy per epoch
    train_loss = []
    train_accuracy = []
    preds = 0
    net.train()
    for epoch in range(epochs):
        correct, total, total_loss = 0, 0, 0.0
        for inputs, labels in tqdm(trainloader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            net.to(DEVICE)

            outputs = net(inputs)

            optimizer.zero_grad()

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            #correct += torch.sum(preds == labels.data)
            correct += (torch.max(outputs.data, 1)[1] == labels.to(DEVICE)).sum().item()

        # Calculate accuracy and save loss and accuracy
        accuracy = correct / len(trainloader.dataset)
        epoch_loss = total_loss / len(trainloader.dataset)
        # accuracy = correct / total
        train_loss.append(epoch_loss)
        train_accuracy.append(accuracy)

        print(f"Epoch {epoch + 1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.2%}")

    add_log('Round_'+str(ROUND)+'_Client_'+str(args.client_id)+' Train Ended')

    # Create and save loss and accuracy plots
    epochs_range = np.arange(1, epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, 'b', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, train_accuracy, 'r', label='Training Accuracy')
    plt.plot(epochs_range, [acc for acc in train_accuracy], 'r', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Save the plot as a PDF
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, model_name)
    output_dir = output_dir + '/train'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'loss_accuracy_plots_clientID_'+str(args.client_id)+'_Round_'+str(ROUND)+'.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches='tight')
    plt.close()

def test(net, testloader, output_dir, model_name):
    global ROUND

    add_log('Round_'+str(ROUND)+'_Client_'+str(args.client_id)+' Test Start')

    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    true_labels = []
    predicted_labels = []
    y_true = []
    y_pred = []
    net.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)

            loss += criterion(outputs, labels).item() * inputs.size(0)

            correct += (predicted == labels).sum().item()

            # correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            # Collect true and predicted labels for confusion matrix
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            y_true += labels.tolist()
            y_pred += predicted.tolist()

    add_log('Client_'+str(args.client_id)+' Test Ended')

    end_time = time.time()
    elapsed_time = end_time - START
    accuracy = correct / len(testloader.dataset)
    real_loss = loss / len(testloader.dataset)

    print(f"Test Loss: {real_loss:.4f}, Test Accuracy: {accuracy:.2%}")

    class_names = ["benign", "malignant"] # Apagar temporario
    class_names = ["dysplasia", "oscc", "without-dysplasia"]
    if args.dataset_id == 1:
        class_names = ["benign", "malignant"]
    elif args.dataset_id == 2:
        class_names = ["dysplasia", "oscc", "without-dysplasia"]
    #class_names = ["benign", "malignant"] # Apagar temporario
    #class_names = ["dysplasia", "oscc", "without-dysplasia"]

    save_confusion_matrix(y_true, y_pred, class_names, output_dir, accuracy, real_loss, elapsed_time, model_name, args.client_id)
    generate_classification_report(net, testloader, class_names, model_name, args.client_id)

    return real_loss, accuracy


def load_data(dataset_id, client_id):
    global OUTPUT
    global CLASS_NUM
    
    print("Loading Dataset ID: "+str(dataset_id))
    #dataset_id = 1 # Apagar pos é temporário
    
    add_log('Client_'+str(args.client_id)+' Loading Dataset')

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

        trainset = ImageFolder("../dataset/production/ufes_iid/train", transform=data_transforms['transform'])
        testset = ImageFolder("../dataset/production/ufes_iid/test", transform=data_transforms['transform'])

        add_log('Client_'+str(args.client_id)+' Dataset Loaded')

        return DataLoader(trainset, batch_size=16, shuffle=True), DataLoader(testset)


trainloader, testloader = load_data(args.dataset_id, args.client_id)

if torch.cuda.is_available():
    print("GPU")
else:
    print("CPU")

SEED = 42


random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    # Função para extrair probabilidades
    def extract_probs(self, model, loader, device):
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=f"Extraindo probs de {model.__class__.__name__}"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(targets)
        return torch.cat(all_probs), torch.cat(all_labels)

    # Função para obter probabilidades a partir de um modelo
    def get_probabilities(self, model, loader, device):
        probs, labels = self.extract_probs(model, loader, device)
        return probs.numpy(), labels.numpy()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Ajusta os pesos recebidos
        self.set_parameters(parameters)

        # Treina o modelo
        train(net, trainloader, epochs=int(args.epochs), output_dir="./results", model_name=args.model_name)

        # Obtém os pesos após o treinamento
        weights = self.get_parameters(config={})

        # Extrai as probabilidades
        probs, labels = self.get_probabilities(net, testloader, DEVICE)

        # Achata as listas antes de enviá-las
        probs_flat = [p for sublist in probs.tolist() for p in sublist]
        labels_flat = labels.tolist()

        # Converte para JSON
        probs_json = json.dumps(probs_flat)
        labels_json = json.dumps(labels_flat)

        # Retorna os pesos, número de amostras e um dicionário com as probabilidades
        return weights, len(trainloader.dataset), {"probs": probs_json, "labels": labels_json}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader, "./results", model_name=args.model_name)
        print("Acurácia do Cliente: " + str(args.dataset_id) + str(" eh: ") + str(accuracy))
        return float(loss), len(testloader.dataset), {"accuracy": round(float(accuracy), 2)}


add_log('Client_'+str(args.client_id)+' CNN Model Loading')
net = Net(model_name=args.model_name, class_num=CLASS_NUM, output=OUTPUT).to(DEVICE)
add_log('Client_'+str(args.client_id)+' CNN Model Loaded')

add_log('Client_'+str(args.client_id)+' Federated Learning Process Started')
TRAIN_START_TIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
fl.client.start_numpy_client(
    server_address="127.0.0.1:"+args.port,
    client=FlowerClient()
)
TRAIN_END_TIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
add_log('Client_'+str(args.client_id)+' Federated Learning Process Ended')



save_logs('execution_log.csv')
#resources_usage(TRAIN_START_TIME, TRAIN_END_TIME)

#Coleta de CPU da Ultima Hora
#http://200.17.78.37:19999/api/v1/data?chart=system.cpu&options=unaligned&group=sum&units=percentage&after=-3600&before=0&points=3600&format=csv

#Coleta de RAM da ultima Hora - Disponível
#http://200.17.78.37:19999/api/v1/data?chart=mem.available&options=unaligned&group=sum&units=percentage&after=-3600&before=0&points=3600&format=csv

#Coleta de RAM da ultima Hora - Usada (Comprometida)
#http://200.17.78.37:19999/api/v1/data?chart=mem.committed&options=unaligned&group=sum&units=percentage&after=-3600&before=0&points=3600&format=csv

#Coleta de GPU da ultima Hora (Comsuption)
#http://200.17.78.37:19999/api/v1/data?chart=nvidia_smi.gpu_gpu-42091774-4461-f9da-a039-2814106b5a77_gpu_utilization&options=unaligned&group=avg&units=%25&after=-86400&before=0&points=86400&format=csv
