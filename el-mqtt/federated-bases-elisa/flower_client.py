# flower_client.py (versão atualizada)
import argparse
import warnings
from collections import OrderedDict
import os
import random
from tqdm import tqdm
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np
import matplotlib
import json
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay



warnings.filterwarnings("ignore", category=UserWarning)

# ... (Funções set_seed, load_data, get_model permanecem as mesmas) ...
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_data(batch_size: int, clientID: int):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainFolder =  '';
    valFolder = '';
    if(clientID == 0):
        trainFolder = '../Dataset/FlameVisionData/FlameVision/Classification_Reduced/train';
        valFolder = '../Dataset/FlameVisionData/FlameVision/Classification/test';
    else:
        trainFolder = '../Dataset/ForestFireData/ForestFire/Classification/train';
        valFolder = '../Dataset/ForestFireData/ForestFire/Classification/test';

    train_dataset = datasets.ImageFolder(root=trainFolder, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=valFolder, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, len(train_dataset.classes), train_dataset.classes

def get_model(name: str, num_classes: int):
    name = name.lower()
    weights = 'IMAGENET1K_V1'
    model_map = {
        "resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
        "alexnet": models.alexnet, "vgg16": models.vgg16, "vgg19": models.vgg19,
        "mobilenet_v2": models.mobilenet_v2, "mobilenet_v3_small": models.mobilenet_v3_small,
        "mobilenet_v3_large": models.mobilenet_v3_large, "squeezenet": models.squeezenet1_0,
        "densenet121": models.densenet121, "efficientnet_b0": models.efficientnet_b0
    }
    if name not in model_map: raise ValueError(f"Modelo '{name}' não suportado.")
    model = model_map[name](weights=weights)
    if "resnet" in name or "densenet" in name:
        in_features = model.fc.in_features if "resnet" in name else model.classifier.in_features
        model.fc = nn.Linear(in_features, num_classes) if "resnet" in name else nn.Linear(in_features, num_classes)
        if "densenet" in name: model.classifier = model.fc
    elif "alexnet" in name or "vgg" in name: model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif "mobilenet_v2" in name or "efficientnet_b0" in name: model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif "mobilenet_v3" in name: model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif "squeezenet" in name:
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    return model

def train(net, trainloader, epochs, client_id, optimizer_name, lr, weight_decay, device):
    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == "adam": optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd": optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else: raise ValueError(f"Otimizador '{optimizer_name}' não suportado.")
    
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        
        # *** MUDANÇA PRINCIPAL AQUI ***
        # O loop agora é envolvido por tqdm para mostrar a barra de progresso
        progress_bar = tqdm(trainloader, 
                            desc=f"  [Cliente {client_id} Epoch {epoch+1}/{epochs}]", 
                            leave=False, 
                            ncols=100) # ncols ajusta a largura da barra
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(); outputs = net(inputs); loss = criterion(outputs, targets)
            loss.backward(); optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
            correct += (torch.max(outputs.data, 1)[1] == targets).sum().item()

        epoch_loss /= total; epoch_acc = correct / total
        # O print do resultado da época foi movido para fora da classe para não interferir
        # com a barra de progresso, mas neste caso podemos mantê-lo.
        print(f"  Epoch {epoch+1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")

def test(net, testloader, device, client_id, model_name, class_names, combo_name):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    all_preds, all_targets = [], []
    net.eval()
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss += criterion(outputs, targets).item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    accuracy = correct / total; avg_loss = loss / total
    
    # *** MUDANÇA AQUI: O caminho de saída agora usa o combo_name ***
    output_dir = f"results/{combo_name}/client_{client_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    report = classification_report(all_targets, all_preds, digits=5, target_names=class_names, zero_division=0)
    report_path = os.path.join(output_dir, f"report_{model_name}.txt")
    with open(report_path, 'w') as f: f.write(report)
    
    save_confusion_matrix(all_targets, all_preds, class_names, output_dir, model_name)
    
    return avg_loss, accuracy, all_preds, all_targets

def save_confusion_matrix(y_true, y_pred, labels, out_dir, model_name):
    # ... (Esta função permanece a mesma) ...
    mpl.rcParams.update({"font.size": 16, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9})
    label_indices = list(range(len(labels))); os.makedirs(out_dir, exist_ok=True)
    cm_counts = confusion_matrix(y_true, y_pred, labels=label_indices)
    fig_c, ax_c = plt.subplots(figsize=(6.9, 6.9), constrained_layout=True)
    disp_c = ConfusionMatrixDisplay(confusion_matrix=cm_counts, display_labels=labels)
    disp_c.plot(cmap="Greys", xticks_rotation=45, ax=ax_c, colorbar=False, values_format="d")
    ax_c.set_xlabel("Predicted label", labelpad=10); ax_c.set_ylabel("True label", labelpad=10)
    fig_c.savefig(os.path.join(out_dir, f"cm_counts_{model_name}.pdf"), bbox_inches="tight")
    plt.close(fig_c)
    cm_norm = confusion_matrix(y_true, y_pred, labels=label_indices, normalize="true")
    fig_n, ax_n = plt.subplots(figsize=(6.9, 6.9), constrained_layout=True)
    disp_n = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp_n.plot(cmap="Greys", xticks_rotation=45, ax=ax_n, colorbar=True, values_format=".2f")
    ax_n.set_xlabel("Predicted label", labelpad=10); ax_n.set_ylabel("True label", labelpad=10)
    fig_n.savefig(os.path.join(out_dir, f"cm_norm_{model_name}.pdf"), bbox_inches="tight")
    plt.close(fig_n)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, device, args, class_names):
        self.net = net; self.trainloader = trainloader; self.valloader = valloader
        self.device = device; self.args = args; self.class_names = class_names

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        local_epochs = config.get("local_epochs", self.args.local_epochs)
        print(f"  [Cliente {self.args.client_id}] Treinando por {local_epochs} épocas...")
        
        # *** ATUALIZE A CHAMADA AQUI para passar o client_id ***
        train(
            self.net, self.trainloader, epochs=local_epochs, 
            client_id=self.args.client_id,  # <--- PARÂMETRO ADICIONADO
            optimizer_name=self.args.optimizer, 
            lr=self.args.lr, weight_decay=self.args.weight_decay, device=self.device
        )
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        print(f"  [Cliente {self.args.client_id}] Avaliando o modelo global...")
        loss, accuracy, preds, targets = test(
            self.net, self.valloader, self.device, self.args.client_id, 
            self.args.model_name, self.class_names, self.args.combo_name
        )
        print(f"  [Cliente {self.args.client_id}] Val Loss: {loss:.4f}, Val Acc: {accuracy:.4f}")
        
        # *** MUDANÇA CRÍTICA: Enviar predições e rótulos para o servidor ***
        # Dentro do método evaluate em flower_client.py
        return float(loss), len(self.valloader.dataset), {
            "accuracy": float(accuracy),
            "predictions": json.dumps(np.array(preds).tolist()),   # <-- CORRIGIDO
            "targets": json.dumps(np.array(targets).tolist())      # <-- CORRIGIDO
        }

def main():
    parser = argparse.ArgumentParser(description="Cliente Flower")
    # Adiciona o argumento combo_name
    parser.add_argument("--combo_name", type=str, required=True, help="Nome da combinação para salvar resultados")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--local_epochs', type=int, default=20) # Treino local mais curto por rodada
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    # set_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, valloader, num_classes, class_names = load_data(args.batch_size, args.client_id)
    net = get_model(args.model_name, num_classes).to(DEVICE)
    client = FlowerClient(net, trainloader, valloader, DEVICE, args, class_names)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()