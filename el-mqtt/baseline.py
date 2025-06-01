# client_el-mqtt.py (baseline - sem MQTT)
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Salvar gráficos
os.makedirs("results", exist_ok=True)
os.makedirs("results/baseline", exist_ok=True)

# Argumentos via CLI
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets
# train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
# val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_dataset = datasets.ImageFolder(root="./AIDER_split/train", transform=transform)
val_dataset = datasets.ImageFolder(root="./AIDER_split/val", transform=transform)

num_classes = train_dataset.classes.__len__()

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

print("Classes identificadas no dataset de TREINO: ", train_dataset.classes)
print("Classes identificadas no dataset de VALIDAÇÃO: ", val_dataset.classes)

def get_model(name, num_classes):
    name = name.lower()
    print(f"[MODEL] Carregando modelo: {name}")
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg19":
        model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "squeezenet":
        model = models.squeezenet1_0(weights=None)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Modelo '{name}' não suportado.")
    return model

model = get_model(args.model_name, num_classes).to(device)

# Otimizador
if args.optimizer.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
else:
    raise ValueError(f"Otimizador '{args.optimizer}' não suportado.")

# Treinamento
criterion = nn.CrossEntropyLoss()
model.train()
train_losses = []
train_accuracies = []

for epoch in range(args.epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / len(train_loader)
    acc = correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(acc)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

# Avaliação
model.eval()
correct_val = 0
total_val = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in tqdm(val_loader, desc="Validating"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        correct_val += (predicted == targets).sum().item()
        total_val += targets.size(0)

val_acc = correct_val / total_val
print(f"Validation Accuracy: {val_acc:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(all_targets, all_preds))
# Salvar classification report em arquivo txt
report_path = f"results/baseline/classification_report_{args.model_name}.txt"
with open(report_path, 'w') as f:
    f.write(classification_report(all_targets, all_preds))
print(f"Classification report salvo em {report_path}")


# Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay(cm, display_labels=val_dataset.classes).plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix - {args.model_name}")
plt.savefig(f"results/baseline/confusion_matrix_{args.model_name}.png")
plt.close()



plt.figure()
plt.plot(train_losses, marker='o', label="Loss")
plt.plot(train_accuracies, marker='x', label="Accuracy")
plt.title(f"Training Metrics - {args.model_name}")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
fig_path = f"results/baseline/metrics_{args.model_name}.png"
plt.savefig(fig_path)
print(f"Gráfico salvo em {fig_path}")
