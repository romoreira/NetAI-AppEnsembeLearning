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
from cycler import cycler
import matplotlib as mpl
import time

os.makedirs("results", exist_ok=True)
os.makedirs("results/baseline", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--weight_decay', type=float, default=0.0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root="./AIDER_split/train", transform=train_transform)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder(root="./AIDER_split/val", transform=val_transform)

num_classes = train_dataset.classes.__len__()

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

print("Classes identificadas no dataset de TREINO: ", train_dataset.classes)
print("Classes identificadas no dataset de VALIDAÇÃO: ", val_dataset.classes)

def get_model(name, num_classes):
    name = name.lower()
    print(f"[MODEL] Carregando modelo: {name}")
    weights = 'IMAGENET1K_V1'

    if name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet34":
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "alexnet":
        model = models.alexnet(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg16":
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg19":
        model = models.vgg19(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "squeezenet":
        model = models.squeezenet1_0(weights=weights)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Modelo '{name}' não suportado.")
    return model

model = get_model(args.model_name, num_classes).to(device)

if args.optimizer.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
else:
    raise ValueError(f"Otimizador '{args.optimizer}' não suportado.")

start_time = time.time()

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

model_path = f"results/baseline/{args.model_name}.pth"
torch.save(model.state_dict(), model_path)
print(f"Modelo salvo em {model_path}")

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


end_time = time.time()
total_time = end_time - start_time
time_message = f"\nTempo total de treino e validação: {total_time / 60:.2f} minutos ({total_time:.2f} segundos)\n" 
print(time_message)

print("Classification Report:")
report = classification_report(all_targets, all_preds, digits=5, target_names=val_dataset.classes)
print(report)
# Salva o classification report como antes
report_path = f"results/baseline/classification_report_{args.model_name}.txt"
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w') as f:
    f.write(report)
    f.write(time_message)
print(f"Classification report salvo em {report_path}")

# ===== Estética ACM-like + fonte 16 (se já tiver acima, pode remover este bloco) =====
mpl.rcParams.update({
    "font.size": 16,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral", "TeX Gyre Termes"],
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,  # texto selecionável no PDF
    "ps.fonttype": 42,
})

# ===== Confusion Matrix (maior, 2 colunas, PDF) =====
labels = getattr(val_dataset, "classes", None)
if labels is None:
    # fallback: usa rótulos numéricos encontrados
    classes = np.unique(np.concatenate([np.asarray(all_targets), np.asarray(all_preds)]))
    labels = [str(c) for c in classes]

# Se seus targets são índices alinhados a 'labels', fixe a ordem explicitamente:
label_indices = list(range(len(labels)))

out_dir = "results/baseline"
os.makedirs(out_dir, exist_ok=True)

# --- Contagens absolutas ---
cm_counts = confusion_matrix(all_targets, all_preds, labels=label_indices)

fig_c, ax_c = plt.subplots(figsize=(6.9, 6.9), constrained_layout=True)  # ~2 colunas ACM
disp_c = ConfusionMatrixDisplay(confusion_matrix=cm_counts, display_labels=labels)
disp_c.plot(cmap="Greys", xticks_rotation=45, ax=ax_c, colorbar=False, values_format="d")

# Título é geralmente na caption do paper; deixe comentado se quiser título na figura.
# ax_c.set_title(f"Confusion Matrix (Counts) - {args.model_name}")

ax_c.set_xlabel("Predicted label", labelpad=10)
ax_c.set_ylabel("True label", labelpad=10)
ax_c.tick_params(axis="x", which="both", pad=6)
ax_c.tick_params(axis="y", which="both", pad=6)

pdf_counts = os.path.join(out_dir, f"confusion_matrix_counts_{args.model_name}.pdf")
fig_c.savefig(pdf_counts, bbox_inches="tight")
plt.close(fig_c)

# --- Normalizada por verdade (linhas somam 1) ---
cm_norm = confusion_matrix(all_targets, all_preds, labels=label_indices, normalize="true")

fig_n, ax_n = plt.subplots(figsize=(6.9, 6.9), constrained_layout=True)
disp_n = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
disp_n.plot(cmap="Greys", xticks_rotation=45, ax=ax_n, colorbar=True, values_format=".2f")

# ax_n.set_title(f"Confusion Matrix (Normalized) - {args.model_name}")
ax_n.set_xlabel("Predicted label", labelpad=10)
ax_n.set_ylabel("True label", labelpad=10)
ax_n.tick_params(axis="x", which="both", pad=6)
ax_n.tick_params(axis="y", which="both", pad=6)

pdf_norm = os.path.join(out_dir, f"confusion_matrix_norm_{args.model_name}.pdf")
fig_n.savefig(pdf_norm, bbox_inches="tight")
plt.close(fig_n)

print(f"Matrizes de confusão salvas em:\n - {pdf_counts}\n - {pdf_norm}")

# Estética ACM-like + fonte 16
mpl.rcParams.update({
    "font.size": 16,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral", "TeX Gyre Termes"],
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.prop_cycle": cycler("color", ["0.0", "0.35"]),  # tons de cinza
    "pdf.fonttype": 42,  # texto selecionável no PDF
    "ps.fonttype": 42,
})

# Tamanho maior (≈ 2 colunas ACM)
FIG_W, FIG_H = 6.9, 4.1  # ajuste a altura se quiser mais/menos espaço vertical
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)

epochs = range(1, args.epochs + 1)
ax.plot(epochs, train_losses, marker='o', markersize=5, linewidth=2.0, label="Loss")
ax.plot(epochs, train_accuracies, marker='s', markersize=5, linewidth=2.0,
        fillstyle='none', label="Accuracy")

# Título geralmente vai na caption do paper; deixe descomentado se quiser:
# ax.set_title(f"Training Metrics - {args.model_name}")

ax.set_xlabel("Epoch", labelpad=8)
ax.set_ylabel("Value", labelpad=8)
ax.set_xticks(list(epochs))

# Mais espaço: legenda acima, fora do eixo
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, handlelength=1.8)

# Espaçamentos adicionais
ax.tick_params(axis='both', which='major', pad=6)
ax.margins(x=0.02, y=0.08)
ax.grid(False)

out_dir = "results/baseline"
os.makedirs(out_dir, exist_ok=True)
pdf_path = os.path.join(out_dir, f"metrics_{args.model_name}.pdf")
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
print(f"Gráfico em PDF salvo em {pdf_path}")