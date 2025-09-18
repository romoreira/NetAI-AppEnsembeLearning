import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
import itertools
import pandas as pd
import os

# ========================
# CONFIGURAÇÕES
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executando no dispositivo: {device}")

# Diretório para salvar os relatórios
REPORTS_DIR = "el-mqtt/results/stacking_reports"
os.makedirs(REPORTS_DIR, exist_ok=True) # Cria o diretório se não existir

# Mapeamento de nomes de modelos para os nomes corretos do torchvision
MODEL_NAME_MAP = {
    "squeezenet": "squeezenet1_0"
}

# Lista de modelos para combinar
models_to_run = [
    {"model_name": "efficientnet_b0", "path": "el-mqtt/results/baseline/efficientnet_b0.pth"},
    {"model_name": "mobilenet_v2", "path": "el-mqtt/results/baseline/mobilenet_v2.pth"},
    {"model_name": "resnet34", "path": "el-mqtt/results/baseline/resnet34.pth"},
    {"model_name": "mobilenet_v3_small", "path": "el-mqtt/results/baseline/mobilenet_v3_small.pth"},
    {"model_name": "squeezenet1_0", "path": "el-mqtt/results/baseline/squeezenet.pth"},
]

# ========================
# DATA LOADERS
# ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root="./el-mqtt/AIDER_split/train", transform=transform)
val_dataset = datasets.ImageFolder(root="./el-mqtt/AIDER_split/val", transform=transform)

num_classes = len(train_dataset.classes)
print(f"Número de classes detectado: {num_classes}")

meta_train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
meta_test_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# ========================
# FUNÇÃO GENÉRICA PARA CARREGAR MODELOS
# ========================
def load_model(model_name, path, num_classes):
    """Carrega um modelo, ajusta a camada final e carrega os pesos."""
    model_name_actual = MODEL_NAME_MAP.get(model_name, model_name)
    
    if model_name_actual == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name_actual == "squeezenet1_0":
        model = models.squeezenet1_0(weights=None)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        model.num_classes = num_classes
    elif model_name_actual == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name_actual == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name_actual == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Modelo '{model_name}' não suportado.")
        
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

# ========================
# EXTRAÇÃO DE PROBABILIDADES
# ========================
def extract_probs(model, loader, model_name):
    """Extrai as probabilidades (softmax) de um modelo para um dado dataset."""
    all_probs = []
    all_labels = []
    with torch.no_grad():
        loop = tqdm(loader, desc=f"Extraindo de {model_name}", leave=False)
        for inputs, targets in loop:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(targets.cpu())
            
    return torch.cat(all_probs), torch.cat(all_labels)


# ========================
# LÓGICA PRINCIPAL DO STACKING
# ========================
results = []
model_combinations = list(itertools.combinations(models_to_run, 3))
# TODO RODAR EM GRUPOS DE 3

print(f"\nIniciando o processo de Stacking para {len(model_combinations)} combinações...")

for combo in model_combinations:
    model_info_1, model_info_2, model_info_3 = combo
    name1, path1 = model_info_1["model_name"], model_info_1["path"]
    name2, path2 = model_info_2["model_name"], model_info_2["path"]
    name3, path3 = model_info_3["model_name"], model_info_3["path"]
    
    combo_name = f"{name1} + {name2} + {name3}"
    print(f"\n--- Processando Combinação: {combo_name} ---")

    model1 = load_model(name1, path1, num_classes)
    model2 = load_model(name2, path2, num_classes)
    model3 = load_model(name3, path3, num_classes)

    probs_train1, labels_train = extract_probs(model1, meta_train_loader, name1)
    probs_train2, _ = extract_probs(model2, meta_train_loader, name2)
    probs_train3, _ = extract_probs(model3, meta_train_loader, name3)

    X_stack_train = torch.cat([probs_train1, probs_train2, probs_train3], dim=1).numpy()
    y_stack_train = labels_train.numpy()

    meta_model = LogisticRegression(max_iter=1000, n_jobs=-1)
    meta_model.fit(X_stack_train, y_stack_train)
    
    probs_test1, labels_test = extract_probs(model1, meta_test_loader, name1)
    probs_test2, _ = extract_probs(model2, meta_test_loader, name2)
    probs_test3, _ = extract_probs(model3, meta_test_loader, name3)

    X_stack_test = torch.cat([probs_test1, probs_test2, probs_test3], dim=1).numpy()
    y_test_true = labels_test.numpy()
    
    y_pred = meta_model.predict(X_stack_test)
    accuracy = accuracy_score(y_test_true, y_pred)
    
    print(f"✅ Acurácia para {combo_name}: {accuracy:.4f}")

    # Salvar o classification report
    report_str = classification_report(y_test_true, y_pred,digits=5, target_names=train_dataset.classes)
    file_content = f"Combination: {combo_name}\n\n{report_str}"
    
    filename = f"report_{name1}_{name2}_{name3}.txt"
    filepath = os.path.join(REPORTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        f.write(file_content)
    
    print(f"   Relatório de classificação salvo em: {filepath}")

    results.append({
        "combination": combo_name,
        "accuracy": accuracy
    })
    
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # ===== Estética ACM-like + fonte 16 =====
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

    # Labels das classes
    labels = getattr(train_dataset, "classes", None)
    if labels is None:
        classes = np.unique(np.concatenate([y_test_true, y_pred]))
        labels = [str(c) for c in classes]
    label_indices = list(range(len(labels)))

    out_dir = os.path.join(REPORTS_DIR, "conf_matrices")
    os.makedirs(out_dir, exist_ok=True)

    # --- Confusion Matrix (counts) ---
    cm_counts = confusion_matrix(y_test_true, y_pred, labels=label_indices)
    fig_c, ax_c = plt.subplots(figsize=(6.9, 6.9), constrained_layout=True)
    disp_c = ConfusionMatrixDisplay(confusion_matrix=cm_counts, display_labels=labels)
    disp_c.plot(cmap="Greys", xticks_rotation=45, ax=ax_c, colorbar=False, values_format="d")
    ax_c.set_xlabel("Predicted label", labelpad=10)
    ax_c.set_ylabel("True label", labelpad=10)
    ax_c.tick_params(axis="x", which="both", pad=6)
    ax_c.tick_params(axis="y", which="both", pad=6)

    pdf_counts = os.path.join(out_dir, f"cm_counts_{name1}_{name2}_{name3}.pdf")
    fig_c.savefig(pdf_counts, bbox_inches="tight")
    plt.close(fig_c)

    # --- Confusion Matrix (normalized) ---
    cm_norm = confusion_matrix(y_test_true, y_pred, labels=label_indices, normalize="true")
    fig_n, ax_n = plt.subplots(figsize=(6.9, 6.9), constrained_layout=True)
    disp_n = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp_n.plot(cmap="Greys", xticks_rotation=45, ax=ax_n, colorbar=True, values_format=".2f")
    ax_n.set_xlabel("Predicted label", labelpad=10)
    ax_n.set_ylabel("True label", labelpad=10)
    ax_n.tick_params(axis="x", which="both", pad=6)
    ax_n.tick_params(axis="y", which="both", pad=6)

    pdf_norm = os.path.join(out_dir, f"cm_norm_{name1}_{name2}_{name3}.pdf")
    fig_n.savefig(pdf_norm, bbox_inches="tight")
    plt.close(fig_n)

    print(f"Matrizes de confusão salvas em:\n - {pdf_counts}\n - {pdf_norm}")

    
    del model1, model2, model3
    torch.cuda.empty_cache()

# ========================
# EXIBIÇÃO DOS RESULTADOS FINAIS
# ========================
print("\n\n==============================================")
print("  Sumário de Acurácia do Stacking Ensemble")
print("==============================================")

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

print(df_results.to_string())