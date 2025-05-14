import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on", device)
num_classes = 10  # Altere conforme sua base
epochs = 5
batch_size = 32

# 1. DATASETS E TRANSFORMAÇÕES
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 2. DEFINIÇÃO DOS MODELOS
def get_resnet():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def get_squeezenet():
    model = models.squeezenet1_0(weights=None)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    return model.to(device)

model1 = get_resnet()
model2 = get_squeezenet()

# 3. FUNÇÃO DE TREINAMENTO
def train(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        loop = tqdm(loader, desc=f"Treinando {model.__class__.__name__} - Época {epoch+1}/{epochs}")
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

# 4. TREINAR OS MODELOS BASE
train(model1, train_loader)
train(model2, train_loader)

# 5. EXTRAIR PROBABILIDADES PARA O STACKING
def extract_probs(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        loop = tqdm(loader, desc=f"Extraindo probs de {model.__class__.__name__}")
        for inputs, targets in loop:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(targets)
    return torch.cat(all_probs), torch.cat(all_labels)

probs_val1, labels_val = extract_probs(model1, val_loader)
probs_val2, _ = extract_probs(model2, val_loader)

# 6. STACKING COM REGRESSÃO LOGÍSTICA
# Combinar as probabilidades
X_stack = torch.cat([probs_val1, probs_val2], dim=1).numpy()
y_stack = labels_val.numpy()

# Meta-modelo
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(X_stack, y_stack)

# 7. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
probs_test1, labels_test = extract_probs(model1, test_loader)
probs_test2, _ = extract_probs(model2, test_loader)
X_test_stack = torch.cat([probs_test1, probs_test2], dim=1).numpy()
y_test = labels_test.numpy()

# Prever com o meta-modelo
y_pred = meta_model.predict(X_test_stack)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy do ensemble com stacking: {acc:.4f}")
