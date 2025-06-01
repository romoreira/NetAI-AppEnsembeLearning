import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
import time
from hyperopt import hp, fmin, tpe, Trials
import functools

MODELS_TO_TEST = [
    'efficientnet_b0',
    'mobilenet_v3_small',
    'resnet34',
    'mobilenet_v2',
    'squeezenet1_0'
]

parser = argparse.ArgumentParser(description="Busca de Hiperparâmetros para múltiplos modelos.")
parser.add_argument('--epochs', type=int, default=10, help="Número de épocas para cada trial do Hyperopt.")
parser.add_argument('--max_evals', type=int, default=50, help="Número de avaliações (trials) do Hyperopt por modelo.")
args = parser.parse_args()


os.makedirs("results", exist_ok=True)
os.makedirs("results/baseline", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizado: {device}")
print(f"GPU disponível? {torch.cuda.is_available()}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carrega os datasets uma única vez
print("Carregando datasets...")
train_dataset = datasets.ImageFolder(root="./AIDER_split/train", transform=transform)
val_dataset = datasets.ImageFolder(root="./AIDER_split/val", transform=transform)
num_classes = len(train_dataset.classes)
print(f"Datasets carregados. Número de classes: {num_classes}")
print(f"Classes identificadas: {train_dataset.classes}")

def get_model(name, num_classes):
    name = name.lower()
    # Carregando modelos com pesos pré-treinados no ImageNet (weights='IMAGENET1K_V1')
    # pode acelerar a convergência. Use weights=None para treinar do zero.
    #weights = 'IMAGENET1K_V1'
    weights = None
    print(f"[MODEL] Carregando modelo: {name} com pesos: {weights}")

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
    elif name == "squeezenet1_0":
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

# Espaço de busca do Hyperopt
space = {
    'batch_size': hp.choice('batch_size', [16, 32, 64]),
    'learning_rate': hp.loguniform('learning_rate', -9.2, -4.6), #Usando logaritmo para melhor busca. Tive que pesquisar esse trem, mas parece que é uma boa prática.
    'optimizer': hp.choice('optimizer', ['adam', 'sgd'])
}

def objective(params, model_name):
    batch_size = int(params['batch_size'])
    learning_rate = params['learning_rate']
    optimizer_choice = params['optimizer']

    print(f"  > Trial: BS={batch_size}, LR={learning_rate:.5f}, OPT={optimizer_choice}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    model = get_model(model_name, num_classes).to(device)
    
    if optimizer_choice == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Treinando (Época {epoch+1})")):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, desc=f"Validando (Época {epoch+1})")):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return {'loss': avg_val_loss, 'status': 'ok'}

if __name__ == "__main__":
    for model_name in MODELS_TO_TEST:
        print("\n" + "="*80)
        print(f"INICIANDO BUSCA DE HIPERPARÂMETROS PARA O MODELO: {model_name.upper()}")
        print(f"Configuração: {args.epochs} épocas por trial, {args.max_evals} trials no total.")
        print("="*80 + "\n")

        start_time = time.time()
        
        # Cria um novo objeto Trials para cada modelo, para não misturar os resultados
        trials = Trials()
        
        # Usa functools.partial para passar o model_name para a função objective
        objective_fn = functools.partial(objective, model_name=model_name)
        
        best_params = fmin(
            fn=objective_fn,
            space=space,
            algo=tpe.suggest,
            max_evals=args.max_evals,
            trials=trials,
            show_progressbar=False
        )

        print(f"\nBusca para o modelo {model_name} finalizada.")
        print("Melhores hiperparâmetros encontrados:", best_params)

        batch_size_options = [16, 32, 64]
        optimizer_options = ['adam', 'sgd']
        
        best_batch_size = batch_size_options[best_params['batch_size']]
        best_lr = best_params['learning_rate']
        best_optimizer = optimizer_options[best_params['optimizer']]
        
        elapsed_time = time.time() - start_time

        best_params_dict = {
            "model_name": model_name,
            "best_params": {
                "batch_size": best_batch_size,
                "learning_rate": best_lr,
                "optimizer": best_optimizer,
            },
            "best_validation_loss": min(trials.losses()),
            "runtime_seconds": round(elapsed_time),
            "total_trials": args.max_evals,
            "epochs_per_trial": args.epochs
        }

        params_path = f"results/baseline/best_hyperparams_{model_name}.json"
        with open(params_path, 'w') as f:
            json.dump(best_params_dict, f, indent=4)

        print(f"Resultados da busca salvos em: {params_path}")

    print("\n" + "="*80)
    print("TODAS AS BUSCAS FORAM CONCLUÍDAS!")
    print("="*80)