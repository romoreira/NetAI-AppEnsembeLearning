import os
import random
import shutil

# Diretório original das imagens
diretorio_origem = './data/biglycan'

# Diretórios de destino
diretorio_train = './dataset/production/train'
diretorio_test = './dataset/production/test'

# Proporção de imagens para o conjunto de treinamento (entre 0 e 1)
proporcao_train = 0.8  # 80% para treinamento, 20% para teste

# Criar diretórios principais e subdiretórios
os.makedirs(os.path.join(diretorio_train, 'CANCER'), exist_ok=True)
os.makedirs(os.path.join(diretorio_train, 'HEALTHY'), exist_ok=True)
os.makedirs(os.path.join(diretorio_test, 'CANCER'), exist_ok=True)
os.makedirs(os.path.join(diretorio_test, 'HEALTHY'), exist_ok=True)

# Listar imagens de CANCER e HEALTHY
imagens_cancer = [img for img in os.listdir(os.path.join(diretorio_origem, 'CANCER')) if img.endswith('.png')]
imagens_healthy = [img for img in os.listdir(os.path.join(diretorio_origem, 'HEALTHY')) if img.endswith('.png')]

# Embaralhar as listas de imagens
random.shuffle(imagens_cancer)
random.shuffle(imagens_healthy)

# Calcular quantidade de imagens para train e test
quantidade_train_cancer = int(len(imagens_cancer) * proporcao_train)
quantidade_train_healthy = int(len(imagens_healthy) * proporcao_train)

# Copiar imagens para os diretórios train e test dentro de production
for img in imagens_cancer[:quantidade_train_cancer]:
    origem_img = os.path.join(diretorio_origem, 'CANCER', img)
    destino_train_img = os.path.join(diretorio_train, 'CANCER', img)
    if os.path.isfile(origem_img):
        shutil.copy(origem_img, destino_train_img)

for img in imagens_cancer[quantidade_train_cancer:]:
    origem_img = os.path.join(diretorio_origem, 'CANCER', img)
    destino_test_img = os.path.join(diretorio_test, 'CANCER', img)
    if os.path.isfile(origem_img):
        shutil.copy(origem_img, destino_test_img)

for img in imagens_healthy[:quantidade_train_healthy]:
    origem_img = os.path.join(diretorio_origem, 'HEALTHY', img)
    destino_train_img = os.path.join(diretorio_train, 'HEALTHY', img)
    if os.path.isfile(origem_img):
        shutil.copy(origem_img, destino_train_img)

for img in imagens_healthy[quantidade_train_healthy:]:
    origem_img = os.path.join(diretorio_origem, 'HEALTHY', img)
    destino_test_img = os.path.join(diretorio_test, 'HEALTHY', img)
    if os.path.isfile(origem_img):
        shutil.copy(origem_img, destino_test_img)
