import os
import shutil
import random
dataset_path = "./AIDER"
train_path = "./AIDER_split/train"
val_path = "./AIDER_split/val"

classes = ["collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident"]

split_ratio = 0.8

for split in [train_path, val_path]:
    os.makedirs(split, exist_ok=True)
    for class_name in classes:
        os.makedirs(os.path.join(split, class_name), exist_ok=True)

for class_name in classes:
    class_dir = os.path.join(dataset_path, class_name)
    images = os.listdir(class_dir)

    # randomizando o dataset antes de separar
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    
    # separando cada imagem em train e validation
    for i, image in enumerate(images):
        src = os.path.join(class_dir, image)
        if i < split_index:
            dst = os.path.join(train_path, class_name, image)
        else:
            dst = os.path.join(val_path, class_name, image)
        shutil.copy(src, dst)

print("O Dataset foi corretamente separado.")
