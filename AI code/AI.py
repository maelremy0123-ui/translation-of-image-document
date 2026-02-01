import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import random
from PIL import ImageDraw
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Dénormalisation : x = x * std + mean
    return tensor.clamp(0, 1)  # Assure que les valeurs sont dans [0, 1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
input_size = 100*100
hidden_size = 100
num_classes = 26
num_epochs = 100
batch_size = 64
learning_rate = 0.001
train_majuscule_root = r"C:\pytorch\projet\trad_jpeg_png\letter_code\dataset_2\train\majuscule"
test_majuscule_root = r"C:\pytorch\projet\trad_jpeg_png\letter_code\dataset_2\test\majuscule"
train_minuscule_root = r"C:\pytorch\projet\trad_jpeg_png\letter_code\dataset_2\train\minuscule"
test_minuscule_root = r"C:\pytorch\projet\trad_jpeg_png\letter_code\dataset_2\test\minuscule"

def reduction(image):
    if random.random() < 0.3:
        largeur, hauteur = image.size
        image_reduite = ImageDraw.Draw(image)
        for x in range (largeur//2):
            for y in range (hauteur //2):
                t = image_reduite.getpixel((2*x, 2*y))
                t +=image_reduite.getpixel((2*x, 2*y+1))
                t +=image_reduite.getpixel((2*x+1, 2*y))
                t +=image_reduite.getpixel((2*x+1, 2*y+1))
                t /=4
                image_reduite.point((2*x, 2*y), fill=(t,t,t))
                image_reduite.point((2*x+1, 2*y), fill=(t,t,t))
                image_reduite.point((2*x, 2*y+1), fill=(t,t,t))
                image_reduite.point((2*x+1, 2*y+1), fill=(t,t,t))
        return image

def bruit(image, taux):
    if random.random() < 0.5:
        largeur, hauteur = image.size
        image_bruite = ImageDraw.Draw(image)
        for x in range(largeur):
            for y in range(hauteur):
                if random.random() < taux:
                    gris = random.randint(0, 255)
                    image_bruite.point((x, y), fill=(gris, gris, gris))
    return image

def add_noise(img, taux):
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img)
    img = bruit(img, taux)
    return transforms.ToTensor()(img)

transform = transforms.Compose([
    transforms.Resize(50),
    transforms.ToTensor(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.1, hue = 0.1),
    transforms.Lambda(lambda img: add_noise(img, taux = 0.05)), 
    transforms.RandomPerspective(distortion_scale = 0.2, p = 0.3),
    transforms.RandomInvert(p = 0.5),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])
])

train_majuscule_dataset = torchvision.datasets.ImageFolder(root=train_majuscule_root, transform=transform)
train_minuscule_dataset = torchvision.datasets.ImageFolder(root=train_minuscule_root, transform=transform)
test_majuscule_dataset = torchvision.datasets.ImageFolder(root=test_majuscule_root, transform=transform)
test_minuscule_dataset = torchvision.datasets.ImageFolder(root=test_minuscule_root, transform=transform)
train_minuscule_dataset.samples = [(img_path, label + 26) for img_path, label in train_minuscule_dataset.samples]
train_minuscule_dataset.targets = [label + 26 for label in train_minuscule_dataset.targets]
test_minuscule_dataset.samples = [(img_path, label + 26) for img_path, label in test_minuscule_dataset.samples]
test_minuscule_dataset.targets = [label + 26 for label in test_minuscule_dataset.targets]

train_dataset = torch.utils.data.ConcatDataset([train_majuscule_dataset, train_minuscule_dataset])
test_dataset = torch.utils.data.ConcatDataset([test_majuscule_dataset, test_minuscule_dataset])

targets = []
for ds in train_dataset.datasets:  # train_dataset est un ConcatDataset
    targets.extend(ds.targets)

# Compter combien d’images par classe
class_counts = Counter(targets)
total_count = sum(class_counts.values())

# Poids inverses à la fréquence
class_weights = [total_count / class_counts[i] for i in range(len(class_counts))]

# Convertir en tenseur PyTorch (sur le même device que le modèle)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size,
                        shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size,
                        shuffle = False)

classes = ("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
           "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z")
examples = iter(train_loader)
samples, labels = next(examples)


# Sélectionner les 4 premières images du lot
n_images = 20
fig, axes = plt.subplots(1, n_images, figsize=(15, 4))

for i, ax in enumerate(axes.flat):
    if i >= len(samples):
        break  # Au cas où le lot contient moins de 4 images

    # Dénormaliser l'image
    img = denormalize(samples[i])

    # Convertir le tensor en tableau NumPy et permuter les dimensions pour matplotlib
    img_np = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)

    # Afficher l'image
    ax.imshow(img_np)
    ax.set_title(f"Label: {classes[labels[i].item()]}")
    ax.axis('off')

plt.tight_layout()
plt.show()

class LetterClassifier(nn.Module):
    def __init__(self):
        super(LetterClassifier, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128*6*6, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 52)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # aplatissement
        x = self.fc_layer(x)
        return x
    
model = LetterClassifier().to(device)
criterion = nn.CrossEntropyLoss(weight = class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)

for epoch in range(num_epochs):
    #model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    scheduler.step()
print('Finished Training')
"""for param in model.parameters():
    #print(param)
    pass"""

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(52)]
    n_class_samples = [0 for i in range(52)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    
        for i in range(labels.size(0)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(52):
        if n_class_samples[i] > 0: 
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')
        else:
            print(f'Accuracy of {classes[i]}: No samples')
            
PATH = r"C:\pytorch\projet\trad_jpeg_png\AI\AI_Final.pth"
torch.save(model.state_dict(), PATH)

all_preds = []
all_labels = []

model.eval()
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# Matrice de confusion complète
cm = confusion_matrix(all_labels, all_preds)

# --- Étape 1 : calcul de l'accuracy par classe ---
n_class_correct = cm.diagonal()
n_class_samples = cm.sum(axis=1)
class_acc = n_class_correct / np.maximum(n_class_samples, 1)  # éviter division par 0

# --- Étape 2 : trouver les 10 pires classes ---
worst_classes_idx = np.argsort(class_acc)[:10]
worst_classes_names = [classes[i] for i in worst_classes_idx]

# --- Étape 3 : extraire la sous-matrice ---
cm_worst = cm[np.ix_(worst_classes_idx, worst_classes_idx)]

# --- Étape 4 : affichage ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm_worst, display_labels=worst_classes_names)
disp.plot(xticks_rotation=90, cmap="viridis")
plt.title("Confusion matrix - 10 worst classes")
plt.show()
