"""Anime Character Image Classification with a CNN.

Portfolio learning project by Rawan Wehbe.
Adapted from an IBM Skills Network educational lab; see README.md for attribution.
Run: python anime_cnn.py
Requires internet access to download the small dataset.
"""

import io
import zipfile
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler


def load_images_from_zip(zip_file):
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        images = {"anastasia": [], "takao": []}
        for file_name in zip_ref.namelist():
            if file_name.startswith("anastasia") and file_name.endswith(".jpg"):
                with zip_ref.open(file_name) as file:
                    img = Image.open(file).convert("RGB")
                    images["anastasia"].append(np.array(img))
            elif file_name.startswith("takao") and file_name.endswith(".jpg"):
                with zip_ref.open(file_name) as file:
                    img = Image.open(file).convert("RGB")
                    images["takao"].append(np.array(img))
    return images


zip_file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/xZQHOyN8ONT92kH-ASb4Pw/data.zip"
response = requests.get(zip_file_url, timeout=60)
response.raise_for_status()
images = load_images_from_zip(io.BytesIO(response.content))

print("Number of images of Anastasia:", len(images["anastasia"]))
print("Number of images of Takao:", len(images["takao"]))


def plot_images(image_list, title):
    fig, axes = plt.subplots(5, 10, figsize=(10, 5))
    fig.suptitle(title, fontsize=16)
    for img, ax in zip(image_list, axes.flatten()):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


plot_images(images["anastasia"], "Anastasia Images")
plot_images(images["takao"], "Takao Images")


class AnimeDataset(Dataset):
    def __init__(self, images, transform=None, classes=None):
        self.images = []
        self.labels = []
        self.transform = transform
        for label, class_name in enumerate(classes):
            for img in images[class_name]:
                self.images.append(img)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = AnimeDataset(images, transform=transform, classes=["anastasia", "takao"])

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(
    indices, test_size=0.2, random_state=seed
)
train_loader = DataLoader(dataset, batch_size=8, sampler=SubsetRandomSampler(train_indices))
val_loader = DataLoader(dataset, batch_size=20, sampler=SubsetRandomSampler(val_indices))
print("Train size:", len(train_indices))
print("Validation size:", len(val_indices))


class AnimeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


model = AnimeCNN()
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
train_losses, val_losses = [], []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / max(1, len(train_loader))
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            val_loss += criterion(model(inputs), labels).item()
    val_loss /= max(1, len(val_loader))
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}: train loss={train_loss:.4f}, validation loss={val_loss:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

model.eval()
correct, total = 0, 0
all_images, all_labels, all_predictions = [], [], []
with torch.no_grad():
    for batch_images, labels in val_loader:
        outputs = model(batch_images)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        all_images.extend(batch_images.cpu())
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

print(f"Validation accuracy: {100 * correct / max(1, total):.2f}% ({correct}/{total})")

class_names = ["Anastasia", "Takao"]
num_to_show = min(10, len(all_images))
fig, axes = plt.subplots(2, num_to_show, figsize=(2 * num_to_show, 4))
if num_to_show == 1:
    axes = np.array(axes).reshape(2, 1)
for i in range(num_to_show):
    image = all_images[i] * 0.5 + 0.5
    axes[0, i].imshow(np.transpose(image.numpy(), (1, 2, 0)).clip(0, 1))
    axes[0, i].set_title(f"Actual: {class_names[all_labels[i]]}")
    axes[0, i].axis("off")
    axes[1, i].text(0.5, 0.5, f"Predicted:\n{class_names[all_predictions[i]]}", ha="center", va="center")
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()
