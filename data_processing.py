import cv2
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from os import listdir

# retele neuronale convolutionale, clasificare imagini un basetime cu knn sau cu un cnn simplu sau mai greu iau feature si pe feature fac cnn sau svm

folder_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification"
img_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification/train"

test_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification/test"
test_file = os.path.join(folder_dir, "test.csv")  # if you have a CSV for test labels

csv_file = os.path.join(folder_dir, "train.csv")

validation_file = os.path.join(folder_dir, "validation.csv")
val_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification/validation"



class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = NeuralNet()

df_train = pd.read_csv(csv_file)

data = []
for idx, row in df_train.iterrows():
    img_name = row['image_id'] + '.png'
    label = row['label']
    img_path = os.path.join(img_dir, img_name)
    image = cv2.imread(img_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (32, 32))
    b, g, r = cv2.split(image_resized)
    b_norm = cv2.normalize(b.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    g_norm = cv2.normalize(g.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    r_norm = cv2.normalize(r.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    normalized_image = cv2.merge((b_norm, g_norm, r_norm))

    image_tensor = torch.from_numpy(normalized_image).permute(2, 0, 1).float()

    data.append((image_tensor, label))

images = torch.stack([item[0] for item in data])
labels = torch.tensor([item[1] for item in data])

dataset = TensorDataset(images, labels)
train_loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=8)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

NUM_EPOCHS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model = model.to(device) 
loss_function = nn.CrossEntropyLoss() 

train_losses = []
train_accuracies = []

model.train(True)
for epoch in range(NUM_EPOCHS): 
    print(f"=== Epoch {epoch+1} ===")
    ####################
    running_loss = 0.0
    correct = 0
    total = 0
    ###################
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

#####################################################
        running_loss += loss.item() * inputs.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
#####################################################

        if i % 100 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")

##################################################################################
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
##################################################################################
print('Finished Training')

epochs = range(1, NUM_EPOCHS + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, train_accuracies, 'r-', label='Training Accuracy')
plt.xlabel('Epoch')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.show()


df_test = pd.read_csv(test_file)

test_data = []
for idx, row in df_test.iterrows():
    img_name = row['image_id'] + '.png'
    img_path = os.path.join(test_dir, img_name)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (32, 32))
    b, g, r = cv2.split(image_resized)
    b_norm = cv2.normalize(b.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    g_norm = cv2.normalize(g.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    r_norm = cv2.normalize(r.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    normalized_image = cv2.merge((b_norm, g_norm, r_norm))
    image_tensor = torch.from_numpy(normalized_image).permute(2, 0, 1).float()
    test_data.append(image_tensor)

test_images = torch.stack(test_data)
test_dataset = TensorDataset(test_images)  # Only images, no labels
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=8)

model.to(device)
model.eval()

predicted_labels = []
with torch.no_grad():
    for image_batch in test_loader:
        images = image_batch[0].to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        predicted_labels.extend(preds.tolist())

print(predicted_labels)

df_val = pd.read_csv(validation_file)

val_data = []
for idx, row in df_val.iterrows():
    img_name = row['image_id'] + '.png'
    label = row['label']
    img_path = os.path.join(val_dir, img_name)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (32, 32))
    b, g, r = cv2.split(image_resized)
    b_norm = cv2.normalize(b.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    g_norm = cv2.normalize(g.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    r_norm = cv2.normalize(r.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    normalized_image = cv2.merge((b_norm, g_norm, r_norm))
    image_tensor = torch.from_numpy(normalized_image).permute(2, 0, 1).float()
    val_data.append((image_tensor, label))

val_images = torch.stack([item[0] for item in val_data])
val_labels = torch.tensor([item[1] for item in val_data])

val_dataset = TensorDataset(val_images, val_labels)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=8)

# Evaluate accuracy on validation set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f'Validation Accuracy: {accuracy*100:.2f}%')
