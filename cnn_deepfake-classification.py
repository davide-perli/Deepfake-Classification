import cv2
import matplotlib.pyplot as plt 
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torchvision.transforms.v2 as T
import seaborn as sns
from PIL import Image, ImageEnhance
from torch.utils.data import TensorDataset, DataLoader
from os import listdir
from sklearn.metrics import confusion_matrix

folder_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification"
img_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification/train"

test_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification/test"
test_file = os.path.join(folder_dir, "test.csv")  

csv_file = os.path.join(folder_dir, "train.csv")

validation_file = os.path.join(folder_dir, "validation.csv")
val_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification/validation"

df_train = pd.read_csv(csv_file)

def augment_image(image):
    image_pil = Image.fromarray(image)
    
    image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
    image_pil = image_pil.transpose(Image.FLIP_TOP_BOTTOM)

    image_pil = image_pil.rotate(40)

    enhancer = ImageEnhance.Brightness(image_pil)
    image_pil = enhancer.enhance(1.185)
    
    image_pil = ImageEnhance.Color(image_pil).enhance(0.8)
    image_pil = ImageEnhance.Sharpness(image_pil).enhance(0.968)

    return np.array(image_pil)

augmented_data = []
print(f"Training set size: {len(df_train)}")
for idx in range(len(df_train)):
    row = df_train.iloc[idx]
    img_path = os.path.join(img_dir, row['image_id'] + '.png')
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_resized = cv2.resize(image_rgb, (100, 100))
    
    augmented_image = augment_image(image_resized)
    
    normalized = cv2.normalize(augmented_image.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    tensor_image = torch.from_numpy(normalized).permute(2, 0, 1).float()
    
    augmented_data.append((tensor_image, row['label']))

print(f"Finished generating augmented images")

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_3 = nn.BatchNorm2d(128)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool(x)

        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x = self.pool(x)

        x = self.global_avg_pool(x) 
        x = torch.flatten(x, 1)      
        x = self.dropout(x)
        x = self.fc(x)
        return x

    
model = NeuralNet()

data = []
for idx, row in df_train.iterrows():
    img_name = row['image_id'] + '.png'
    label = row['label']
    img_path = os.path.join(img_dir, img_name)
    image = cv2.imread(img_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # height, width, channels = image_rgb.shape

    # print(f"Original image width: {width} pixels")   100
    # print(f"Original image height: {height} pixels") 100
    # print(f"Number of channels: {channels}")         3
    image_resized = cv2.resize(image_rgb, (100, 100))
    normalized_image = cv2.normalize(image_resized.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)

    image_tensor = torch.from_numpy(normalized_image).permute(2, 0, 1).float()

    data.append((image_tensor, label))

data += augmented_data
print(f"Total training data size: {len(data)}")

images = torch.stack([item[0] for item in data])
labels = torch.tensor([item[1] for item in data])

dataset = TensorDataset(images, labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# residual blocks and skip connection 

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.82, weight_decay=1e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=2000)

# 186
NUM_EPOCHS = 186
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

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

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
    # scheduler.step(epoch_loss)
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    # if epoch_loss < 0.001:
    #     print(f"Stopping early at epoch {epoch+1} as loss is {epoch_loss: .4f}")
    #     break
##################################################################################
print('Finished training')
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, train_accuracies, 'r-', label='Training Accuracy')
plt.xlabel('Epoch')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.show()

df_test = pd.read_csv(test_file)

test_data = []
image_ids = []
for idx, row in df_test.iterrows():
    img_name = row['image_id'] + '.png'
    img_path = os.path.join(test_dir, img_name)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_resized = cv2.resize(image_rgb, (100, 100))
    
    normalized_image = cv2.normalize(image_resized.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    image_tensor = torch.from_numpy(normalized_image).permute(2, 0, 1).float()
    test_data.append(image_tensor)
    image_ids.append(row['image_id'])

test_images = torch.stack(test_data)
test_dataset = TensorDataset(test_images) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

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
predictions_df = pd.DataFrame({'image_id': image_ids, 'label': predicted_labels})
predictions_df.to_csv('deepfake-classification/cnn_prediction.csv', index=False)
print("Predictions saved to prediction.csv")


df_val = pd.read_csv(validation_file)

val_data = []
validation_predictions = []
validation_correct_labels = []
for idx, row in df_val.iterrows():
    img_name = row['image_id'] + '.png'
    label = row['label']
    img_path = os.path.join(val_dir, img_name)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_resized = cv2.resize(image_rgb, (100, 100))
    
    normalized_image = cv2.normalize(image_resized.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    image_tensor = torch.from_numpy(normalized_image).permute(2, 0, 1).float()
    val_data.append((image_tensor, label))

val_images = torch.stack([item[0] for item in val_data])
val_labels = torch.tensor([item[1] for item in val_data])

val_dataset = TensorDataset(val_images, val_labels)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        validation_predictions.extend(preds.cpu().numpy())
        validation_correct_labels.extend(labels.cpu().numpy())
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f'Validation Accuracy: {accuracy*100:.2f}%')

conf_matrix = confusion_matrix(validation_correct_labels, validation_predictions)
print(f"Confusion Matrix:\n{conf_matrix}")

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 92.88%