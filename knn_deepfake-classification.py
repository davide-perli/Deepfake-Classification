import cv2, pandas as pd, os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image, ImageEnhance

folder_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification"
img_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification/train"
test_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification/test"
test_file = os.path.join(folder_dir, "test.csv")  
csv_file = os.path.join(folder_dir, "train.csv")
validation_file = os.path.join(folder_dir, "validation.csv")
val_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification/validation"

df = pd.read_csv(csv_file)

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

augmented_features = []
augmented_labels = []

for idx in range(len(df)):
    row = df.iloc[idx]
    img_path = os.path.join(img_dir, row['image_id'] + '.png')
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (100, 100))
    
    augmented_image = augment_image(image_resized)

    image_hsv = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2HSV)
    
    hist = cv2.calcHist([image_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
  
    augmented_features.append(hist)
    augmented_labels.append(row['label'])


print(f"Finished generating augmented images")

def load_images_and_labels(csv_path, img_folder, flatten=True):
    df = pd.read_csv(csv_path)
    features = []
    labels = []
    image_ids = []
    for idx, row in df.iterrows():
        img_name = row['image_id'] + '.png'
        img_path = os.path.join(img_folder, img_name)
        image = cv2.imread(img_path)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_resized = cv2.resize(image_hsv, (100, 100))

        hist = cv2.calcHist([image_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten() 

        features.append(hist)
        if 'label' in row:
            labels.append(row['label'])
        image_ids.append(row['image_id'])

    if labels:
        return np.array(features), np.array(labels), image_ids
    else:
        return np.array(features), image_ids

X_orig, y_orig, _ = load_images_and_labels(csv_file, img_dir)

X_train = np.vstack([X_orig, np.array(augmented_features)])
y_train = np.hstack([y_orig, np.array(augmented_labels)])

knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
knn.fit(X_train, y_train)

train_preds = knn.predict(X_train)
train_acc = accuracy_score(y_train, train_preds)
print(f"Training Accuracy: {train_acc*100:.2f}%")

X_val, y_val, _ = load_images_and_labels(validation_file, val_dir)
val_preds = knn.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
conf_matrix = confusion_matrix(y_val, val_preds)
print(f"Confusion Matrix:\n{conf_matrix}")

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

X_test, test_image_ids = load_images_and_labels(test_file, test_dir)
test_preds = knn.predict(X_test)

predictions_df = pd.DataFrame({'image_id': test_image_ids, 'label': test_preds})
predictions_df.to_csv(os.path.join(folder_dir, 'knn_prediction.csv'), index=False)
print("Predictions saved to knn_prediction.csv")
