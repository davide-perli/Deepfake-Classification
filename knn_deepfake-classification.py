import cv2, pandas as pd, os, numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

folder_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification"
img_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification/train"
test_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification/test"
test_file = os.path.join(folder_dir, "test.csv")  
csv_file = os.path.join(folder_dir, "train.csv")
validation_file = os.path.join(folder_dir, "validation.csv")
val_dir = "/home/davide/Documents/Deepfake-Classification/deepfake-classification/validation"

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

X_train, y_train, _ = load_images_and_labels(csv_file, img_dir)

knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
knn.fit(X_train, y_train)

train_preds = knn.predict(X_train)
train_acc = accuracy_score(y_train, train_preds)
print(f"Training Accuracy: {train_acc*100:.2f}%")

X_val, y_val, _ = load_images_and_labels(validation_file, val_dir)
val_preds = knn.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

X_test, test_image_ids = load_images_and_labels(test_file, test_dir)
test_preds = knn.predict(X_test)

predictions_df = pd.DataFrame({'image_id': test_image_ids, 'label': test_preds})
predictions_df.to_csv(os.path.join(folder_dir, 'knn_prediction.csv'), index=False)
print("Predictions saved to knn_prediction.csv")
