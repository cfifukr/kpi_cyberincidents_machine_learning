import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.neighbors import NearestNeighbors


# Ініціалізація попередньо навченої моделі VGG16 для витягання фіч
model = VGG16(weights='imagenet', include_top=False, pooling='avg')


# Функція для витягання фіч зображення
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()


# Завантажуємо фічі з тренувального набору
def load_dataset_features(folder_path):
    features = []
    labels = []
    for label_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, label_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                feature = extract_features(img_path, model)
                features.append(feature)
                labels.append(label_folder)
    return np.array(features), np.array(labels)


# Завантаження тренувальних фіч з папки ./Animals
train_folder = './Animals'
train_features, train_labels = load_dataset_features(train_folder)

# Ініціалізація моделі LSH
lsh_model = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='cosine')
lsh_model.fit(train_features)


# Функція для ідентифікації зображень з тестової папки ./Test
def classify_test_images(test_folder, lsh_model, train_features, train_labels):
    test_results = {}
    for img_file in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_file)
        test_feature = extract_features(img_path, model).reshape(1, -1)

        # Пошук найближчого сусіда за допомогою LSH
        _, indices = lsh_model.kneighbors(test_feature)
        predicted_label = train_labels[indices[0][0]]

        test_results[img_file] = predicted_label
        print(f"Зображення {img_file} ідентифіковане як: {predicted_label}")
    return test_results


# Ідентифікуємо тестові зображення
test_folder = './Test'
results = classify_test_images(test_folder, lsh_model, train_features, train_labels)
