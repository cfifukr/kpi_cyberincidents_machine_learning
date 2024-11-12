import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications import ResNet50
from datasketch import MinHash, MinHashLSH
from sklearn.preprocessing import Binarizer

# Ініціалізація моделі для витягання фіч
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


# Функція для витягання фіч зображення
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()


# Функція для перетворення вектора фіч на MinHash об'єкт
def create_minhash(vector, num_perm=64):
    # Перетворюємо вектор на бінарний формат
    binarizer = Binarizer(threshold=0.0)
    binary_vector = binarizer.fit_transform(vector.reshape(1, -1))[0]

    # Створюємо MinHash
    m = MinHash(num_perm=num_perm)
    for idx, val in enumerate(binary_vector):
        if val > 0:
            m.update(str(idx).encode('utf8'))
    return m


# Завантаження фіч з тренувального набору і збереження в MinHashLSH
def load_dataset_to_lsh(folder_path, lsh, num_perm=64):
    labels = []
    for label_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, label_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                feature = extract_features(img_path, model)

                # Створюємо MinHash і додаємо в LSH
                minhash = create_minhash(feature, num_perm)
                lsh.insert(img_file, minhash)

                labels.append((img_file, label_folder))
    return labels


# Ініціалізація MinHashLSH
num_perm = 64
lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)

# Завантаження тренувальних фіч у LSH з папки ./Animals
train_folder = './Animals'
labels = load_dataset_to_lsh(train_folder, lsh, num_perm)


# Функція для ідентифікації зображень з тестової папки ./Test
def classify_test_images(test_folder, lsh, labels, num_perm=64):
    test_results = {}
    label_dict = dict(labels)

    for img_file in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_file)
        test_feature = extract_features(img_path, model)

        # Створюємо MinHash для тестового зображення
        test_minhash = create_minhash(test_feature, num_perm)

        # Пошук найближчих сусідів у LSH
        matches = lsh.query(test_minhash)

        if matches:
            predicted_label = label_dict[matches[0]]
            print(f"Зображення {img_file} ідентифіковане як: {predicted_label}")
            test_results[img_file] = predicted_label
        else:
            print(f"Зображення {img_file} не знайдено в тренувальній базі.")
            test_results[img_file] = "Unknown"

    return test_results


# Ідентифікуємо тестові зображення
test_folder = './Test'
results = classify_test_images(test_folder, lsh, labels, num_perm)
