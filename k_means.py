import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv('attacks.csv')

print(data.dtypes)

#вибір числових стовпців для кластеризації
features = data[['Anomaly Scores', 'Source Port', 'Destination Port', 'Packet Length']]

features = features.dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# визначення оптимальної кількості кластерів за допомогою методу ліктя
k_values = range(1, 11)
inertia = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# побудова графіка для пошуку ліктя
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Інерція')
plt.title('Метод ліктя для оптимального k')
plt.grid()
plt.show()

optimal_k = 8

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(scaled_features)

data['Cluster'] = kmeans.labels_

numeric_data = data.select_dtypes(include=['float64', 'int64'])
grouped = numeric_data.groupby(data['Cluster']).mean(numeric_only=True)
print(grouped)

pca = PCA(n_components=4)
reduced_data = pca.fit_transform(scaled_features)

plt.figure(figsize=(8, 5))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.title('Результати кластеризації K-means')
plt.xlabel('PCA Компонент 1')
plt.ylabel('PCA Компонент 2')
plt.colorbar(label='Мітка кластера')
plt.grid()
plt.show()
