from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("./dataset3.csv")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

numeric_features = df.select_dtypes(include=[np.number]).columns
categorical_features = df.select_dtypes(include=[object]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

processed_data = preprocessor.fit_transform(df)

if isinstance(processed_data, np.ndarray) == False:
    processed_data = processed_data.toarray()

pca = PCA(n_components=5)  # 5 компонентів
pca_data = pca.fit_transform(processed_data)

neigh = NearestNeighbors(n_neighbors=5)
nbrs = neigh.fit(pca_data)
distances, indices = nbrs.kneighbors(pca_data)

distances = np.sort(distances[:, 4], axis=0)
plt.plot(distances)
plt.ylabel("5-Nearest Neighbor Distance")
plt.xlabel("Points sorted by distance")
plt.show()

eps_values = [0.3, 0.5]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, eps in enumerate(eps_values):
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan_labels = dbscan.fit_predict(pca_data)

    df[f'Cluster_eps_{eps}'] = dbscan_labels

    axes[i].scatter(pca_data[:, 0], pca_data[:, 1], c=dbscan_labels, cmap="viridis", s=50, alpha=0.7)
    axes[i].set_title(f"DBSCAN Clustering (eps={eps})")
    axes[i].set_xlabel("PCA Component 1")
    axes[i].set_ylabel("PCA Component 2")

plt.tight_layout()
plt.show()

for eps in eps_values:
    unique_labels = set(df[f'Cluster_eps_{eps}'])
    print(f"Unique clusters for eps={eps}: {unique_labels}")
    print(f"Number of clusters for eps={eps}: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")
