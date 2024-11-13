from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
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

pca = PCA(n_components=5)
pca_data = pca.fit_transform(processed_data)

neigh = NearestNeighbors(n_neighbors=5)
nbrs = neigh.fit(pca_data)
distances, indices = nbrs.kneighbors(pca_data)

distances = np.sort(distances[:, 4], axis=0)
plt.plot(distances)
plt.ylabel("5-Nearest Neighbor Distance")
plt.xlabel("Points sorted by distance")
plt.show()

dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(pca_data)

unique_labels = set(dbscan_labels)
print("Unique clusters:", unique_labels)
print("Number of clusters:", len(unique_labels) - (1 if -1 in unique_labels else 0))

df['Cluster'] = dbscan_labels


plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 4], hue=dbscan_labels, palette="viridis", s=70, alpha=0.9)
plt.title("DBSCAN Clustering Results")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.show()
