import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

df = pd.read_csv('api.csv')

numerical_features = ['vsession_duration(min)', 'num_sessions', 'num_users', 'num_unique_apis']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numerical_features])

results = []

for n_clusters in range(2, 10):
    Z = linkage(scaled_data, method='ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    score = silhouette_score(scaled_data, labels)
    results.append((n_clusters, score))

n_clusters, scores = zip(*results)
plt.plot(n_clusters, scores, marker='o')
plt.title('Silhouette Score vs Кількість кластерів')
plt.xlabel('Кількість кластерів')
plt.ylabel('Silhouette Score')
plt.xticks(n_clusters)
plt.show()
