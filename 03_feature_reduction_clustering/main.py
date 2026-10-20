import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

dataset = load_wine()
X = StandardScaler().fit_transform(dataset.data)
scores = {}
for k in range(2, 6):
    labels = KMeans(n_clusters=k, random_state=31, n_init=10).fit_predict(X)
    scores[k] = silhouette_score(X, labels)
best_k = max(scores, key=scores.get)
labels = KMeans(n_clusters=best_k, random_state=31, n_init=10).fit_predict(X)
coords = PCA(n_components=2).fit_transform(X)
result = pd.DataFrame({"component_1": coords[:, 0], "component_2": coords[:, 1], "cluster": labels})
os.makedirs("output", exist_ok=True)
result.to_csv("output/pca_clusters.csv", index=False)
plt.figure(figsize=(8, 5))
plt.scatter(coords[:, 0], coords[:, 1], c=labels)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Wine Data Clusters")
plt.tight_layout()
plt.savefig("output/pca_clusters.png", dpi=150)
plt.show()
print("Silhouette scores:", scores)
print("Selected clusters:", best_k)
