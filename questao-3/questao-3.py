"""
Considere os dados apresentados abaixo. Escolha um dos algoritmos
estudados em sala de aula para resolver esse problema. (K-Médias)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# inicializa k centroides aleatorios a partir dos dados
def initialize_centroids(X, k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

# atribui cada ponto ao centroide mais proximo
def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = [np.linalg.norm(x - c) for c in centroids]
        cluster_id = np.argmin(distances)
        clusters.append(cluster_id)
    return np.array(clusters)

# recalcula os centroides como a media dos pontos de cada cluster
def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroids.append(np.mean(cluster_points, axis=0))
        else:
            new_centroids.append(X[np.random.choice(len(X))])
    return np.array(new_centroids)

# k-medias
def kmeans(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids
    return labels, centroids

# graficos
def plot_graph(X, labels, title, filename, centroids=None):
    import matplotlib.patches as mpatches

    os.makedirs("questao-3/graficos", exist_ok=True)
    plt.figure(figsize=(8, 6))

    colors = ['royalblue', 'seagreen', 'gold', 'purple']
    unique_labels = sorted(set(labels))
    
    legend_patches = []
    for i, label in enumerate(unique_labels):
        color = colors[i % len(colors)]
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], c=color, s=20, label=f"Cluster {label}")
        legend_patches.append(mpatches.Patch(color=color, label=f"Cluster {label}"))

    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='X', label="Centróides")
        legend_patches.append(mpatches.Patch(color='black', label="Centróides"))

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(handles=legend_patches, title="Clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"questao-3/graficos/{filename}", dpi=300)
    plt.close()



if __name__ == "__main__":
    # gerar dados
    X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.5, random_state=23)
    plot_graph(X, y_true, "Dados Gerados com make_blobs", "original.png")

    # k-medias
    k = 4
    labels, centroids = kmeans(X, k)
    plot_graph(X, labels, "Clusters Encontrados com K-Médias", "kmeans_clusters.png", centroids=centroids)

    # contagem de pontos por cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_stats = dict(zip(unique, counts))

    print("\nQuantidade de pontos por cluster:")
    for cluster_id, count in sorted(cluster_stats.items()):
        print(f"Cluster {cluster_id}: {count}")

    print("\nCoordenadas dos centróides encontrados:")
    for i, c in enumerate(centroids):
        print(f"Centróide {i}: {c}")