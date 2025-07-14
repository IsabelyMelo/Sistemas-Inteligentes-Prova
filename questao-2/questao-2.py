"""
Questão 02 – Considere os dados apresentados abaixo. Escolha um dos algoritmos
estudados em sala de aula para resolver esse problema. (DBSCAN)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# funcao para calcular distancia euclidiana
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# retorna indices dos vizinhos no raio eps
def region_query(X, point_idx, eps):
    neighbors = []
    for i, point in enumerate(X):
        if euclidean_distance(X[point_idx], point) < eps:
            neighbors.append(i)
    return neighbors

# expande o cluster a partir de um ponto nucleo
def expand_cluster(X, labels, point_idx, cluster_id, eps, min_pts):
    neighbors = region_query(X, point_idx, eps)
    if len(neighbors) < min_pts:
        labels[point_idx] = -1  # ruido
        return False
    else:
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                new_neighbors = region_query(X, neighbor_idx, eps)
                if len(new_neighbors) >= min_pts:
                    neighbors += new_neighbors
            i += 1
        return True

# DBSCAN
def dbscan(X, eps, min_pts):
    labels = np.zeros(len(X), dtype=int)
    cluster_id = 0
    for i in range(len(X)):
        if labels[i] != 0:
            continue
        if expand_cluster(X, labels, i, cluster_id + 1, eps, min_pts):
            cluster_id += 1
    return labels

def plot_graph(X, labels, title, filename):
    import matplotlib.patches as mpatches

    os.makedirs("questao-2/graficos", exist_ok=True)
    plt.figure(figsize=(8, 6))

    colors = ['royalblue', 'seagreen']
    unique_labels = sorted(set(labels))
    
    legend_patches = []
    for i, label in enumerate(unique_labels):
        color = 'black' if label == -1 else colors[i % len(colors)]
        cluster_name = 'Ruído' if label == -1 else f'Cluster {label}'
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], c=color, s=20, label=cluster_name)
        legend_patches.append(mpatches.Patch(color=color, label=cluster_name))

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(handles=legend_patches, title="Clusters", loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"questao-2/graficos/{filename}", dpi=300)
    plt.close()


if __name__ == "__main__":
    # gerar dados
    X, y = make_moons(n_samples=500, noise=0.06, random_state=23)

    eps = 0.2
    min_pts = 5
    labels = dbscan(X, eps, min_pts)
    plot_graph(X, labels, "Clusters Detectados pelo DBSCAN", "dbscan_clusters.png")

    unique, counts = np.unique(labels, return_counts=True)
    cluster_stats = dict(zip(unique, counts))

    print("\nQuantidade de pontos por cluster:")
    for cluster_id, count in sorted(cluster_stats.items()):
        if cluster_id == -1:
            print(f"Ruído (-1): {count}")
        else:
            print(f"Cluster {cluster_id}: {count}")
