#%%
import sys, os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(parent_dir)
import numpy as np
import importlib

import regime_cluster_detection.MMD.mmd as mmd
from regime_cluster_detection.MMD.mmd import MMD
importlib.reload(mmd)


# Custom Manhattan distance function
def custom_distance(X,Y):
    return MMD(X,Y, bias=True)

# Assign clusters based on custom distance function
def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = [custom_distance(x, c) for c in centroids]
        clusters.append(np.argmin(distances))
    return clusters

# Compute new centroids as mean of assigned points
def compute_centroids(X, labels, k):
    centroids = []
    for i in range(k):
        points = X[np.array(labels) == i]
        centroids.append(points.mean(axis=0))
    return np.array(centroids)

# Main function to perform custom K-Means clustering
def k_means_custom(X, k, max_iter=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iter):
        labels = assign_clusters(X, centroids)
        new_centroids = compute_centroids(X, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Example usage
