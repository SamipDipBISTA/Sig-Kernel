


from datetime import datetime

import esig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap

from helper import *

def pca_clustering_workflow(df, scale_func, n_components=2, k_range=(2, 10), color_by='month'):
    X_scaled = scale_func(df)
    
    # PCA
    pca_emb = perform_pca(X_scaled, n_components=n_components)
    
    # Find best k using silhouette score
    best_score = -1
    best_k = k_range[0]
    for k in range(k_range[0], k_range[1] + 1):
        labels = perform_kmeans(pca_emb, n_clusters=k)
        score = silhouette_score(pca_emb, labels)
        if score > best_score:
            best_score = score
            best_k = k

    labels = perform_kmeans(pca_emb, n_clusters=best_k)
    
    # Plot full PCA with clusters
    plot_pca(df, pca_emb, labels=labels, color_by=color_by)
    print(f"Best k: {best_k}, Silhouette score: {best_score:.3f}")
    
    # Plot individual clusters
    unique_clusters = np.unique(labels)
    for cluster in unique_clusters:
        mask = labels == cluster
        print(f'Plot for cluster {cluster}')
        plot_pca(df.iloc[mask], pca_emb[mask], labels=None, color_by=color_by)
    
    return pca_emb, labels, best_k, best_score
