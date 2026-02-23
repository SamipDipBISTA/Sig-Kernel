#%%
import sys, os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(parent_dir)


import importlib
from signature.tensor_algebra import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import umap

import regime_cluster_detection.MMD.mmd as mmd
from regime_cluster_detection.MMD.mmd import MMD
importlib.reload(mmd)

np.random.seed(42)  
M= 1
T = 1000
time = np.arange(T)/1000

N_wind = 100 #300 and 0.3,4,5,6
sigma1 = 0.35
sigma2 = 0.4
sigma3 = 0.6
sigma4 = 0.8

increments1 = np.random.normal(0, 1, size=(T, M))
increments2 = np.random.normal(0, 1, size=(T, M))
increments3 = np.random.normal(0, 1, size=(T, M))
increments4 = np.random.normal(0, 1, size=(T, M))

S1 = sigma1 * np.cumsum(increments1, axis=1).T
S2 = sigma2 * np.cumsum(increments2, axis=1).T
S3 = sigma3 * np.cumsum(increments3, axis=1).T
S4 = sigma4 * np.cumsum(increments4, axis=1).T

window_size = T // N_wind  
S1_batched = np.array([S1[:, k*window_size : (k+1)*window_size] for k in range(N_wind)])
S2_batched = np.array([S2[:, k*window_size : (k+1)*window_size] for k in range(N_wind)])
S3_batched = np.array([S3[:, k*window_size : (k+1)*window_size] for k in range(N_wind)])
S4_batched = np.array([S4[:, k*window_size : (k+1)*window_size] for k in range(N_wind)])

S1_batched = np.squeeze(S1_batched, axis=1)
S2_batched = np.squeeze(S2_batched, axis=1)
S3_batched = np.squeeze(S3_batched, axis=1)
S4_batched = np.squeeze(S4_batched, axis=1)


#%%
all_batched = [S1_batched, S2_batched, S3_batched, S4_batched]
N = len(all_batched)

dist_matrix = np.zeros((N, N))

for i in range(N):
    for j in range(i+1, N):  
        d = MMD(all_batched[i], all_batched[j], bias=True)
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d  

print("Distance matrix:\n", dist_matrix)
#%%
X = StandardScaler().fit_transform(X)

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric = "precomputed"
)

embedding = reducer.fit_transform(dist_matrix)

# Plot
plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1], cmap='tab10',  s=10)
plt.colorbar()
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP projection")
plt.show()
#%%
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
#%%
np.random.seed(42)
N = 4
M = 1
T = 1000
N_wind = 100
sigma1 = np.linspace(0.2,0.3,N)
sigma2 = np.linspace(0.3,0.4,N)[::-1]
window_size = T // N_wind

all_paths_batched1 = []
all_paths_batched2 = []
for sigma in sigma1:
    increments = np.random.normal(0, 1, size=(T, M))
    S = sigma * np.cumsum(increments, axis=0).T
    batched = np.array([S[:, k*window_size : (k+1)*window_size] for k in range(N_wind)])
    batched = np.squeeze(batched, axis=1)
    all_paths_batched1.append(batched)

for sigma in sigma2:
    increments = np.random.normal(0, 1, size=(T, M))
    S = sigma * np.cumsum(increments, axis=0).T
    batched = np.array([S[:, k*window_size : (k+1)*window_size] for k in range(N_wind)])
    batched = np.squeeze(batched, axis=1)
    all_paths_batched2.append(batched)
#%%
dist_matrix1 = []
for i in range(N):
    dist_matrix1.append(MMD(all_paths_batched1[i], all_paths_batched2[i]))
#%%


#%%
plt.plot(np.abs(sigma2-sigma1), dist_matrix1)