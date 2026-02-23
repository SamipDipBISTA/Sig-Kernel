#%%
from signature.tensor_algebra import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

np.random.seed(42)
M = 100
T = 100
sigma1, sigma2, sigma3, sigma4 = 0.2, 0.4, 0.6, 0.8
theta1, theta2, theta3, theta4 = 0.1,0.2,0.3,0.4
mu = 0.0

def ornstein_uhlenbeck_process(M, T, theta, mu, sigma):
    dt = 1
    X = np.zeros((M, T))
    for i in range(M):
        for t in range(1, T):
            X[i,t] = X[i,t-1] + theta * (mu - X[i,t-1]) * dt + sigma * np.random.normal()
    return X

S1 = ornstein_uhlenbeck_process(M, T, theta1, mu, sigma1)
S2 = ornstein_uhlenbeck_process(M, T, theta2, mu, sigma2)
S3 = ornstein_uhlenbeck_process(M, T, theta3, mu, sigma3)
S4 = ornstein_uhlenbeck_process(M, T, theta4, mu, sigma4)

lam = 2
level = 3
time = np.arange(T)
sig = []
sig_efm = []
labels = []

for i in range(M):
    path1 = np.vstack([time, S1[i,:]]).T
    path2 = np.vstack([time, S2[i,:]]).T
    path3 = np.vstack([time, S3[i,:]]).T
    path4 = np.vstack([time, S4[i,:]]).T

    sig1 = path_to_signature(path1, trunc=level).array.T
    sig2 = path_to_signature(path2, trunc=level).array.T
    sig3 = path_to_signature(path3, trunc=level).array.T
    sig4 = path_to_signature(path4, trunc=level).array.T

    sig1_efm = path_to_fm_signature(path1, trunc=level, t_grid=time, lam=lam).array.T
    sig2_efm = path_to_fm_signature(path2, trunc=level, t_grid=time, lam=lam).array.T
    sig3_efm = path_to_fm_signature(path3, trunc=level, t_grid=time, lam=lam).array.T
    sig4_efm = path_to_fm_signature(path4, trunc=level, t_grid=time, lam=lam).array.T

    sig.extend([sig1[-1], sig2[-1], sig3[-1], sig4[-1]])
    sig_efm.extend([sig1_efm[-1], sig2_efm[-1], sig3_efm[-1], sig4_efm[-1]])
    labels.extend([1, 2, 3, 4])

X = np.array(sig)
X_efm = np.array(sig_efm)
labels = np.array(labels)

perm = np.random.permutation(len(X))
X = X[perm]
X_efm = X_efm[perm]
labels = labels[perm]

def kmeans_silhouette_pca(X, labels, k_min=2, k_max=5):
    best_score = -1
    best_k = k_min

    for k in range(k_min, k_max):
        kmeans = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k

    kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=42).fit(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans.labels_, cmap='tab10', s=50)
    plt.colorbar()
    plt.title(f'Best k: {best_k}')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', s=50)
    plt.colorbar()
    plt.title(f'Best k: {best_k}')
    plt.show()
    accuracy = None
    if len(np.unique(kmeans.labels_))==len(np.unique(labels)):
         accuracy = np.mean(np.abs(kmeans.labels_==(labels-1)))
    return best_k, accuracy

print('Normal sig')
print(kmeans_silhouette_pca(X, labels))
print('EFM sig')
print(kmeans_silhouette_pca(X_efm, labels))
#%%
