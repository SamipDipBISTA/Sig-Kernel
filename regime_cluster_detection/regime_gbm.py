#%%
from signature.tensor_algebra import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import umap
#%%
np.random.seed(42)  

M = 50
T = 5000      
dt = 1/1000
time = np.arange(T) * dt

N_wind = 500 #300 and 0.3,4,5,6
sigma1 = 0.2
sigma2 = 0.4
sigma3 = 0.6
sigma4 = 0.8
mu1 = 0
mu2 = 0
mu3 = 0
mu4 = 0
increments1 = np.random.normal(0, np.sqrt(dt), size=(T, M)).T
increments2 = np.random.normal(0, np.sqrt(dt), size=(T, M)).T
increments3 = np.random.normal(0, np.sqrt(dt), size=(T, M)).T
increments4 = np.random.normal(0, np.sqrt(dt), size=(T, M)).T

# Brownian motion
W1 = np.cumsum(increments1, axis=1)
W2 = np.cumsum(increments2, axis=1)
W3 = np.cumsum(increments3, axis=1)
W4 = np.cumsum(increments4, axis=1)

# GBM with mu = 0 and S0 = 1
S1 = 2*np.exp((mu1-0.5 * sigma1**2 )* time + sigma1 * W1)
S2 = 4*np.exp((mu2-0.5 * sigma2**2 )* time + sigma2 * W2)
S3 = 10*np.exp((mu3-0.5 * sigma3**2 )* time + sigma3 * W3)
S4 = np.exp(-0.5 * sigma4**2 * time + sigma4 * W4)

#%%


lam = 10
level = 3

sig = []
sig_efm = []
labels = []
path = []

for i in range(M):
    print(f'iteration {i} total: {M}')
    path1 = np.vstack([time, S1[i,:]]).T
    path2 = np.vstack([time, S2[i,:]]).T
    path3 = np.vstack([time, S3[i,:]]).T
    path4 = np.vstack([time, S4[i,:]]).T

    exp_sig1 = path_to_signature(path1, trunc=level).array.T[0] * 0
    exp_sig2 = path_to_signature(path2, trunc=level).array.T[0] * 0
    exp_sig3 = path_to_signature(path3, trunc=level).array.T[0] * 0
    #exp_sig4 = path_to_signature(path3, trunc=level).array.T[0] * 0
   
    #exp_sig1_efm = path_to_signature(np.zeros_like(path1), trunc=level).array.T[0] * 0
    #exp_sig2_efm = path_to_signature(np.zeros_like(path1), trunc=level).array.T[0] * 0
    #exp_sig3_efm = path_to_signature(np.zeros_like(path1), trunc=level).array.T[0] * 0
    for k in range(N_wind):
        exp_sig1 += path_to_signature(path1[k * (T // N_wind):(k + 1) * (T // N_wind)], trunc=level).array.T[-1] / N_wind
        exp_sig2 += path_to_signature(path2[k * (T // N_wind):(k + 1) * (T // N_wind)], trunc=level).array.T[-1] / N_wind
        exp_sig3 += path_to_signature(path3[k * (T // N_wind):(k + 1) * (T // N_wind)], trunc=level).array.T[-1] / N_wind
        #exp_sig4 += path_to_signature(path4[k * (T // N_wind):(k + 1) * (T // N_wind)], trunc=level).array.T[-1] / N_wind
        #exp_sig1_efm += path_to_fm_signature(path1[k * (T // N_wind):(k + 1) * (T // N_wind)], trunc=level, t_grid=time[k * (T // N_wind):(k + 1) * (T // N_wind)], lam=lam).array.T[-1] / N_wind
        #exp_sig2_efm += path_to_fm_signature(path2[k * (T // N_wind):(k + 1) * (T // N_wind)], trunc=level, t_grid=time[k * (T // N_wind):(k + 1) * (T // N_wind)], lam=lam).array.T[-1] / N_wind
        #exp_sig3_efm += path_to_fm_signature(path3[k * (T // N_wind):(k + 1) * (T // N_wind)], trunc=level, t_grid=time[k * (T // N_wind):(k + 1) * (T // N_wind)], lam=lam).array.T[-1] / N_wind
    '''
    # sig1 = path_to_signature(path1, trunc=level).array.T
    # sig2 = path_to_signature(path2, trunc=level).array.T
    # #sig3 = path_to_signature(path3, trunc=level).array.T
    # #sig4 = path_to_signature(path4, trunc=level).array.T

    sig1_efm = path_to_fm_signature(path1, trunc=level, t_grid=time, lam=lam).array.T
    sig2_efm = path_to_fm_signature(path2, trunc=level, t_grid=time, lam=lam).array.T
    sig3_efm = path_to_fm_signature(path3, trunc=level, t_grid=time, lam=lam).array.T
     '''
    #sig4_efm = path_to_fm_signature(path4, trunc=level, t_grid=time, lam=lam).array.T

    path.extend([S1[i], S2[i], S3[i]])
    sig.extend([exp_sig1, exp_sig2, exp_sig3]) #, , sig4[-1]])
    #sig_efm.extend([exp_sig1_efm, exp_sig2_efm, exp_sig3_efm ])
    #sig.extend([sig1[-1], sig2[-1]]) #, sig3[-1], sig4[-1]])
    #sig_efm.extend([sig1_efm[-1], sig2_efm[-1], sig3_efm[-1]]) #, sig3_efm[-1], sig4_efm[-1]])
    labels.extend([1,2,3])#, 3, 4])

print(labels)

X = np.array(sig)
X_efm = np.array(sig_efm)
labels = np.array(labels)
print(X.shape)
#%%
# perm = np.random.permutation(len(X))
# X = X[perm]
# S = S[perm]
# X_efm = X_efm[perm]
# labels = labels[perm]

def kmeans_silhouette_pca(X, labels, k_min=2, k_max=5, model='kmeans'):
    best_score = -1
    best_k = k_min

    for k in range(k_min, k_max):
        kmeans = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k

    kmeans = KMeans(n_clusters=3, n_init=5, random_state=42).fit(X)
    spectral = SpectralClustering(n_clusters=3).fit(X)
    if model=='kmeans':
        model = kmeans
    else:
        model = spectral
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=model.labels_, cmap='winter', s=50)
    plt.colorbar()
    plt.title(f'Best k: {best_k}')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='winter', s=50)
    plt.colorbar()
    plt.title(f'Best k: {best_k}')
    plt.show()

    plt.figure(figsize=(8,6))
    # print(spectral.labels_)
    print(model)
    for i in range(len(S)):
        c = "red" if model.labels_[i] else "blue"
        plt.plot(path[i], color=c)


    plt.figure(figsize=(8,6))
    # print(spectral.labels_)
    for i in range(len(S)):
        c = "red" if labels[i] else "blue"
        plt.plot(path[i], color=c)

    accuracy = None
    if len(np.unique(model.labels_))==len(np.unique(labels)):
         accuracy = np.mean(np.abs(model.labels_==(labels)))
    print('Accuracy:', accuracy)
    return best_k

print('Normal sig')
print(kmeans_silhouette_pca(X, labels))
# print('EFM sig')
# print(kmeans_silhouette_pca(X_efm, labels))
#%%
print('Normal sig')
print(kmeans_silhouette_pca(X, labels,model='sp'))
#%%
print('EFM sig')
print(kmeans_silhouette_pca(X_efm, labels))
#%%
print('EFM sig')
print(kmeans_silhouette_pca(X_efm, labels, model='sp'))

#%%
i = 8
j = 10

plt.scatter(X[:,i], X[:,j], c = labels, cmap='winter')
plt.colorbar()

#%%
X = StandardScaler().fit_transform(X)
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state = 42,
)

embedding = reducer.fit_transform(X)

# Plot
plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1],c=labels, cmap='tab10',  s=10)
plt.colorbar()
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP projection GBM")
plt.show()


# %%
