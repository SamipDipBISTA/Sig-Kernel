#%%
from signature.tensor_algebra import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import umap
np.random.seed(42)  
M= 50
T = 5000
time = np.arange(T)/T

N_wind = 500 #300 and 0.3,4,5,6


increments1 = np.random.normal(0, 1, size=(T, M))
increments2 = np.random.normal(0, 1, size=(T, M))
increments3 = np.random.normal(0, 1, size=(T, M))
increments4 = np.random.normal(0, 1, size=(T, M))

sigma1, sigma2, sigma3, sigma4 = 15, 30, 45, 0.8
theta1, theta2, theta3, theta4 = sigma1**2 / 2, sigma2**2 / 2, sigma3**2 / 2, 0.4
print(theta1, theta2, theta3)

mu = 0.0

def ornstein_uhlenbeck_process(M, T, theta, mu, sigma, dt = 1 / T):

    X = np.zeros((M, T))
    X[:,0] = np.random.normal(0,1, size=(M))
    for i in range(1, T):
        X[:,i] = (
            np.exp(-theta*dt)*X[:, i-1] +
            sigma * np.sqrt((1-np.exp(-2*theta*dt))/(2*theta)) * np.random.normal(0,1, size=(M)))
    plt.plot(time[:], X[0, :])
    return X

S1 = ornstein_uhlenbeck_process(M, T, theta1, mu, sigma1)
S2 = ornstein_uhlenbeck_process(M, T, theta2, mu, sigma2)
S3 = ornstein_uhlenbeck_process(M, T, theta3, mu, sigma3)
#S4 = ornstein_uhlenbeck_process(M, T, theta4, mu, sigma4)

#%%
S = np.vstack([S1, S2])

lam = 1
level = 4

sig = []
sig_efm = []
labels = []
path = []

for i in tqdm(range(M)):
    # print(f'iteration {i} total: {M}')
    path1 = np.vstack([time, S1[i,:]]).T
    path2 = np.vstack([time, S2[i,:]]).T
    path3 = np.vstack([time, S3[i,:]]).T
    # path4 = np.vstack([time, S4[i,:]]).T

    exp_sig1 = path_to_signature(path1, trunc=level).array.T[0] * 0
    exp_sig2 = path_to_signature(path2, trunc=level).array.T[0] * 0
    exp_sig3 = path_to_signature(path3, trunc=level).array.T[0] * 0
    # exp_sig4 = path_to_signature(path3, trunc=level).array.T[0] * 0
    # exp_sig1_efm = path_to_signature(path1, trunc=level).array.T[0] * 0
    # exp_sig2_efm = path_to_signature(path2, trunc=level).array.T[0] * 0
    for k in range(N_wind):
        exp_sig1 += path_to_signature(path1[k * (T // N_wind):(k + 1) * (T // N_wind)], trunc=level).array.T[-1] / N_wind
        exp_sig2 += path_to_signature(path2[k * (T // N_wind):(k + 1) * (T // N_wind)], trunc=level).array.T[-1] / N_wind
        exp_sig3 += path_to_signature(path3[k * (T // N_wind):(k + 1) * (T // N_wind)], trunc=level).array.T[-1] / N_wind
        #exp_sig4 += path_to_signature(path4[k * (T // N_wind):(k + 1) * (T // N_wind)], trunc=level).array.T[-1] / N_wind
        #exp_sig1_efm += path_to_fm_signature(path1[k * (T // N_wind):(k + 1) * (T // N_wind)], trunc=level, t_grid=time[k * (T // N_wind):(k + 1) * (T // N_wind)], lam=lam).array.T[-1] / N_wind
        #exp_sig2_efm += path_to_fm_signature(path2[k * (T // N_wind):(k + 1) * (T // N_wind)], trunc=level, t_grid=time[k * (T // N_wind):(k + 1) * (T // N_wind)], lam=lam).array.T[-1] / N_wind
    # sig1 = path_to_signature(path1, trunc=level).array.T
    # sig2 = path_to_signature(path2, trunc=level).array.T
    # #sig3 = path_to_signature(path3, trunc=level).array.T
    # #sig4 = path_to_signature(path4, trunc=level).array.T

    # sig1_efm = path_to_fm_signature(path1, trunc=level, t_grid=time, lam=lam).array.T
    # sig2_efm = path_to_fm_signature(path2, trunc=level, t_grid=time, lam=lam).array.T
    #sig3_efm = path_to_fm_signature(path3, trunc=level, t_grid=time, lam=lam).array.T
    #sig4_efm = path_to_fm_signature(path4, trunc=level, t_grid=time, lam=lam).array.T

    path.extend([S1[i], S2[i], S3[i]])
    sig.extend([exp_sig1, exp_sig2, exp_sig3]) #, sig3[-1], sig4[-1]])
    #sig_efm.extend([exp_sig1_efm, exp_sig2_efm])
    # sig.extend([sig1[-1], sig2[-1]]) #, sig3[-1], sig4[-1]])
    # sig_efm.extend([sig1_efm[-1], sig2_efm[-1]]) #, sig3_efm[-1], sig4_efm[-1]])
    labels.extend([0, 1, 2])#, 3, 4])

print(labels)

X = np.array(sig)
X_efm = np.array(sig_efm)
labels = np.array(labels)+1
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
#%%
i = 5
j = 10

plt.scatter(X[:,i], X[:,j], c = labels, cmap='winter')
plt.colorbar()

#%%
X = StandardScaler().fit_transform(X)
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.01,
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
plt.title("UMAP projection OU")
plt.show()

















# %%
