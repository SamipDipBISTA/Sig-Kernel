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
T = 100
time = np.arange(T)/T

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
    #plt.plot(time[:], X[0, :])
    return X

S1 = ornstein_uhlenbeck_process(M, T, theta1, mu, sigma1)
S2 = ornstein_uhlenbeck_process(M, T, theta2, mu, sigma2)
S3 = ornstein_uhlenbeck_process(M, T, theta3, mu, sigma3)
#S4 = ornstein_uhlenbeck_process(M, T, theta4, mu, sigma4)

S = np.hstack([S2,S3]) #, S3, S4])

plt.plot(S[0])
plt.show()

time = np.arange(2*T)/T
#%%

n = S.shape[1]
lam = 1
window = 1
start = 30
dsig = [0]*start
thres = 10
detect = [0]
for i in range(start, n):
    val = MMD(X = S[:,:i-window],Y = S[:,:i], efm = True, lam=lam, t_grid_X=time[:i-window], t_grid_Y=time[:i])
    dsig.append(val)
    if val>thres:
        detect.append(i)

#%%
std = [np.std(np.array(dsig[start:window]))]
idx = [start+window] 
for i in range(start+window, n-window, window):
    idx.append(i+window)
    std.append(np.std(np.array(dsig[i:i+window])))

plt.plot(dsig)
plt.vlines([i*T for i in range(0,2)], ymin=0, ymax=max(dsig)/2, color = 'red')
#plt.vlines(detect, ymin=0, ymax=max(dsig), color = 'black', linestyles='dotted')
plt.title('MMD vs timestamps')
plt.show()

plt.plot(idx, std)
plt.title('Std of MMD over different intervals')
plt.show()
# %%====================================================================================================
#----------------------------Test with EFMsig + UMAP---------------------------------
from mpl_toolkits.mplot3d import Axes3D
time = np.arange(2*T)/T
path = np.vstack([S,time]).T
lam = 10
S_efm = path_to_fm_signature(path, t_grid=time, lam=lam, trunc=3).array.T

#%%
S_efm = StandardScaler().fit_transform(S_efm)

time = time[:4000]
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=1,
   
)

reducer.fit(S_efm[:4000])
embedding = reducer.transform(S_efm)


# Plot
plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1],c=time*T, cmap='viridis',  s=10)
plt.colorbar()
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP projection")
plt.show()
#%%


x = embedding[:, 0]
y = embedding[:, 1]
z = time*T

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# scatter plot
sc = ax.scatter(x, y, z, c=time*T, cmap='viridis', s=5, alpha=0.7)

xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(y.min(), y.max(), 10))
zz = np.full_like(xx, 250)
ax.plot_surface(xx, yy, zz, color='red', alpha=0.1)

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('')

cbar = plt.colorbar(sc, ax=ax, shrink=0.6)

plt.show()
# %%
plt.scatter( z, y)
#plt.axhline(y=250, color='red', linestyle='--', linewidth=2)
plt.show()
# %%====================================================================================================
#--------------------Test with EFMsig + TSNE---------------------------
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(S_efm)



scatter=plt.scatter(X_tsne[:,0], X_tsne[:,1], c=time*T,
    cmap="viridis", s=5, alpha=0.7)  
plt.colorbar(scatter, label="") 
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.title('Unsupervised t-SNE')
# %%
x = X_tsne[:, 0]
y = X_tsne[:, 1]
z = time * T#embedding[:, 2]  # z-axis

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# scatter plot
sc = ax.scatter(x, y, z, c=time, cmap='viridis', s=5, alpha=0.7)
# labels
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('')
# colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)

plt.show()
#%%
plt.scatter(y,z)