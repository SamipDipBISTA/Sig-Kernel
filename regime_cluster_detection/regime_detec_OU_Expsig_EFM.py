#%%
import iisignature as ii
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import umap

from signature.tensor_algebra import *

np.random.seed(42)  
M= 1
T = 1000
time = np.arange(T)/T

N_wind = 100 #300 and 0.3,4,5,6

increments1 = np.random.normal(0, 1, size=(T, M))
increments2 = np.random.normal(0, 1, size=(T, M))
increments3 = np.random.normal(0, 1, size=(T, M))
increments4 = np.random.normal(0, 1, size=(T, M))

sigma1, sigma2, sigma3, sigma4 = 15, 30, 45, 0.8
theta1, theta2, theta3, theta4 = sigma1**2 / 2, sigma2**2 / 2, sigma3**2 / 2, 0.4

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

T2 = 2*T
time2 = np.arange(T2)/T
S = np.hstack([S2,S3]) #, S3, S4])


#%%
plt.plot(time2[:T], S[0,:T])
plt.plot(time2[T:], S[0,T:])
plt.show()

#%%
#OU + Expected Signature
lam = 1
level = 3
exp_window = 10
split_for_umap = 500
efm = False

path = np.vstack([time2, S.ravel()]).T

timestamp = []
exp_sig_arr = []
exp_sig = path_to_signature(path, trunc=level).array.T[0] * 0

for k in tqdm(range(N_wind)):
    timestamp.append((k+1)*T2//(N_wind))
    if not efm:
        exp_sig += path_to_signature(path[k * (T2 // N_wind):(k + 1) * (T2 // N_wind)], trunc=level).array.T[-1] 
    else:
        exp_sig += path_to_fm_signature(path[k * (T2 // N_wind):(k + 1) * (T2 // N_wind)], trunc=level,t_grid=time2[k * (T2 // N_wind):(k + 1) * (T2 // N_wind)],lam=lam).array.T[-1] 
    exp_sig_arr.append(exp_sig/(k+1))
exp_sig_arr = np.array(exp_sig_arr)
timestamp = np.array(timestamp)/1000

######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
'''
cumsum = exp_sig_prev =  path_to_signature(path, trunc=level).array.T[0] * 0

for k in tqdm(range(N_wind)):
    
    #timestamp.append((k+1)*T2//(N_wind))
    if not efm:
        cumsum += path_to_signature(path[k * (T2 // N_wind):(k + 1) * (T2 // N_wind)+1], trunc=level).array.T[-1] 
    else:
        cumsum += path_to_fm_signature(path[k * (T2 // N_wind):(k + 1) * (T2 // N_wind)+1], trunc=level,t_grid=time2[k * (T2 // N_wind):(k + 1) * (T2 // N_wind)],lam=lam).array.T[-1] 
    if wdw == exp_window:
        timestamp.append((k+1)*T2//(N_wind))
        exp_sig = cumsum - exp_sig_prev
        exp_sig_arr.append(exp_sig/exp_window)
        exp_sig_prev = cumsum
        wdw = 1
    wdw +=1
    
exp_sig_arr = np.array(exp_sig_arr)
timestamp = np.array(timestamp)/1000
'''
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################

'''
step_sigs = []
for k in tqdm(range(N_wind)):
    start = k * (T2 // N_wind)
    end = (k + 1) * (T2 // N_wind) + 1 # 
    
    seg = path[start:end]
    
    if not efm:
        sig = path_to_signature(seg, trunc=level).array.T[-1]
    else:
        t_seg = time2[start:end]
        sig = path_to_fm_signature(seg, trunc=level, t_grid=t_seg, lam=lam).array.T[-1]
    
    step_sigs.append(sig)

zero_sig = np.zeros_like(step_sigs[0])
prefix_sums = np.cumsum([zero_sig] + step_sigs, axis=0)

# 3. Extract Rolling Averages
exp_sig_arr = []
timestamp = []

for k in range(exp_window, len(prefix_sums)):

    window_sum = prefix_sums[k] - prefix_sums[k - exp_window]
    exp_sig_arr.append(window_sum / exp_window)

    timestamp.append(k * (T2 // N_wind))
'''
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
#%%
X = StandardScaler().fit_transform(exp_sig_arr)
reducer = umap.UMAP(
    n_neighbors=20,
    min_dist=0.01,
    n_components=2,
    random_state = 42,
   
)

reducer.fit(X[:split_for_umap])
embedding = reducer.transform(X)
# Plot
plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1],c=timestamp, cmap='gnuplot',  s=10)
plt.colorbar()
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP projection with expected signatures")
plt.show()

x = embedding[:, 0]
y = embedding[:, 1]
z = timestamp

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(x, y, z, c=z, cmap='gnuplot', s=5, alpha=0.7)

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('Time')

plt.title('UMAP with expected signatures')
# colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)

plt.show()
#%%
np.where(embedding[:,0]<10)[0]

#%%
#Raw Signatures
if not efm:
    exp_sig_arr1 = path_to_signature(path, trunc=level).array.T
else:
    exp_sig_arr1 = path_to_fm_signature(path, trunc=level,t_grid=time2,lam=lam).array.T
exp_sig_arr1 = np.array(exp_sig_arr1)


X1 = StandardScaler().fit_transform(exp_sig_arr1)
reducer1 = umap.UMAP(
    n_neighbors=15,
    min_dist=0.01,
    n_components=2,
    random_state = 42,
   
)

reducer1.fit(X1[:split_for_umap])
embedding1 = reducer1.transform(X1)
# Plot
plt.figure()
plt.scatter(embedding1[:, 0], embedding1[:, 1],c=time2, cmap='gnuplot',  s=10)
plt.colorbar()
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP projection with raw signatures")
plt.show()

x = embedding1[:, 0]
y = embedding1[:, 1]
z = time2

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(-x, y, z, c=z, cmap='gnuplot', s=5, alpha=0.7)

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('Time')
plt.title('UMAP with raw signatures')

# colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)

plt.show()
# %%
