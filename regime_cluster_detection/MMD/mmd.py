#%%
from signature.tensor_algebra import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import sigkernel
import torch

def efm_transform(X, lam, t_grid):
    """EFM transformation (piece-wise linear approximation) for Sigkernel"""
    T=np.max(t_grid)
    dX=np.diff(X)
    t_grid_p1=t_grid[1:]
    t_grid=t_grid[:-1]
    coeff=(np.exp(-lam*(T - t_grid_p1))-np.exp(-lam*(T - t_grid)))/(lam*(t_grid_p1 - t_grid ))
    dX1=dX*coeff
    X1=np.cumsum(dX1)
    return X1

def MMD(X, Y, bias=False, efm =False, lam = None, t_grid_X = None, t_grid_Y = None):
    """ X,Y of dimension (batch, len) or X,Y of dimension (batch, len, dim)
        Compute MMD distance of X, Y with normal signatures  vs EFM signatures
    """
    print('X',X.shape, t_grid_X.shape )
    print('Y',Y.shape, t_grid_Y.shape )
    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=1)
    if efm:
        if (t_grid_X is None) or (t_grid_Y is None) or (lam is None):
            raise ValueError('Specify time grid and lambda for EFM sig')
        X = efm_transform(X, lam, t_grid_X).reshape(1,-1)
        Y = efm_transform(Y, lam, t_grid_Y).reshape(1,-1)

    #convert to torch for sigkernel
    X_batch = torch.from_numpy(X).double()
    Y_batch = torch.from_numpy(Y).double()

    if X_batch.ndim == 2:
        X_batch = X_batch.unsqueeze(2)
    if Y_batch.ndim == 2:
        Y_batch = Y_batch.unsqueeze(2)

    nx, ny = X.shape[0], Y.shape[0]
    kxx, kyy, kxy = 0, 0, 0
    for i in range(nx):
        xi = X_batch[i:i+1]  
        for j in range(nx):
            xj = X_batch[j:j+1]
            kxx += signature_kernel.compute_kernel(xi, xj)[0]
    for i in range(ny):
        yi = Y_batch[i:i+1]
        for j in range(ny):
            yj = Y_batch[j:j+1]
            kyy += signature_kernel.compute_kernel(yi, yj)[0]
    for i in range(nx):
        xi = X_batch[i:i+1]
        for j in range(ny):
            yj = Y_batch[j:j+1]
            kxy += signature_kernel.compute_kernel(xi, yj)[0]
    
    if bias or (nx==1) or (ny==1): #if nx==1 or ny==1 unbiased has division by 0
        return kxx/(nx**2) + kyy/(ny**2) - 2*kxy/(nx*ny)
    else:
        return kxx/(nx*(nx-1)) + kyy/(ny*(ny-1)) - 2*kxy/(nx*ny)

#%%
if __name__ == '__main__':
    #%%
    np.random.seed(42)  
    M= 10
    T = 100
    sigma1 = 0.2
    sigma2 = 0.3
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

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=1)
    dsig = MMD(S1,S1, bias=True)
    print(dsig)
# %%



