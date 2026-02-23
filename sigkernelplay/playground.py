#%%

import numpy as np
import sigkernel
import matplotlib.pyplot as plt
from signature.tensor_algebra import *
import time
import torch



def compute_efm_signatures(
    n_obs: int,
    t_grid: np.ndarray,
    covariates: np.ndarray,
    signature_type: str ='EFM',
    start_idx: int = 1,
    lam = None,
    trunc: int = 3,
    verbose: bool = True
):

    if t_grid.ndim != 2 or covariates.ndim != 2:
        raise ValueError("t_grid and simulations must be 2D arrays (N_paths, N_times).")

    n_times = t_grid.shape[1]
    if covariates.shape[1] != n_times:
        raise ValueError("t_grid and simulations must have the same number of time columns.")


    y = []

    t_grid_shifted = t_grid - t_grid[0][start_idx]        
    sim_path = covariates
    path = sim_path.T #np.vstack([t_grid_shifted, sim_path]).T 
    

    start = time.perf_counter()
    for t in range(start_idx, n_obs):
         y.append(covariates[0, t])

    print('t_grid:', t_grid_shifted.shape)
    print('Path shape:', path.shape)
    print('y size:', len(y))

    SigS = path_to_fm_signature(path=path, trunc=trunc, t_grid=t_grid_shifted.flatten(), lam=lam)
    end = time.perf_counter()
    elapsed = end - start
    print(f"Time taken to compute EFM signatures: {elapsed:.2f} seconds")
    
    return y, np.array(SigS.array.T)

def EFM_transform(X, lam, t_grid):
    T = np.max(t_grid)
    dX = np.diff(X)         
    t_grid_p1 = t_grid[1:]
    t_grid_ = t_grid[:1] 
    coeff = (np.exp(-lam*(T-t_grid_p1))-np.exp(-lam*(T-t_grid_)))/(lam*(t_grid_p1-t_grid_))
    dX1 = dX*coeff
    X1 = np.cumsum(dX1)
    print('------------------------')
    print('dX:', dX.shape)
    print('coeff:', coeff.shape)
    print('t_grid:', len(t_grid))
    print('------------------------')
    return X1
#===================================================================================================================================================
lam = np.array([0.5])
t_grid = np.arange(0,20)
s_grid = np.arange(0,10)
X = t_grid 
Y = np.cos(s_grid) #s_grid 
 #np.concatenate((Y_, np.zeros(t_grid.shape[0]-s_grid.shape[0])))

'''
X_th = torch.from_numpy(X.reshape(-1,1)).unsqueeze(0).double().cpu()
Y_th = torch.from_numpy(Y.reshape(-1,1)).unsqueeze(0).double().cpu()
print(X_th.shape)
static_kernel = sigkernel.LinearKernel()
signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=1)
print('Normal Kernel:', signature_kernel.compute_kernel(X_th, Y_th))
'''

plt.figure()
plt.plot(X)
plt.plot(Y)
plt.show()


start_idx = 0
'''
Xefm = EFM_transform(X, lam, t_grid).reshape(-1,1)
Yefm = EFM_transform(Y, lam, s_grid).reshape(-1,1)
plt.figure()
plt.plot(Xefm)
plt.plot(Yefm)
plt.show()

#Xefm = torch.from_numpy(Xefm).unsqueeze(0).double().cpu()
#Yefm = torch.from_numpy(Yefm).unsqueeze(0).double().cpu()

#static_kernel = sigkernel.LinearKernel()
#signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=2)

#K_true_efm = signature_kernel.compute_kernel(Xefm, Yefm)
#print("SigKernel EFM:",K_true_efm)
'''
vals_efm = []
levels = range(2, 9)
for level in levels:
    print('trunc:', level)
    _, Stat_sig = compute_efm_signatures(n_obs=len(X), t_grid=t_grid[None,:],covariates=X[None,:], start_idx=start_idx, lam=lam, trunc=level)
    _, Stat_sig2 = compute_efm_signatures(n_obs=len(Y), t_grid=s_grid[None,:],covariates=Y[None,:], start_idx=start_idx, lam=lam, trunc=level)
    xsig = Stat_sig[-1]
    ysig = Stat_sig2[-1]
    print(xsig, ysig)
    vals_efm.append(np.sum(xsig*ysig))

print('values:', vals_efm)
plt.figure()
plt.plot(levels, vals_efm, label="Truncated signature kernel EFM")
#plt.axhline(K_true_efm, linestyle="--", label="SigKernel EFM")
plt.legend()
plt.show()

print(vals_efm)
# %%
