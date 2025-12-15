
#%%
from signature import *
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import signature.tensor_algebra as ta
from simulation.diffusion import Diffusion
from arch import arch_model

import matplotlib
# %%
np.random.seed(0)
ret = np.random.randn(1000)

am = arch_model(ret, vol="Garch", p=1, q=1)
res = am.fit(disp="off")

n_sims = 1000
n_obs = 1
simulations = np.zeros((n_sims, n_obs))

for i in range(n_sims):
    sim = am.simulate(res.params, nobs=n_obs)
    simulations[i, :] = sim['data']

W = simulations
t_grid = np.linspace(0, 1, n_sims)

plt.plot(t_grid, W)
# %%
print(W.shape, t_grid.shape)
path = np.column_stack([t_grid, W])
print(path.shape)

#%%

N = 3
Sig = ta.path_to_signature(path=path, trunc=N)
print(Sig[:,-1])
#%%

lam = jnp.array([10])
SigS = ta.path_to_fm_signature(path=path, trunc=N, t_grid=t_grid, lam=lam)

#%%
print('Sig', Sig[:,-1])
print('EFM Sig', SigS[:,-1])
plt.scatter(Sig[:,-1], SigS[:,-1])

# %%
num_iterations = Sig.shape[0]  # number of iterations

for i in range(num_iterations):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

    # Plot Sig
    axes[0].plot(Sig[i, :], color='blue')
    axes[0].set_title(f'Sig - Iteration {i+1}')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Amplitude')

    # Plot SigS
    axes[1].plot(SigS[i, :], color='orange')
    axes[1].set_title(f'EFM Sig - Iteration {i+1}')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
# %%
