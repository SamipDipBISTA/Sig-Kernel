
#%%
from signature import *
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import signature.tensor_algebra as ta
from simulation.diffusion import Diffusion

import matplotlib
# %%
T = 10

t_grid = jnp.linspace(-T, T, (T * 2) * 365 + 1)
rng = np.random.default_rng(seed=4843)
size = 10

diffusion = Diffusion(t_grid=t_grid - t_grid[0], size=size, rng=rng)
W = diffusion.brownian_motion()[0, 0, :]

fig, ax = plt.subplots()

shift = 365
ax.plot(t_grid, W, label="BM")
ax.plot(t_grid[:-shift], W[shift:], label="Shifted BM")
ax.vlines(0, -3, 2, "k", ls='--')
ax.legend()
ax.set_title("Driving Brownian motion W")

fig.tight_layout()

# %%
W = np.sin(t_grid)
plt.plot(W)
#%%
t_grid_pos = t_grid[t_grid >= 0]
W_pos = W[t_grid >= 0]
W_pos.shape
#%%
path = np.vstack([t_grid, W]).T
path_pos = np.vstack([t_grid_pos, W_pos]).T
dim = path.shape[1]

print(path.shape)
#%%

N = 3
Sig = ta.path_to_signature(path=path_pos, trunc=N)

lam = jnp.array([10])
SigS = ta.path_to_fm_signature(path=path, trunc=N, t_grid=t_grid, lam=lam)
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
