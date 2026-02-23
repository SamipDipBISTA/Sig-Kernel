import iisignature
import jax.numpy as jnp
import numpy as np
from signature.path_signature import path_to_fm_signature
import time

#compute signatures

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
    """
    Compute EFM signatures for a single path across time steps and
    """
    # Default lam if not provided
    if lam is None:
        lam = jnp.array([0.1])

    # Basic shape checks
    if t_grid.ndim != 2 or covariates.ndim != 2:
        raise ValueError("t_grid and covariates must be 2D arrays (N_paths, N_times).")

    n_times = t_grid.shape[1]
    if covariates.shape[1] != n_times:
        raise ValueError("t_grid and covariates must have the same number of time columns.")


    y = []

    t_grid_shifted = t_grid - t_grid[0][start_idx]        
    sim_path = covariates 
    path = np.vstack([t_grid_shifted, sim_path]).T 
    

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


def compute_sliding_window_signatures(
    covariates: np.ndarray,
    t_grid: np.ndarray,
    trunc: int = 3,
    window: int = 5,
    verbose: bool = True,
):
    if t_grid.ndim != 2 or covariates.ndim != 2:
        raise ValueError("t_grid and covariates must be 2D arrays")

    n_obs = covariates.shape[1]
    y = []
    signatures = []
    t_grid_t = t_grid[0, 0:window]
    sim_path = covariates[0, 0:window]
    path = np.vstack([t_grid_t, sim_path]).T

    Sig = path_to_signature(path=path, trunc=trunc)
    for t in range(window, n_obs):
        if verbose:
            print("Processing time step:", t)

        t_grid_t = t_grid[0, t - window:t]
        sim_path = covariates[0, t - window:t]
        path = np.vstack([t_grid_t, sim_path]).T

        Sig = path_to_signature(path=path, trunc=trunc)
        signatures.append(Sig[:, -1])
        y.append(covariates[0, t])

    return y, signatures



def compute_sliding_window_signatures_optimized(
    covariates: np.ndarray,
    window: int,
    trunc_order: int = 4,
    drop: bool = False
) -> np.ndarray:
    """
    Compute truncated signatures on sliding windows of a time series of covariates.

    Parameters
    ----------
    covariates : np.ndarray
        Input covariates time series of shape (n_samples, n_covariates).
    window : int
        Size of the sliding window (in terms of datapoints).
    trunc_order : int, optional
        Truncation order of the signature (default=4).

    Returns
    -------
    np.ndarray
        Matrix of shape (n_samples, signature_length), where each row
        contains the signature corresponding to the sliding window ending at that index.
    """

    # Time augmentation:
    n_samples = covariates.shape[0]
    time = np.arange(n_samples) / window
    path = np.column_stack((time, np.array(covariates)))

    path_dim = path.shape[1]
    sig_len = iisignature.siglength(path_dim, trunc_order)
    n_windows = n_samples - window
    sig_matrix = np.zeros((n_samples, sig_len))

    # First window:
    first_path = path[:window+1]
    current_sig = iisignature.sig(first_path,trunc_order)
    sig_matrix[window, :] = current_sig
    X0 = [covariates[0,:]]
    # Remaining windows:
    for i in range(1, n_windows):
        # Remove oldest segment
        X0.append(covariates[i,:])
        old_segment = path[i-1:i+1][::-1]
        sig_old = iisignature.sig(old_segment, trunc_order)
        current_sig = iisignature.sigcombine(sig_old, current_sig, path_dim, trunc_order)

        # Add newest segment
        new_segment = path[i-1+window:i+window+1]
        sig_new = iisignature.sig(new_segment, trunc_order)
        current_sig = iisignature.sigcombine(current_sig, sig_new, path_dim, trunc_order)

        sig_matrix[window+i, :] = current_sig
    
    if drop==True :
         # Drop time-related components
        powers = [path_dim**k for k in range(1, trunc_order)]   # d^1, d^2, ..., d^(trunc_order-1)
        indices = np.insert(np.cumsum(powers), 0, 0)     # 0, d, d+d^2, ...
        sig_matrix = np.delete(sig_matrix, indices, axis=1)
     
    return np.array(X0), sig_matrix

