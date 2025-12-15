
import time
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Optional
from signature.path_signature import path_to_fm_signature
def compute_efm_signatures(
    n_obs: int,
    t_grid: np.ndarray,
    simulations: np.ndarray,
    lam: Optional[jnp.ndarray] = None,
    trunc: int = 3,
    verbose: bool = True,
):
    """
    Compute EFM signatures for a single path across time steps and
    collect the corresponding target values.
    
    Parameters
    ----------
    n_obs : int
        Number of observations/time steps to process. The loop will run for t = 1..(n_obs-1).
    t_grid : np.ndarray
        Time grid of shape (N_paths, N_times) or (1, N_times).
        This function uses the first path row: t_grid[0, :t].
    simulations : np.ndarray
        Simulated paths of shape (N_paths, N_times). This function uses simulations[0, :t].
    lam : Optional[jnp.ndarray], default=None
        Damping parameter for the signature function. Defaults to jnp.array([0.1]) if None.
    trunc : int, default=3
        Truncation level passed to `path_to_fm_signature`.
    verbose : bool, default=True
        If True, prints the processed time step index.
    Returns
    -------
    y : List[float]
        List of target values y_t = simulations[0, t] for t = 1..(n_obs-1).
    efm_signatures : List[np.ndarray]
        List of signature vectors at each time step (last column of SigS).
    elapsed : float
        Elapsed wall-clock time (in seconds) for the full computation.
    """
    # Default lam if not provided
    if lam is None:
        lam = jnp.array([0.1])

    # Basic shape checks
    if t_grid.ndim != 2 or simulations.ndim != 2:
        raise ValueError("t_grid and simulations must be 2D arrays (N_paths, N_times).")

    n_times = t_grid.shape[1]
    if simulations.shape[1] != n_times:
        raise ValueError("t_grid and simulations must have the same number of time columns.")

    if n_obs > n_times:
        raise ValueError(f"n_obs ({n_obs}) cannot exceed the number of columns in t_grid ({n_times}).")

    y: List[float] = []
    efm_signatures: List[np.ndarray] = []

    start = time.perf_counter()
    for t in range(1, n_obs):
        if verbose:
            print("Processing time step:", t)

        t_grid_t = t_grid[0, :t]          # shape: (t,)
        sim_path = simulations[0, :t]     # shape: (t,)
        path = np.vstack([t_grid_t, sim_path]).T  # shape: (t, 2)

        # Call the provided signature function
        SigS = path_to_fm_signature(path=path, trunc=trunc, t_grid=t_grid_t, lam=lam)

        # Collect the last signature column (vector at current step)
        efm_signatures.append(SigS[:, -1])

        # Collect the target value at time t for the first path
        y.append(simulations[0, t])

    end = time.perf_counter()
    elapsed = end - start
    print(f"Time taken to compute EFM signatures: {(elapsed)/60:.2f} minutes")
    return y, efm_signatures
