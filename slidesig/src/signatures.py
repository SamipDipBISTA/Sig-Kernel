"""
sliding_signatures.py — Functions to compute signatures on a sliding window.
"""

import numpy as np
import iisignature 

def sig_sliding(
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

    # Remaining windows:
    for i in range(1, n_windows):
        # Remove oldest segment
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
     
    return sig_matrix

