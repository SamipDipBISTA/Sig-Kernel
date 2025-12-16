"""
exp_smoothing.py — Function for exponential smoothing
"""

import numpy as np
import matplotlib as plt

def exponential_smoothing(series, smooth_param):
    """
    Apply exponential smoothing to a 1D series.

    Parameters
    ----------
    series : array-like
        The input time series (1D).
    smooth_param : float, default=0.1
        Smoothing parameter in (0,1].
        - smooth_param close to 1 → lesser influence of past values (less smoothing).
        - smooth_param close to 0 → greater influence of past values (more smoothing).

    Returns
    -------
    smoothed : np.ndarray
        Exponentially smoothed series.
    """
    series = np.asarray(series, dtype=float)
    smoothed = np.zeros(len(series))
    smoothed[0] = series[0]
    for t in range(1, len(series)):
        smoothed[t] = smooth_param * series[t] + (1 - smooth_param) * smoothed[t - 1]
    return smoothed




def setup_latex_plotting():
    """Configure Matplotlib for LaTeX rendering and set font parameters."""
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern'
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['axes.titlesize'] = 16

