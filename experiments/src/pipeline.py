
import sys
import os
import time
import warnings

# Add parent directory to path so we can import from nwsig package
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# -------------------------------------------------------
# Imports for updating modifications
# -------------------------------------------------------
import experiments.src.signature_functions as signature_functions
import importlib
importlib.reload(signature_functions)

import experiments.src.optimisation_functions as optimisation_functions
importlib.reload(optimisation_functions)

import experiments.src.regression_functions as regression_functions
importlib.reload(regression_functions)

# -------------------------------------------------------
# Project imports
# -------------------------------------------------------
from nwsig import NWSignatureRegressor, augment_with_time
from experiments.src.signature_functions import *
from experiments.src.regression_functions import *
from experiments.src.optimisation_functions import *


# EFM signatures imports
import signature as efm_sig
from signature.path_signature import path_to_fm_signature
import jax.numpy as jnp
import signature.tensor_algebra as ta


# ===================================================================
#                       EFM PIPELINE
# ===================================================================

class EFM_Pipeline:
    def __init__(
        self,
        alphas=None,
        lambda_bounds=None,
        bandwidth_grid=None,
        trunc=None,
        val_end=None,
        in_sample_T=None,
        path=None,
        y=None,
        time_grid=None,
        train_start=0,
        train_end=None,
        delay=0,
        max_window=0
    ):
        # Hyperparameters
        self.alphas = alphas
        self.lambda_bounds = lambda_bounds
        self.bandwidth_grid = bandwidth_grid
        self.trunc = trunc

        # Data and time information
        self.path = path
        self.y = y
        self.time_grid = time_grid

        # Training/validation split
        self.train_start = train_start
        self.train_end = train_end
        self.val_end = val_end
        self.in_sample_T = in_sample_T

        # Additional settings
        self.delay = delay
        self.max_window = max_window

        # Store validation results
        self.val_results = {}

    # --------------------------------------------------------------

    def EFM_ridge(self):
        """
        EFM + Ridge regression pipeline.
        """

        best_alpha, best_lambda, best_score, _ = ridge_lambda_grid_search(
            alphas=self.alphas,
            path=self.path,
            y=self.y,
            time_grid=self.time_grid,
            trunc=self.trunc,
            lambda_bounds=self.lambda_bounds,
            train_start=self.train_start,
            train_end=self.train_end,
            val_end=self.val_end
        )

        self.val_results['efm+ridge'] = best_score

        # Compute functional signatures
        StatSig = path_to_fm_signature(
            path=self.path,
            trunc=self.trunc,
            t_grid=self.time_grid,
            lam=best_lambda
        )

        X = StatSig.array.T

        # Fit ridge regression and reconstruct
        results = ridge_regression_diff_and_reconstruct(
            X=X,
            y=self.y,
            tune=False,
            train_start=self.train_start,
            train_end=self.train_end,
            alpha=best_alpha,
            verbose=True,
            fit_intercept=True
        )

        return results, self.val_results

    # --------------------------------------------------------------

    def EFM_local_reg(self, X0):
        """
        EFM + Local regression pipeline.
        """

        best_bandwidth, best_lambda, best_score, nw_results = (
            NW_bandwith_lambda_grid_search(
                path=self.path,
                y=self.y,
                X0=X0,
                time_grid=self.time_grid,
                trunc=self.trunc,
                lambda_bounds=self.lambda_bounds,
                train_end=self.train_end,
                val_end=self.val_end,
                bandwidth_grid=self.bandwidth_grid,
                verbose=True
            )
        )

        self.val_results['efm+local_reg'] = best_score

        StatSig = path_to_fm_signature(
            path=self.path,
            trunc=self.trunc,
            t_grid=self.time_grid,
            lam=best_lambda
        )
        X = StatSig.array.T

        results = local_reg_nw_signature(
            X=X,
            y=self.y,
            X0=X0,
            train_start=self.train_start,
            train_end=self.train_end,
            bandwidth=best_bandwidth
        )

        return results, self.val_results


# ===================================================================
#                       SLIDING SIGNATURE PIPELINE
# ===================================================================

class SlidingSig_Pipeline:
    def __init__(
        self,
        alphas=None,
        bandwidth_grid=None,
        windows=None,
        trunc=None,
        val_end=None,
        in_sample_T=None,
        path=None,
        y=None,
        time_grid=None,
        train_start=None,
        train_end=None,
        delay=None,
        max_window=None,
        standard_scale=False,
    ):
        # Hyperparameters
        self.alphas = alphas
        self.bandwidth_grid = bandwidth_grid
        self.windows = windows
        self.trunc = trunc

        # Additional settings
        self.delay = delay
        self.max_window = max_window
        self.standard_scale = standard_scale

        # Data
        self.path = path
        self.y = y
        self.time_grid = time_grid

        # Train/validation
        self.train_start = train_start
        self.train_end = train_end
        self.val_end = val_end
        self.in_sample_T = in_sample_T

        self.val_results = {}

    # --------------------------------------------------------------

    def SlidingSig_ridge(self):
        """
        Sliding signatures + Ridge regression.
        """

        best_window, best_alpha, best_score = ridge_window_grid_search(
            windows=self.windows,
            alphas=self.alphas,
            path=self.path,
            y=self.y,
            max_window=self.max_window,
            train_start=self.train_start,
            train_end=self.train_end,
            val_end=self.val_end,
            trunc_order=self.trunc,
            delay=self.delay,
            verbose=True,
            standard_scale=self.standard_scale
        )

        self.val_results['sliding_sig_ridge'] = best_score

        _, SlideSig_raw = compute_sliding_window_signatures_optimized(
            covariates=self.path,
            window=best_window,
            trunc_order=self.trunc
        )

        # Remove all-zero rows
        SlideSig = SlideSig_raw[np.any(SlideSig_raw != 0, axis=1)]
        results = ridge_regression_diff_and_reconstruct(
            X=SlideSig,
            y=self.y,
            max_window=self.max_window,
            tune=False,
            validation_type='val',
            train_start=self.train_start,
            train_end=self.train_end,
            val_end=None,
            delay=self.delay,
            alpha=best_alpha,
            verbose=True,
            fit_intercept=True
        )

        return results, self.val_results

    # --------------------------------------------------------------

    def SlidingSig_local_reg(self, X0=None):
        """
        Sliding signatures + Local Nadaraya–Watson regression.
        """

        best_window, best_bandwidth, best_score = (
            NW_bandwidth_window_grid_search(
                windows=np.arange(10, 100, 10),
                path=self.path,
                y=self.y,
                X0=X0,
                trunc_order=self.trunc,
                max_window=self.max_window,
                delay=self.delay,
                train_start=self.train_start,
                train_end=self.train_end,
                val_end=self.val_end,
                bandwidth_grid=self.bandwidth_grid,
                verbose=True,
                standard_scale=self.standard_scale,
            )
        )

        self.val_results['sliding_sig_local_reg'] = best_score

        _, SlideSig_raw = compute_sliding_window_signatures_optimized(
            covariates=self.path,
            window=best_window,
            trunc_order=self.trunc
        )

        SlideSig = SlideSig_raw[np.any(SlideSig_raw != 0, axis=1)]

        results = local_reg_nw_signature(
            X=SlideSig,
            y=self.y,
            X0=X0,
            delay=self.delay,
            max_window=self.max_window,
            trunc_order=self.trunc,
            train_start=self.train_start,
            train_end=self.train_end,
            bandwidth=best_bandwidth,
            tune=False,
            verbose=True,
        )

        return results, self.val_results