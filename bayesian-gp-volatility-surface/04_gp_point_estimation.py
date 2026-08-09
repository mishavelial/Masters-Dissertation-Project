"""
This module fits a point-estimated Gaussian process to residual total variance. Specifically, we load the processed design matrix for a given date, then reconstruct a parametric prior that we fitter in 03_prior_model module.
Later compute residuals around that prior fit a Gaussian-process model to the residual structure.

"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import argparse
import os
import json
import importlib
TotalVariancePrior = importlib.import_module("03_prior_model").TotalVariancePrior


def run_point_gp(date_str, proc_dir="data/processed", prior_dir="results/surfaces", out_dir="results/diagnostics"):
    """Run point-estimated GP calibration for one date.

    Parameters
    ----------
    date_str : str
        Run date in YYYY-MM-DD format.
    proc_dir : str, optional
        Directory containing the processed design matrix.
    prior_dir : str, optional
        Directory containing the fitted prior parameters.
    out_dir : str, optional
        Directory used to store diagnostic output.

    Returns
    -------
    None
        Diagnostics are written to disk.
    """
    print(f"Running point-estimated GP for {date_str}...")

    data = np.load(os.path.join(proc_dir, f"design_matrix_{date_str}.npz"))
    X_train = data['X_raw'][data['train_idx']]
    X_test = data['X_raw'][data['test_idx']]
    w_train = data['w_raw'][data['train_idx']]
    w_test = data['w_raw'][data['test_idx']]
    #load prior
    prior_data = np.load(os.path.join(prior_dir, f"prior_fit_{date_str}.npz"))
    prior = TotalVariancePrior()
    prior.params = prior_data['params']

    #residuals
    res_train = w_train - prior.predict(X_train)
    res_test = w_test - prior.predict(X_test)

    #standardize inputs and residuals for GP stability
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(res_train.reshape(-1, 1))

    X_train_s = scaler_X.transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.transform(res_train.reshape(-1, 1)).ravel()

    #define anisotropic Matérn 5/2, the whitekernel is a covariance function which models white noise iid gaussian. we are adding 10^-5 to the diagonal of the covariance matrix. Refer to B.6 proof of the main text. Where sigma^2 is represented exactly by the WhiteNoise cov. function.
    kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0, 1.0], nu=2.5) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)

    #gp fit
    gp.fit(X_train_s, y_train_s)
    y_pred_s, std_s = gp.predict(X_test_s, return_std=True)
    res_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

    #reconstruct total var and implied var
    w_pred = prior.predict(X_test) + res_pred
    iv_pred = np.sqrt(np.maximum(w_pred / X_test[:, 1], 1e-6))
    iv_true = np.sqrt(np.maximum(w_test / X_test[:, 1], 1e-6))

    rmse_iv = np.sqrt(np.mean((iv_true - iv_pred) ** 2))

    # Extract flat parameters from the optimized theta array (which is log-transformed)
    params = np.exp(gp.kernel_.theta)

    #save the diagnostics
    os.makedirs(out_dir, exist_ok=True)
    results = {
        'lml': float(gp.log_marginal_likelihood_value_),
        'variance_f': float(params[0]),
        'ell_k': float(params[1]),
        'ell_t': float(params[2]),
        'variance_n': float(params[3]),
        'rmse_iv': float(rmse_iv)
    }
    with open(os.path.join(out_dir, f"point_gp_{date_str}.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"Point GP LML: {results['lml']:.2f}, OOS IV RMSE: {rmse_iv:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    args = parser.parse_args()
    run_point_gp(args.date)
