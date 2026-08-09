"""
This module serves as a visual diagonstic for our models. By scanning the log marginal likelihood topology of the GP length-scale parameters, we can determine whether the GP objective is smooth or potentially multimodal in the length-scales directions.

"""
import numpy as np
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import os
import argparse
import importlib
TotalVariancePrior = importlib.import_module("03_prior_model").TotalVariancePrior


def scan_topology(date_str, grid_size=30, proc_dir="data/processed", prior_dir="results/surfaces", out_dir="results/diagnostics"):
    """scan_topology function evaluates the GP log marginal likelihood on a 2D length-scale grid.

        Parameters
        ----------
        date_str : str
            Run date in YYYY-MM-DD format.
        grid_size : int, optional
            Number of grid points per length-scale axis.
        proc_dir : str, optional
            Directory containing the processed design matrix.
        prior_dir : str, optional
            Directory containing the fitted prior parameters.
        out_dir : str, optional
            Directory used to store the topology scan output.

        Returns
        -------
        None
            The LML surface is saved to disk as an .npz archive.
    """
    print(f"Scanning LML topology for {date_str}...")

    #load and prep data (same as point GP)
    data = np.load(os.path.join(proc_dir, f"design_matrix_{date_str}.npz"))
    X_train = data['X_raw'][data['train_idx']]
    w_train = data['w_raw'][data['train_idx']]
    prior_data = np.load(os.path.join(prior_dir, f"prior_fit_{date_str}.npz"))
    prior = TotalVariancePrior()
    prior.params = prior_data['params']
    res_train = w_train - prior.predict(X_train)
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(res_train.reshape(-1, 1))
    X_train_s = scaler_X.transform(X_train)
    y_train_s = scaler_y.transform(res_train.reshape(-1, 1)).ravel()

    #define grid for length scales
    ell_k_grid = np.logspace(-1, 1, grid_size)
    ell_t_grid = np.logspace(-1, 1, grid_size)
    LML_matrix = np.zeros((grid_size, grid_size))

    fixed_variance = 1.0
    fixed_noise = 1e-5

    #grid search
    for i, lk in enumerate(ell_k_grid):
        for j, lt in enumerate(ell_t_grid):
            kernel = ConstantKernel(fixed_variance, "fixed") * Matern(length_scale=[lk, lt], nu=2.5) + WhiteKernel(
                noise_level=fixed_noise, noise_level_bounds="fixed")
            #clone kernel to evaluate without fitting
            kernel_eval = kernel.clone_with_theta(kernel.theta)
            #scikit-learn trick: internal LML evaluation
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor(kernel=kernel_eval, optimizer=None)
            gp.fit(X_train_s, y_train_s)
            LML_matrix[i, j] = gp.log_marginal_likelihood_value_

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"topology_{date_str}.npz")
    np.savez(out_path, LML=LML_matrix, ell_k=ell_k_grid, ell_t=ell_t_grid)
    print(f"Topology scan saved to {out_path}. Max LML: {np.max(LML_matrix):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    args = parser.parse_args()
    scan_topology(args.date)
