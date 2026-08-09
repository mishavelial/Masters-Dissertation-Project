"""
As stated in the main text, the deterministic point estimator's results are sibject to the initialisation position. This module

This module reuses the fitted parametric prior, computes residual total
variance on the training data, standardises the inputs and targets, and then
runs repeated GP fits with different random seeds / initialisations.

The output is a table of fitted kernel parameters and log marginal likelihood
values. This is useful for diagnosing local optima, flat regions, and general
instability in the optimisation landscape.
"""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import os
import argparse
import importlib
TotalVariancePrior = importlib.import_module("03_prior_model").TotalVariancePrior


def run_restart_sensitivity(date_str, n_runs=20, proc_dir="data/processed", prior_dir="results/surfaces", out_dir="results/tables"):
    """This function checks for different fits of the GP depending on the re-runs/initial coordinates.

        Parameters
        ----------
        date_str : str
            Run date in YYYY-MM-DD format.
        n_runs : int, optional
            Number of random initialisations / fits to perform.
        proc_dir : str, optional
            Directory containing the processed design matrix.
        prior_dir : str, optional
            Directory containing the fitted prior parameters.
        out_dir : str, optional
            Directory used to store the summary table.

        Returns
        -------
        None
            Results are written to a CSV file.
    """
    print(f"Running restart sensitivity for {date_str}...")

    data = np.load(os.path.join(proc_dir, f"design_matrix_{date_str}.npz"))
    X_train, w_train = data['X_raw'][data['train_idx']], data['w_raw'][data['train_idx']]

    prior_data = np.load(os.path.join(prior_dir, f"prior_fit_{date_str}.npz"))
    prior = TotalVariancePrior()
    prior.params = prior_data['params']
    #the model doesnt actually train the entire surface, we only model the residual correction. this is done for simplicity
    res_train = w_train - prior.predict(X_train)
    scaler_X, scaler_y = StandardScaler().fit(X_train), StandardScaler().fit(res_train.reshape(-1, 1))
    X_train_s, y_train_s = scaler_X.transform(X_train), scaler_y.transform(res_train.reshape(-1, 1)).ravel()

    results = []

    for seed in range(n_runs):
        #build the gp covariance function.
        kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0, 1.0], nu=2.5) + WhiteKernel(noise_level=1e-5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=seed)

        #inject random noise into initial theta to simulate random starts, intuitively, we are perturbing the existing parameters so that each run starts from a different point in parameter space
        theta_initial = kernel.theta + np.random.normal(0, 1, size=kernel.theta.shape)
        kernel.theta = theta_initial

        gp.fit(X_train_s, y_train_s)
        params = np.exp(gp.kernel_.theta)  #kernel.theta is in the log space, so we convert it back

        results.append({
            'seed': seed,
            'variance': params[0],
            'ell_k': params[1],
            'ell_t': params[2],
            'noise': params[3],
            'lml': gp.log_marginal_likelihood_value_
        })

    df = pd.DataFrame(results)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"restart_runs_{date_str}.csv")
    df.to_csv(out_path, index=False)
    print(f"Completed {n_runs} restarts. Standard deviation of LML: {df['lml'].std():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    args = parser.parse_args()
    run_restart_sensitivity(args.date)