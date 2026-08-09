"""
This module fits a parametric prior to the data and saves the results to an .npz file which we later pass to the deterministic point estimator and other modules in the pipeline.

Outputs
-------
prior_fit_{date_str}.npz
    File that contains a numpy array with the residuals, mae score and the optimised parameters.
params: np.ndarray
    The optimised parameters (a, b, c) in the case of the prior._
residuals: np.ndarray
    Fitted residuals between the training and prediction
mae: float
    MAE loss of the fit, and the MAE score of the fit

Notes
-----
None
"""

import numpy as np
from scipy.optimize import minimize
import os
import argparse


class TotalVariancePrior:
    def __init__(self):
        self.params = np.array([0.01, 0.01, 0.0])  # a, b, c. starting guesses for the optimiser because Nelder-Mead needs a starting point.

    def formula(self, X, params):
        # X[:, 0] = Moneyness (log K/S), X[:, 1] = Maturity (T)
        k = X[:, 0]
        T = X[:, 1]
        a, b, c = params
        #w(k, T) = T * (|a| + |b|k^2 + c*k)
        return T * (np.abs(a) + np.abs(b) * (k ** 2) + c * k)

    def mae_loss(self, params, X, y_true):
        y_pred = self.formula(X, params)
        return np.mean(np.abs(y_true - y_pred))

    def fit(self, X, y):
        res = minimize(self.mae_loss, self.params, args=(X, y), method='Nelder-Mead')
        self.params = res.x
        return self.params

    def predict(self, X):
        return self.formula(X, self.params)

#fit the prior, calculate the mae score and store in the path
def fit_prior(date_str, proc_dir="data/processed", out_dir="results/surfaces"):
    print(f"Fitting parametric prior for {date_str}...")
    data = np.load(os.path.join(proc_dir, f"design_matrix_{date_str}.npz"))
    X_train = data['X_raw'][data['train_idx']]
    w_train = data['w_raw'][data['train_idx']]

    prior = TotalVariancePrior()
    params = prior.fit(X_train, w_train)

    w_pred = prior.predict(X_train)
    mae = np.mean(np.abs(w_train - w_pred))
    residuals = w_train - w_pred

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"prior_fit_{date_str}.npz")
    np.savez(out_path, params=params, prior_mae=mae, residuals=residuals)
    print(f"Prior MAE: {mae:.6f}. Saved to {out_path}")

#makes this modile callable through the command line with a specific date
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    args = parser.parse_args()
    fit_prior(args.date)