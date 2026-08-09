import os
import argparse
import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import importlib

#dynamic imports matching workspace structure
TotalVariancePrior = importlib.import_module("03_prior_model").TotalVariancePrior
utils = importlib.import_module("utils")


def matern52_kernel(X, Z, lengthscales, variance):
    """
    Computes the Matérn 5/2 kernel covariance matrix using JAX operations.
    """
    X_scaled = X / lengthscales
    Z_scaled = Z / lengthscales
    dist_sq = jnp.sum(X_scaled ** 2, axis=-1)[:, None] + jnp.sum(Z_scaled ** 2, axis=-1) - 2 * jnp.dot(X_scaled, Z_scaled.T)
    dist_sq = jnp.maximum(dist_sq, 0.0)
    d = jnp.sqrt(dist_sq + 1e-12)
    return variance * (1.0 + jnp.sqrt(5.0) * d + (5.0 / 3.0) * dist_sq) * jnp.exp(-jnp.sqrt(5.0) * d)


def predict_gp(X_train, y_train, X_eval, ell_k, ell_t, var_f, var_n, jitter=1e-4):
    """
    Predicts out-of-sample points using Cholesky factorisation in JAX. Slightly higher default jitter (1e-4) ensures numerical stability.
    """
    ell = jnp.array([ell_k, ell_t])
    K_XX = matern52_kernel(X_train, X_train, ell, var_f) + (var_n + jitter) * jnp.eye(X_train.shape[0])
    K_XZ = matern52_kernel(X_train, X_eval, ell, var_f)
    L = jax.scipy.linalg.cho_factor(K_XX, lower=True)
    alpha = jax.scipy.linalg.cho_solve(L, y_train)
    return jnp.dot(K_XZ.T, alpha)


#vectorised prediction mapping for sampling across posterior sub-batches
vmap_predict = jax.vmap(predict_gp, in_axes=(None, None, None, 0, 0, 0, 0))


def compare_methods(date_str, proc_dir="data/processed", prior_dir="results/surfaces",
                    hmc_dir="results/diagnostics", tables_dir="results/tables"):
    """
    Loads point-estimated and Bayesian HMC outputs, calculates OOS RMSE, and checks for calendar/butterfly arbitrage violations.
    """
    #safe validation of processed design matrix
    proc_path = os.path.join(proc_dir, f"design_matrix_{date_str}.npz")
    if not os.path.exists(proc_path):
        print(f"Skipping {date_str}: Processed design matrix not found at {proc_path}")
        return None

    # safe validation of prior parametric surface fit
    prior_path = os.path.join(prior_dir, f"prior_fit_{date_str}.npz")
    if not os.path.exists(prior_path):
        print(f"Skipping {date_str}: Parametric prior fit file not found at {prior_path}")
        return None

    print(f"Calculating out-of-sample metrics for {date_str}...")

    #soad dataset features
    data = np.load(proc_path)
    X_train, w_train = data['X_raw'][data['train_idx']], data['w_raw'][data['train_idx']]
    X_test, w_test = data['X_raw'][data['test_idx']], data['w_raw'][data['test_idx']]
    iv_test = data['iv_raw'][data['test_idx']]
    spot = data['spot']

    #load prior parametric mean surface
    prior_data = np.load(prior_path)
    prior = TotalVariancePrior()
    prior.params = prior_data['params']
    res_train = w_train - prior.predict(X_train)
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(res_train.reshape(-1, 1))
    X_train_s = scaler_X.transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.transform(res_train.reshape(-1, 1)).ravel()

    k_grid_1d = np.linspace(-0.2, 0.2, 30)
    t_grid_1d = np.linspace(0.1, 2.0, 20)
    K_mesh, T_mesh = np.meshgrid(k_grid_1d, t_grid_1d, indexing='ij')
    X_grid = np.column_stack((K_mesh.ravel(), T_mesh.ravel()))
    X_grid_s = scaler_X.transform(X_grid)

    prior_test = prior.predict(X_test)
    prior_grid = prior.predict(X_grid)
    point_params_path = os.path.join(prior_dir, f"point_fit_{date_str}.npz")
    restarts_path = os.path.join(tables_dir, f"restart_runs_{date_str}.csv")

    if os.path.exists(point_params_path):
        #loaded directly from script 04 outputs
        point_params = np.load(point_params_path)
        var_f = point_params['var_f']
        ell_k = point_params['ell_k']
        ell_t = point_params['ell_t']
        var_n = point_params['var_n']
    elif os.path.exists(restarts_path):
        #loaded from script 06 restarts csv
        restarts_df = pd.read_csv(restarts_path)
        best_point = restarts_df.loc[restarts_df['lml'].idxmax()]
        var_f = best_point['variance']
        ell_k = best_point['ell_k']
        ell_t = best_point['ell_t']
        var_n = best_point['noise']
    else:
        print(f"Skipping {date_str}: No point estimation parameters found.")
        print(f"Ensure 04_gp_point_estimation or 06_restart_sensitivity ran successfully.")
        return None

    #construct the Point-Estimated baseline GP model
    point_kernel = ConstantKernel(var_f, "fixed") * \
                   Matern(length_scale=[ell_k, ell_t], length_scale_bounds="fixed", nu=2.5) + \
                   WhiteKernel(noise_level=var_n, noise_level_bounds="fixed")

    gp_point = GaussianProcessRegressor(kernel=point_kernel, optimizer=None)
    gp_point.fit(X_train_s, y_train_s)

    res_pred_point_s = gp_point.predict(X_test_s)
    res_pred_point = scaler_y.inverse_transform(res_pred_point_s.reshape(-1, 1)).ravel()
    w_pred_point = prior_test + res_pred_point
    iv_pred_point = np.sqrt(np.maximum(w_pred_point / X_test[:, 1], 1e-6))
    point_rmse = np.sqrt(np.mean((iv_test - iv_pred_point) ** 2))
    grid_pred_point_s = gp_point.predict(X_grid_s)
    w_grid_point = prior_grid + scaler_y.inverse_transform(grid_pred_point_s.reshape(-1, 1)).ravel()
    iv_grid_point = np.sqrt(np.maximum(w_grid_point / X_grid[:, 1], 1e-6))
    strike_grid_raw = spot * np.exp(K_mesh)
    call_prices_point = utils.black_scholes_call(spot, strike_grid_raw, T_mesh, 0.05, iv_grid_point.reshape(30, 20))
    point_cal_viol = utils.compute_calendar_arbitrage(t_grid_1d, w_grid_point.reshape(30, 20))
    point_bl_viol = utils.compute_butterfly_arbitrage(strike_grid_raw[:, 0], call_prices_point)

    hmc_params_path = os.path.join(hmc_dir, f"hmc_samples_{date_str}.npz")
    if not os.path.exists(hmc_params_path):
        print(f"Skipping {date_str}: HMC samples not found at {hmc_params_path}")
        return None

    hmc_samples = np.load(hmc_params_path)
    ell_k_samples = jnp.array(hmc_samples['ell_k'])
    ell_t_samples = jnp.array(hmc_samples['ell_t'])
    var_f_samples = jnp.exp(jnp.array(hmc_samples['log_sigma_f'])) ** 2
    var_n_samples = jnp.exp(jnp.array(hmc_samples['log_sigma_n'])) ** 2

    # Process posterior samples in mini-batches to prevent GPU VRAM exhaustion; even though this pipeline was executed on
    # an RTX 4090 graphics card, I still had to break it up into smaller batches.
    batch_size = 200
    num_samples = len(ell_k_samples)
    hmc_test_preds_list = []
    hmc_grid_preds_list = []
    X_train_s_jax = jnp.array(X_train_s)
    y_train_s_jax = jnp.array(y_train_s)
    X_test_s_jax = jnp.array(X_test_s)
    X_grid_s_jax = jnp.array(X_grid_s)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        
        ek_b = ell_k_samples[start_idx:end_idx]
        et_b = ell_t_samples[start_idx:end_idx]
        vf_b = var_f_samples[start_idx:end_idx]
        vn_b = var_n_samples[start_idx:end_idx]
        
        test_pred_b = vmap_predict(X_train_s_jax, y_train_s_jax, X_test_s_jax, ek_b, et_b, vf_b, vn_b)
        grid_pred_b = vmap_predict(X_train_s_jax, y_train_s_jax, X_grid_s_jax, ek_b, et_b, vf_b, vn_b)
        
        # offload GPU arrays to CPU immediately
        hmc_test_preds_list.append(np.asarray(test_pred_b))
        hmc_grid_preds_list.append(np.asarray(grid_pred_b))

    hmc_test_preds_s = np.vstack(hmc_test_preds_list)
    hmc_grid_preds_s = np.vstack(hmc_grid_preds_list)

    #compute predictions using CPU array means
    res_pred_hmc = scaler_y.inverse_transform(np.mean(hmc_test_preds_s, axis=0).reshape(-1, 1)).ravel()
    w_pred_hmc = prior_test + res_pred_hmc
    iv_pred_hmc = np.sqrt(np.maximum(w_pred_hmc / X_test[:, 1], 1e-6))
    hmc_rmse = np.sqrt(np.mean((iv_test - iv_pred_hmc) ** 2))

    res_pred_grid_hmc = scaler_y.inverse_transform(np.mean(hmc_grid_preds_s, axis=0).reshape(-1, 1)).ravel()
    w_grid_hmc = prior_grid + res_pred_grid_hmc

    iv_grid_hmc = np.sqrt(np.maximum(w_grid_hmc / X_grid[:, 1], 1e-6))
    call_prices_hmc = utils.black_scholes_call(spot, strike_grid_raw, T_mesh, 0.05, iv_grid_hmc.reshape(30, 20))

    hmc_cal_viol = utils.compute_calendar_arbitrage(t_grid_1d, w_grid_hmc.reshape(30, 20))
    hmc_bl_viol = utils.compute_butterfly_arbitrage(strike_grid_raw[:, 0], call_prices_hmc)

    results = {
        "Date": [date_str, date_str],
        "Method": ["Point-Estimated GP", "Bayesian HMC GP"],
        "OOS_RMSE_IV": [point_rmse, hmc_rmse],
        "Calendar_Violations": [point_cal_viol, hmc_cal_viol],
        "Butterfly_BL_Violations": [point_bl_viol, hmc_bl_viol]
    }

    df = pd.DataFrame(results)
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, f"comparison_{date_str}.csv")
    df.to_csv(out_path, index=False)

    return df


def run_all_comparisons():
    """Finds all processed dates and runs the comparison for each."""
    proc_dir = "data/processed"
    if not os.path.exists(proc_dir):
        print(f"Directory {proc_dir} does not exist.")
        return

    dates = []
    for filename in os.listdir(proc_dir):
        if filename.startswith("design_matrix_") and filename.endswith(".npz"):
            date_str = filename.replace("design_matrix_", "").replace(".npz", "")
            dates.append(date_str)

    dates.sort()

    if not dates:
        print("No processed data files found.")
        return

    print(f"Found {len(dates)} dates to process: {dates}")

    all_results = []
    for date_str in dates:
        try:
            df = compare_methods(date_str)
            if df is not None:
                all_results.append(df)
        except Exception as e:
            print(f"Error processing {date_str}: {e}")

    if all_results:
        master_df = pd.concat(all_results, ignore_index=True)
        master_path = os.path.join("results/tables", "MASTER_COMPARISON_ALL_DATES.csv")
        master_df.to_csv(master_path, index=False)

        print("\n" + "=" * 80)
        print("FINAL DISSERTATION METRICS")
        print("=" * 80)
        print(master_df.to_string(index=False))
        print(f"\nSaved master table to: {master_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=False,
                        help="Specific date to process. If omitted, runs all dates.")
    args = parser.parse_args()

    if args.date:
        df = compare_methods(args.date)
        if df is not None:
            print("\n" + "=" * 60)
            print(f"METRICS FOR {args.date}")
            print("=" * 60)
            print(df.to_string(index=False))
    else:
        run_all_comparisons()