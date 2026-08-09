import importlib
import json
import os
import argparse
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler


def matern52_kernel(X, Z, lengthscales, variance):
    """
    Computes the Matérn 5/2 kernel covariance matrix using JAX operations.
    """
    X_scaled = X / lengthscales
    Z_scaled = Z / lengthscales
    dist_sq = jnp.sum(X_scaled ** 2, axis=-1)[:, None] + jnp.sum(Z_scaled ** 2, axis=-1) - 2 * jnp.dot(X_scaled,
                                                                                                       Z_scaled.T)
    dist_sq = jnp.maximum(dist_sq, 0.0)
    d = jnp.sqrt(dist_sq + 1e-12)
    return variance * (1.0 + jnp.sqrt(5.0) * d + (5.0 / 3.0) * dist_sq) * jnp.exp(-jnp.sqrt(5.0) * d)


def plot_topology(date_str, diag_dir="results/diagnostics", out_dir="results/figures"):
    print(f"Generating topology plot for {date_str}...")
    data = np.load(os.path.join(diag_dir, f"topology_{date_str}.npz"))
    LML = data['LML']
    ell_k = data['ell_k']
    ell_t = data['ell_t']

    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(ell_k, ell_t)
    cp = plt.contourf(X, Y, LML.T, levels=50, cmap='viridis')
    plt.contour(X, Y, LML.T, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(cp, label="Log Marginal Likelihood")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Strike Length-Scale $\ell_K$')
    plt.ylabel(r'Maturity Length-Scale $\ell_T$')
    plt.title(f"LML Topology - {date_str}")

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"topology_{date_str}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_hmc_posteriors(date_str, diag_dir="results/diagnostics", out_dir="results/figures"):
    print(f"Generating HMC posterior plots for {date_str}...")
    data = np.load(os.path.join(diag_dir, f"hmc_samples_{date_str}.npz"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(data['ell_k'], bins=30, density=True, alpha=0.7, color='steelblue')
    axes[0].set_title(r'Posterior $\ell_K$')
    axes[0].set_xlabel("Value")

    axes[1].hist(data['ell_t'], bins=30, density=True, alpha=0.7, color='darkorange')
    axes[1].set_title(r'Posterior $\ell_T$')
    axes[1].set_xlabel("Value")

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"hmc_posterior_{date_str}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def generate_3d_surfaces(date_str, proc_dir="data/processed", prior_dir="results/surfaces",
                         point_dir="results/surfaces", hmc_dir="results/diagnostics", out_dir="results/figures"):
    """
    Generates a side-by-side 3D plot of the Implied Volatility surfaces: Point-Estimated GP vs. Bayesian HMC GP.
    """
    print(f"Generating 3D surfaces for {date_str}...")
    data = np.load(os.path.join(proc_dir, f"design_matrix_{date_str}.npz"))
    X_train = data['X_raw'][data['train_idx']]

    k_min, k_max = np.min(X_train[:, 0]), np.max(X_train[:, 0])
    t_min, t_max = np.min(X_train[:, 1]), np.max(X_train[:, 1])

    k_grid = np.linspace(k_min, k_max, 50)
    t_grid = np.linspace(max(0.01, t_min), t_max, 50)
    K_mesh, T_mesh = np.meshgrid(k_grid, t_grid)
    X_plot = np.vstack([K_mesh.ravel(), T_mesh.ravel()]).T

    #scalers
    scaler_X = StandardScaler().fit(X_train)
    X_plot_s = jnp.array(scaler_X.transform(X_plot))

    #prior model
    prior_data = np.load(os.path.join(prior_dir, f"prior_fit_{date_str}.npz"))
    prior_model = importlib.import_module("03_prior_model").TotalVariancePrior()
    prior_model.params = prior_data['params']
    prior_pred_w = prior_model.predict(X_plot)

    #compute point-estimated predictions
    json_path = os.path.join(point_dir, f"point_gp_{date_str}.json")
    with open(json_path, "r") as f:
        point_data = json.load(f)

    if 'ell_k' in point_data:
        ell_point = jnp.array([point_data['ell_k'], point_data['ell_t']])
        var_f_point = float(point_data.get('variance', point_data.get('variance_f', 1.0)))
        var_n_point = float(point_data.get('noise', point_data.get('variance_n', 1e-5)))
    elif 'kernel_params' in point_data:
        k_params = point_data['kernel_params']
        ell_point = jnp.array(k_params['k1__k2__length_scale'])
        var_f_point = float(k_params['k1__k1__constant_value'])
        var_n_point = float(k_params['k2__noise_level'])
    else:
        raise KeyError(f"Could not find parameter keys in {json_path}. Please check the JSON structure.")

    #we need the training residuals to make predictions
    w_train = data['w_raw'][data['train_idx']]
    res_train = w_train - prior_model.predict(X_train)
    scaler_y = StandardScaler().fit(res_train.reshape(-1, 1))
    X_train_s = jnp.array(scaler_X.transform(X_train))
    y_train_s = jnp.array(scaler_y.transform(res_train.reshape(-1, 1)).ravel())

    K_train_point = matern52_kernel(X_train_s, X_train_s, ell_point, var_f_point) + (var_n_point + 1e-6) * jnp.eye(
        len(X_train_s))
    K_cross_point = matern52_kernel(X_plot_s, X_train_s, ell_point, var_f_point)

    inv_Ky_point = jnp.linalg.solve(K_train_point, y_train_s)
    y_pred_s_point = K_cross_point @ inv_Ky_point
    res_pred_point = scaler_y.inverse_transform(np.array(y_pred_s_point).reshape(-1, 1)).ravel()

    #convert w back to IV
    w_pred_point = prior_pred_w + res_pred_point
    iv_pred_point = np.sqrt(np.maximum(w_pred_point / X_plot[:, 1], 1e-6))
    IV_mesh_point = iv_pred_point.reshape(50, 50)

    #compute HMC predictions
    hmc_data = np.load(os.path.join(hmc_dir, f"hmc_samples_{date_str}.npz"))
    ell_k_samples = hmc_data['ell_k']
    ell_t_samples = hmc_data['ell_t']
    log_sig_f_samples = hmc_data['log_sigma_f']
    log_sig_n_samples = hmc_data['log_sigma_n']

    #subsample to save time
    num_samples = len(ell_k_samples)
    idx_sub = np.linspace(0, num_samples - 1, 50, dtype=int)

    y_pred_s_hmc_accum = np.zeros(len(X_plot_s))

    for i in idx_sub:
        ell_hmc = jnp.array([ell_k_samples[i], ell_t_samples[i]])
        var_f_hmc = jnp.exp(log_sig_f_samples[i]) ** 2
        var_n_hmc = jnp.exp(log_sig_n_samples[i]) ** 2
        K_train_hmc = matern52_kernel(X_train_s, X_train_s, ell_hmc, var_f_hmc) + (var_n_hmc + 1e-6) * jnp.eye(
            len(X_train_s))
        K_cross_hmc = matern52_kernel(X_plot_s, X_train_s, ell_hmc, var_f_hmc)
        inv_Ky_hmc = jnp.linalg.solve(K_train_hmc, y_train_s)
        y_pred_s_hmc_accum += K_cross_hmc @ inv_Ky_hmc

    y_pred_s_hmc = y_pred_s_hmc_accum / len(idx_sub)
    res_pred_hmc = scaler_y.inverse_transform(np.array(y_pred_s_hmc).reshape(-1, 1)).ravel()

    w_pred_hmc = prior_pred_w + res_pred_hmc
    iv_pred_hmc = np.sqrt(np.maximum(w_pred_hmc / X_plot[:, 1], 1e-6))
    IV_mesh_hmc = iv_pred_hmc.reshape(50, 50)

    #plotting
    fig = plt.figure(figsize=(16, 7))

    #surface 1: point-estimated
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(K_mesh, T_mesh, IV_mesh_point, cmap=cm.viridis, edgecolor='none', alpha=0.9)
    ax1.set_title(f"Point-Estimated GP Surface\n{date_str}")
    ax1.set_xlabel("Moneyness (K)")
    ax1.set_ylabel("Maturity (T)")
    ax1.set_zlabel("Implied Volatility")
    ax1.view_init(elev=25, azim=-135)

    #surface 2: bayesian HMC
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(K_mesh, T_mesh, IV_mesh_hmc, cmap=cm.plasma, edgecolor='none', alpha=0.9)
    ax2.set_title(f"Bayesian HMC GP Surface\n{date_str}")
    ax2.set_xlabel("Moneyness (K)")
    ax2.set_ylabel("Maturity (T)")
    ax2.set_zlabel("Implied Volatility")
    ax2.view_init(elev=25, azim=-135)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"3D_surface_comparison_{date_str}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D surface plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    args = parser.parse_args()

    #call all three plotting functions
    plot_topology(args.date)
    plot_hmc_posteriors(args.date)
    generate_3d_surfaces(args.date)

    print("All figures generated and saved to results/figures/")