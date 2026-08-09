"""
In this file, we define the functions we need for static arbitrage diagnostics on implied volatility surfaces.

Note that, the functions in this module are intended for research code and dissertation experiments,
not production trading systems as they are not yet optimized for live-trading. Lacking performance
tuning and test coverage.

Attend to the main text, page 1 for the Glossary of the Financial Terms.

The conventions used:
----------
S: float
    Spot price of the underlying.
K_grid: ndarray, shape (nK,)
    Sorted strike grid.
T_grid: ndarray, shape (nT,)
    Sorted maturity grid in years.
total_variance_grid: ndarray, shape (nK, nT)
    Total variance surface w(K,T) = sigma_imp(K,T)^2 * T.
call_prices: ndarray, shape (nK, nT)
    European call price surface.

Notes
-----
- Calendar arbitrage is checked by requiring w(K,T) to be non-decreasing in T.
- Butterfly arbitrage is checked by requiring call prices to be convex in K.
- Finite-difference approximations are used, so results are grid-dependent.
"""


import numpy as np

def compute_calendar_arbitrage(T_grid, total_variance_grid):
    """
    This function checks for calendar arbitrage: dw/dT >= 0 as outlined in Section 2.2.1 of the main text. Intuitively, we are testing whether total variance increases with maturity for each strike.
    T_grid should be 1D array of unique maturities and total_variance_grid should be of (len(K_grid), len(T_grid)) shape.
    Caveat: This is not a full proof diagnostic, being a finite-difference, it checks for the condition on the sample grid, but a surface can still have issue between grid points.
    """
    dT = np.diff(T_grid)
    dw = np.diff(total_variance_grid, axis=1)
    derivatives = dw / dT
    violations = np.sum(derivatives < 0)
    return violations


def compute_butterfly_arbitrage(K_grid, call_prices):
    """
    This function checks for the Breeden-Litzenberger (Butterfly) arbitrage, specifically d^2C / dK^2 >= 0. See Section 2.2.2 and Appendix B.4 of the main text.
    K_grid should be 1D array of unique strikes and call_prices should be shape (len(K_grid), len(T_grid)).
    """
    violations = 0
    dK = np.diff(K_grid) #Strike first differences

    for t_idx in range(call_prices.shape[1]): #We take one maturity slice at a time, measuring convexity of call prices. Essentially, we are testing whether the call price decreases with the increasing strike. As the strike moves away from the spot price, the chances of the call landing there become smaller and smaller.
        prices_t = call_prices[:, t_idx]

        # First derivative dC/dK
        dC = np.diff(prices_t)
        first_deriv = dC / dK

        # Second derivative d^2C/dK^2
        dK_mid = K_grid[1:-1]
        d2C = np.diff(first_deriv)
        dK_step = np.diff(dK_mid)
        # Pad dK_step to avoid division by zero issues on irregular grids, or assume uniform. Because this approximation is not very consistent on irregular grids.
        second_deriv = d2C / dK[:-1]

        violations += np.sum(second_deriv < 0)

    return violations


def black_scholes_call(S, K, T, r, sigma):
    """Standard BS formula for translating IV back to prices for arbitrage checks.
    """
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
