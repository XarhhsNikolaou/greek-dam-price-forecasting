"""
conditional_gpd.py
=====================
Πυρήνας του "conditional GPD" -- η scale παράμετρος (σ) της Generalized
Pareto Distribution εξαρτάται από features μέσω log-link:

    σ_i = exp(β0 + β1*x1_i + β2*x2_i + ... )

Η παράμετρος σχήματος (shape, ξ) παραμένει ΣΤΑΘΕΡΗ (όχι εξαρτημένη από
features) -- συνήθης πρακτική στη βιβλιογραφία, αφού είναι δύσκολο να
εκτιμηθεί αξιόπιστα ακόμα και χωρίς εξάρτηση από features.

Fitting: Maximum Likelihood Estimation μέσω scipy.optimize (δεν υπάρχει
έτοιμη sklearn-style συνάρτηση για conditional GPD).

ΣΗΜΑΝΤΙΚΟ: τα features πρέπει να είναι standardized (z-score) πριν το
fitting, αλλιώς η αριθμητική βελτιστοποίηση είναι ασταθής (π.χ. αν ένα
feature έχει τιμές στις χιλιάδες MW ενώ άλλο είναι 0/1, η optimization
"χάνεται").
"""

import numpy as np
from scipy import optimize, stats


def _neg_log_likelihood(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    beta = params[:-1]
    shape = params[-1]
    log_sigma = beta[0] + X @ beta[1:]
    sigma = np.exp(log_sigma)

    if shape != 0:
        z = 1 + shape * y / sigma
        if np.any(z <= 0) or np.any(sigma <= 0):
            return 1e10
        ll = -np.log(sigma) - (1 + 1 / shape) * np.log(z)
    else:
        ll = -np.log(sigma) - y / sigma

    if not np.all(np.isfinite(ll)):
        return 1e10
    return -np.sum(ll)


def fit_conditional_gpd(X: np.ndarray, y: np.ndarray, init_shape: float = 0.2) -> dict:
    """
    X: (n, k) standardized features
    y: (n,) ΘΕΤΙΚΕΣ υπερβάσεις πάνω από το όριο (price - threshold, όλες > 0)

    Επιστρέφει dict με 'beta' (μήκος k+1, πρώτο στοιχείο = intercept),
    'shape', 'success'.
    """
    n_features = X.shape[1]
    init = np.zeros(n_features + 2)
    init[0] = np.log(max(y.mean(), 1e-3))  # λογικό αρχικό σημείο για το intercept
    init[-1] = init_shape

    result = optimize.minimize(
        _neg_log_likelihood, init, args=(X, y),
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6},
    )

    beta = result.x[:-1]
    shape = result.x[-1]

    return {"beta": beta, "shape": shape, "success": result.success, "n_train": len(y)}


def predict_scale(X_new: np.ndarray, beta: np.ndarray) -> np.ndarray:
    log_sigma = beta[0] + X_new @ beta[1:]
    return np.exp(log_sigma)


def predict_point_price(threshold: float, X_new: np.ndarray, beta: np.ndarray, shape: float) -> np.ndarray:
    """Point estimate = threshold + E[GPD(shape, scale)] = threshold + scale/(1-shape).
    Ισχύει μόνο για shape < 1 (πάντα αληθές στην πράξη για τιμές ενέργειας)."""
    sigma = predict_scale(X_new, beta)
    shape_safe = min(shape, 0.99)  # ασφάλεια, το mean απειρίζεται αν shape>=1
    return threshold + sigma / (1 - shape_safe)


if __name__ == "__main__":
    # --- Αυτοέλεγχος σε συνθετικά δεδομένα με γνωστές παραμέτρους ---
    np.random.seed(42)
    n = 3000
    X = np.column_stack([np.random.uniform(0, 1, n), np.random.uniform(-1, 1, n)])
    true_beta = np.array([3.0, 0.8, -0.5])
    true_shape = 0.25

    true_sigma = np.exp(true_beta[0] + X @ true_beta[1:])
    y = np.array([stats.genpareto.rvs(c=true_shape, scale=s) for s in true_sigma])

    fit = fit_conditional_gpd(X, y)
    print("Αληθινές παράμετροι (β0,β1,β2,ξ):", true_beta, true_shape)
    print("Εκτιμημένες:                     ", np.round(np.append(fit["beta"], fit["shape"]), 3))
    print("Επιτυχία:", fit["success"])