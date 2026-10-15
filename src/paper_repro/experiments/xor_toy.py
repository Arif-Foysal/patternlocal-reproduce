"""XOR suppressor toy problem from Section 3.2 of the paper.

Generative process (Eq. 8):
  x1 = m1 + d
  x2 = m2 - d
  x3 = d
  y = sign(m1 * m2)

where m1, m2 ~ N(0, 1) and d ~ N(0, sigma_d^2) are independent.
x3 is a suppressor variable.

The smooth classifier: f_tau(x) = tanh(f_tilde(x) / tau)
where f_tilde(x) = (x1 - x3) * (x2 + x3)

PatternLocal should drive the attribution of x3 to ~0.
"""

import numpy as np


def generate_xor_data(n_samples=2500, sigma_d=1.0, seed=42):
    """Generate XOR toy data following Eq. (8)."""
    rng = np.random.default_rng(seed)
    m1 = rng.standard_normal(n_samples)
    m2 = rng.standard_normal(n_samples)
    d = rng.standard_normal(n_samples) * sigma_d

    x1 = m1 + d
    x2 = m2 - d
    x3 = d

    y = np.sign(m1 * m2)
    y[y == 0] = 1

    X = np.column_stack([x1, x2, x3])
    return X, y


def smooth_classifier(X, tau=0.1):
    """Smooth approximation to the optimal XOR classifier."""
    f_tilde = (X[:, 0] - X[:, 2]) * (X[:, 1] + X[:, 2])
    return np.tanh(f_tilde / tau)


def compute_gradient_weights(X, tau=0.1):
    """Compute gradient weights w = grad f_tau(x) following Eq. (9).

    w = c_star * [x2* + x3*, x1* - x3*, x1* - x2* - 2*x3*]
    where c_star = (1/tau) * sech^2(f_tilde(x*) / tau)
    """
    f_tilde = (X[:, 0] - X[:, 2]) * (X[:, 1] + X[:, 2])
    c_star = (1.0 / tau) * (1.0 / np.cosh(f_tilde / tau))**2

    w = np.zeros_like(X)
    w[:, 0] = c_star * (X[:, 1] + X[:, 2])
    w[:, 1] = c_star * (X[:, 0] - X[:, 2])
    w[:, 2] = c_star * (X[:, 0] - X[:, 1] - 2 * X[:, 2])

    return w


def lime_explanation(X_train, y_train, x_star, kernel_width=0.75, n_samples=5000, seed=42):
    """Simple LIME-like explanation for the XOR toy problem.

    Returns weights that represent local linear approximation.
    For the XOR problem, we can use the analytical gradient as a proxy.
    """
    w = compute_gradient_weights(x_star.reshape(1, -1))
    return w.ravel()


def patternlocal_xor(X_train, x_star, lime_weights, bandwidth=None, k_ratio=0.1, lam=1e-5):
    """Apply PatternLocal to the XOR toy problem.

    Returns pattern weights that should suppress x3 attribution.
    Uses global covariance to match theoretical derivation in Section 3.2.
    """
    from xai.patternlocal import patternlocal_explain
    return patternlocal_explain(
        lime_weights=lime_weights,
        lime_intercept=0.0,
        instance=x_star,
        X_train=X_train,
        bandwidth=bandwidth,
        kernel="gaussian",
        k_ratio=k_ratio,
        lam=lam,
        use_global_cov=True,
    )


def run_xor_toy(n_samples=2500, sigma_d=1.0, tau=0.1, seed=42):
    """Run the complete XOR toy experiment.

    Returns:
        results: Dictionary with attributions and metrics
    """
    X, y = generate_xor_data(n_samples, sigma_d, seed)

    rng = np.random.default_rng(seed + 1)
    n_test = min(500, n_samples // 5)
    test_idx = rng.choice(n_samples, n_test, replace=False)
    X_test = X[test_idx]

    W_lime = compute_gradient_weights(X_test, tau)

    lime_suppressor = np.abs(W_lime[:, 2])
    lime_suppressor_mean = lime_suppressor.mean()

    k_ratio = 0.2
    W_pattern = np.zeros_like(X_test)
    for i in range(len(X_test)):
        W_pattern[i] = patternlocal_xor(
            X, X_test[i], W_lime[i],
            k_ratio=k_ratio, lam=1e-5
        )

    pattern_suppressor = np.abs(W_pattern[:, 2])
    pattern_suppressor_mean = pattern_suppressor.mean()

    lime_signal = (np.abs(W_lime[:, 0]) + np.abs(W_lime[:, 1])) / 2
    pattern_signal = (np.abs(W_pattern[:, 0]) + np.abs(W_pattern[:, 1])) / 2

    lime_suppressor_normalized = lime_suppressor / (lime_signal + 1e-10)
    pattern_suppressor_normalized = pattern_suppressor / (pattern_signal + 1e-10)

    return {
        "n_test": n_test,
        "lime_suppressor_mean": float(lime_suppressor_mean),
        "pattern_suppressor_mean": float(pattern_suppressor_mean),
        "lime_suppressor_normalized_mean": float(lime_suppressor_normalized.mean()),
        "pattern_suppressor_normalized_mean": float(pattern_suppressor_normalized.mean()),
        "lime_weights_sample": W_lime[:5].tolist(),
        "pattern_weights_sample": W_pattern[:5].tolist(),
        "suppression_ratio": float(pattern_suppressor_mean / (lime_suppressor_mean + 1e-10)),
    }
