"""Weighted local-linear LIME surrogate (paper Eq. 5).

LIME replaces the model locally with a linear surrogate fitted to perturbed
samples around the instance. With the identity representation the perturbation
is a binary mask ``z`` and the reconstruction is ``h^-1(z) = x * z + r * (1-z)``;
samples are weighted by the locality kernel ``exp(-||1 - z||^2 / bandwidth^2)``.
"""

import numpy as np
import torch


def model_logits(model, X, batch_size=256, device="cpu"):
    """Evaluate a PyTorch model and return logits/probabilities as numpy."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(
                X[i:i + batch_size], dtype=torch.float32, device=device
            )
            outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def lime_explain(model, x, n_samples=2000, bandwidth=None, lam=1.0, seed=0,
                 reference=None, device="cpu"):
    """Fit a weighted local-linear LIME surrogate around ``x``.

    Args:
        model: PyTorch model returning logits of shape (batch, C).
        x: Instance to explain, shape (D,).
        n_samples: Number of perturbed samples.
        bandwidth: Locality-kernel width. Defaults to ``0.75 * sqrt(D)``.
        lam: Ridge strength.
        seed: RNG seed for the perturbation masks.
        reference: Reference value ``r`` for ``h^-1`` (default 0).
        device: Torch device.

    Returns:
        (weights, intercept): The surrogate coefficients (D,) and bias.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32).ravel()
    d = x.shape[0]
    if bandwidth is None:
        bandwidth = 0.75 * np.sqrt(d)
    if reference is None:
        reference = np.zeros(d, dtype=np.float32)
    else:
        reference = np.asarray(reference, dtype=np.float32).ravel()

    z = (rng.random((n_samples, d)) < 0.5).astype(np.float64)
    Xp = x[None, :] * z + reference[None, :] * (1.0 - z)

    logits = model_logits(model, Xp, device=device)
    pred = logits.argmax(axis=1)
    y = logits[np.arange(n_samples), pred]

    dist2 = ((1.0 - z) ** 2).sum(axis=1)
    w = np.exp(-dist2 / (bandwidth ** 2))
    W = w / w.sum()

    zbar = W @ z
    ybar = W @ y
    zc = z - zbar
    yc = y - ybar

    # Weighted ridge via the Woodbury identity (n x n instead of D x D),
    # using the symmetric form with float32 + Cholesky for speed/stability:
    #   (Zc^T W Zc + lam I)^-1 Zc^T W yc
    #     = (S Zc)^T (lam I + (S Zc)(S Zc)^T)^-1 (S yc),  S = sqrt(W)
    import scipy.linalg as sla

    zc32 = zc.astype(np.float32)
    s32 = np.sqrt(W).astype(np.float32)
    SZc = zc32 * s32[:, None]
    G = SZc @ SZc.T + np.float32(lam) * np.eye(n_samples, dtype=np.float32)
    c, low = sla.cho_factor(G)
    alpha = sla.cho_solve((c, low), s32 * yc.astype(np.float32))
    v = (SZc.T @ alpha).astype(np.float64)
    b = ybar - v @ zbar
    return v, b
