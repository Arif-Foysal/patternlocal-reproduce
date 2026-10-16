"""Analytical PatternLocal: fast, hyperparameter-free local explanations.

A drop-in, O(K * D) replacement for LIME-based PatternLocal. Instead of
fitting a local linear surrogate (LIME) and forming the D x D local
covariance, it uses:

1. the exact input gradient of the predicted-class logit (replaces LIME), and
2. the *kernel-weighted* empirical covariance of the K nearest neighbours
   (replaces the kernel-weighted global covariance of paper Eq. (7)).

The explanation is therefore ``a = C_K w`` with
``C_K = sum_i w_i x~_i x~_i^T`` and ``x~_i = x_i - mu`` the centred
K-neighbourhood, ``w_i`` the (normalized) kernel weights. The matrix-vector
product is evaluated as ``X~^T (w * (X~ w))``, avoiding both the O(D^2)
covariance matrix and any O(D^3) solve.

This is the gradient-surrogate special case of paper Eq. (7): the scalar
denominator ``Var[y~] + lambda`` cancels after unit-mass normalisation, so
only the direction of ``C_K w`` matters.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def kernel_weights(distances, bandwidth, kernel="gaussian"):
    """Compute kernel weights from distances."""
    u = distances / bandwidth
    if kernel == "gaussian":
        return np.exp(-0.5 * u ** 2)
    if kernel == "epanechnikov":
        return np.where(np.abs(u) <= 1, 0.75 * (1.0 - u ** 2), 0.0)
    if kernel == "uniform":
        return np.ones_like(distances)
    raise ValueError(f"Unknown kernel: {kernel}")


class AnalyticalPatternLocalExplainer:
    """Exact-gradient + kernel-weighted KNN-covariance PatternLocal explainer.

    Args:
        model: Differentiable PyTorch model. Moved to ``device`` and set to
            eval mode.
        X_train: Training data, numpy array or torch tensor of shape (N, D).
        k_neighbors: Number of nearest neighbours for the local covariance,
            clamped to ``N``.
        device: ``'cpu'`` or ``'cuda'``.
        normalize: If True, scale each explanation to max-abs 1.
        kernel: Locality kernel ('gaussian', 'epanechnikov', 'uniform').
        bandwidth: Kernel bandwidth. If None, auto-estimated per instance as
            the median distance to its K neighbours.
    """

    def __init__(self, model, X_train, k_neighbors=100, device="cpu",
                 normalize=True, kernel="gaussian", bandwidth=None):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        self.X_train = np.asarray(X_train, dtype=np.float32)
        if self.X_train.ndim != 2:
            raise ValueError("X_train must be 2-D (N, D)")
        self.n_train, self.dim = self.X_train.shape
        self.k_neighbors = int(max(1, min(k_neighbors, self.n_train)))
        self.normalize = normalize
        self.kernel = kernel
        self.bandwidth = bandwidth

        self.knn = NearestNeighbors(
            n_neighbors=self.k_neighbors, metric="euclidean", n_jobs=-1
        )
        self.knn.fit(self.X_train)

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------
    def gradient(self, x_star):
        """Exact gradient of the predicted-class logit at ``x_star``."""
        x = torch.as_tensor(
            x_star, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        x.requires_grad_(True)
        self.model.zero_grad()
        logits = self.model(x)
        if logits.dim() > 1:
            logits = logits[0]
        if logits.numel() > 1:
            target = logits[logits.argmax()]
        else:
            target = logits.reshape(())
        target.backward()
        return x.grad.detach().cpu().numpy().flatten()

    def _bandwidth_for(self, distances):
        if self.bandwidth is not None:
            return float(self.bandwidth)
        bw = float(np.median(distances))
        return bw if bw > 0 else np.finfo(float).eps

    def _attribution(self, w_grad, X_K, weights):
        """a = C_K w = X~^T (w * (X~ w)) / sum(w), X~ centred neighbourhood."""
        weights = np.asarray(weights, dtype=np.float64)
        w_sum = weights.sum()
        if w_sum <= 0:
            raise ValueError("Kernel weights sum to zero. Increase k or bandwidth.")
        weights = weights / w_sum

        local_mean = (weights[:, None] * X_K).sum(axis=0)
        X_tilde = X_K - local_mean
        v = X_tilde @ w_grad
        a = X_tilde.T @ (weights * v)

        if self.normalize:
            max_abs = float(np.max(np.abs(a)))
            if max_abs > 0:
                a = a / max_abs
        return a

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def explain(self, x_star, k=None, bandwidth=None, kernel=None):
        """Return the attribution vector (D,) for a single instance."""
        k = int(k if k is not None else self.k_neighbors)
        k = max(1, min(k, self.n_train))
        bw = self.bandwidth if bandwidth is None else float(bandwidth)
        ker = kernel if kernel is not None else self.kernel

        w_grad = self.gradient(x_star)
        x_np = np.asarray(x_star, dtype=np.float32).reshape(1, -1)
        distances, indices = self.knn.kneighbors(x_np, n_neighbors=k)
        X_K = self.X_train[indices[0]]
        dists = distances[0]
        if bw is None:
            bw = self._bandwidth_for(dists)
        weights = kernel_weights(dists, bw, ker)
        return self._attribution(w_grad, X_K, weights)

    def explain_batch(self, X_batch, k=None, bandwidth=None, kernel=None):
        """Return attributions (B, D) for a batch of instances."""
        X_batch = np.asarray(X_batch, dtype=np.float32)
        k = int(k if k is not None else self.k_neighbors)
        k = max(1, min(k, self.n_train))
        bw = self.bandwidth if bandwidth is None else float(bandwidth)
        ker = kernel if kernel is not None else self.kernel

        distances, indices = self.knn.kneighbors(X_batch, n_neighbors=k)
        out = np.empty((len(X_batch), self.dim), dtype=np.float64)
        for i in range(len(X_batch)):
            w_grad = self.gradient(X_batch[i])
            X_K = self.X_train[indices[i]]
            dists = distances[i]
            if bw is None:
                bw = self._bandwidth_for(dists)
            weights = kernel_weights(dists, bw, ker)
            out[i] = self._attribution(w_grad, X_K, weights)
        return out


def analytical_patternlocal_explain(explainer, x_star, **kwargs):
    """Functional wrapper around :class:`AnalyticalPatternLocalExplainer`."""
    return explainer.explain(x_star, **kwargs)
