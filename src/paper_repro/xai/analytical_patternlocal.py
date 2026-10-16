"""Analytical PatternLocal: fast, hyperparameter-free local explanations.

A drop-in, O(K * D) replacement for LIME-based PatternLocal. Instead of
fitting a local linear surrogate (LIME) and forming the D x D local
covariance, it uses:

1. the exact input gradient of the predicted-class logit (replaces LIME), and
2. the empirical covariance of the K nearest neighbours (replaces the
   kernel-weighted global covariance).

The explanation is therefore ``a = C_K w`` with
``C_K = (1/K) X~^T X~`` and ``X~`` the centred K-neighbourhood. The
matrix-vector product is evaluated as ``X~^T (X~ w) / K``, avoiding both the
O(D^2) covariance matrix and any O(D^3) solve.

This is the gradient-surrogate special case of paper Eq. (7): the scalar
denominator ``Var[y~] + lambda`` cancels after unit-mass normalisation, so
only the direction of ``C_K w`` matters.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


class AnalyticalPatternLocalExplainer:
    """Exact-gradient + KNN-covariance PatternLocal explainer.

    Args:
        model: Differentiable PyTorch model. It is moved to ``device`` and set
            to eval mode.
        X_train: Training data, numpy array or torch tensor of shape (N, D).
        k_neighbors: Number of nearest neighbours used for the local
            covariance. Clamped to ``N``.
        device: ``'cpu'`` or ``'cuda'``.
        normalize: If True, scale each explanation to max-abs 1.
    """

    def __init__(self, model, X_train, k_neighbors=100, device="cpu",
                 normalize=True):
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

        self.knn = NearestNeighbors(
            n_neighbors=self.k_neighbors, metric="euclidean", n_jobs=-1
        )
        self.knn.fit(self.X_train)

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------
    def _gradient(self, x_star):
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

    def gradient(self, x_star):
        """Public wrapper around :meth:`_gradient`."""
        return self._gradient(x_star)

    def _attribution(self, w_grad, X_K):
        """a = C_K w = X~^T (X~ w) / K, with X~ the centred neighbourhood."""
        local_mean = X_K.mean(axis=0)
        X_tilde = X_K - local_mean
        v = X_tilde @ w_grad
        a = (X_tilde.T @ v) / X_K.shape[0]
        if self.normalize:
            max_abs = float(np.max(np.abs(a)))
            if max_abs > 0:
                a = a / max_abs
        return a

    def attribution(self, w_grad, X_K):
        """Public wrapper around :meth:`_attribution`."""
        return self._attribution(w_grad, X_K)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def explain(self, x_star):
        """Return the attribution vector (D,) for a single instance."""
        w_grad = self._gradient(x_star)
        x_np = np.asarray(x_star, dtype=np.float32).reshape(1, -1)
        _, indices = self.knn.kneighbors(x_np)
        return self._attribution(w_grad, self.X_train[indices[0]])

    def explain_batch(self, X_batch):
        """Return attributions (B, D) for a batch of instances."""
        X_batch = np.asarray(X_batch, dtype=np.float32)
        _, indices = self.knn.kneighbors(X_batch)
        out = np.empty((len(X_batch), self.dim), dtype=np.float64)
        for i in range(len(X_batch)):
            w_grad = self._gradient(X_batch[i])
            out[i] = self._attribution(w_grad, self.X_train[indices[i]])
        return out


def analytical_patternlocal_explain(explainer, x_star):
    """Functional wrapper around :class:`AnalyticalPatternLocalExplainer`."""
    return explainer.explain(x_star)
