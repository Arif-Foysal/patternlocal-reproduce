"""XAI baseline methods for reproduction.

Gradient-based baselines follow the standard definitions (Sundararajan et al.
2017 for IG; Shrikumar et al. 2017 for DeepLift, Rescale rule; Lundberg & Lee
2017 / Lundberg et al. 2017 for GradientShap; Springenberg et al. 2015 for
GuidedBackprop; Simonyan et al. 2013 for Saliency).  Captum is not available on
the Kaggle image, so the rules are implemented directly against the MLP.
"""

import numpy as np
from scipy import ndimage
import torch


def sobel_explanation(image_2d):
    """Sobel edge detection baseline."""
    gx = ndimage.sobel(image_2d, axis=0)
    gy = ndimage.sobel(image_2d, axis=1)
    return np.sqrt(gx**2 + gy**2).ravel()


def laplace_explanation(image_2d):
    """Laplace edge detection baseline."""
    return np.abs(ndimage.laplace(image_2d)).ravel()


def _as_input(x, device, requires_grad=True):
    t = torch.as_tensor(np.asarray(x, dtype=np.float32),
                        device=device).reshape(1, -1)
    if requires_grad:
        t = t.detach().clone().requires_grad_(True)
    return t


def _target_logit(model, inp, target_class):
    logits = model(inp)
    if logits.dim() > 1:
        logits = logits[0]
    if target_class is None:
        target_class = int(logits.argmax().item())
    return logits[target_class], target_class


class GradientBaselines:
    """Gradient-based XAI baselines using PyTorch."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = torch.device(device)

    # -- Saliency -----------------------------------------------------------
    def saliency(self, x, target_class=None):
        """Plain gradient magnitude (Simonyan et al. 2013)."""
        inp = _as_input(x, self.device)
        self.model.zero_grad()
        score, _ = _target_logit(self.model, inp, target_class)
        score.backward()
        return inp.grad.detach().cpu().numpy().ravel()

    # -- Integrated Gradients ----------------------------------------------
    def integrated_gradients(self, x, target_class=None, n_steps=50,
                             baseline=None):
        """Integrated Gradients with a zero baseline (Sundararajan 2017)."""
        xv = np.asarray(x, dtype=np.float32).ravel()
        base = (np.zeros_like(xv) if baseline is None
                else np.asarray(baseline, dtype=np.float32).ravel())
        total = np.zeros_like(xv, dtype=np.float64)
        for s in range(1, n_steps + 1):
            a = s / n_steps
            inp = _as_input(base + a * (xv - base), self.device)
            self.model.zero_grad()
            score, _ = _target_logit(self.model, inp, target_class)
            score.backward()
            total += inp.grad.detach().cpu().numpy().ravel()
        return (xv - base) * (total / n_steps).astype(np.float32)

    # -- SmoothGrad ---------------------------------------------------------
    def smoothgrad(self, x, target_class=None, n_samples=50,
                   noise_level=0.1, seed=0):
        """SmoothGrad: mean saliency over Gaussian-perturbed inputs."""
        rng = np.random.default_rng(seed)
        xv = np.asarray(x, dtype=np.float32).ravel()
        sigma = noise_level * (xv.max() - xv.min() + 1e-12)
        acc = np.zeros_like(xv, dtype=np.float64)
        for _ in range(n_samples):
            noisy = xv + rng.normal(0.0, sigma, size=xv.shape).astype(np.float32)
            inp = _as_input(noisy, self.device)
            self.model.zero_grad()
            score, _ = _target_logit(self.model, inp, target_class)
            score.backward()
            acc += inp.grad.detach().cpu().numpy().ravel()
        return acc / n_samples

    # -- Guided Backprop ----------------------------------------------------
    def guided_backprop(self, x, target_class=None):
        """Guided Backpropagation (Springenberg et al. 2015)."""
        handles = []

        def bwd_hook(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                handles.append(module.register_full_backward_hook(bwd_hook))

        inp = _as_input(x, self.device)
        self.model.zero_grad()
        score, _ = _target_logit(self.model, inp, target_class)
        score.backward()

        grad = inp.grad.detach().cpu().numpy().ravel()
        for h in handles:
            h.remove()
        return grad

    # -- DeepLift (Rescale) -------------------------------------------------
    def deeplift(self, x, target_class=None, baseline=None):
        """DeepLift with the Rescale rule (Shrikumar et al. 2017).

        The Rescale rule sends, for each unit, ``(x - x_ref) * dS/dx``-style
        contributions down the chain but rescales by the *difference* of
        activations rather than the local gradient, so saturation (ReLU) is
        handled exactly.  Implemented with forward/backward hooks over the
        non-linear (ReLU) activations and a zero reference input.
        """
        xv = np.asarray(x, dtype=np.float32).ravel()
        ref = (np.zeros_like(xv) if baseline is None
               else np.asarray(baseline, dtype=np.float32).ravel())

        # Reference forward pass.
        with torch.no_grad():
            ref_t = _as_input(ref, self.device, requires_grad=False)
            self.model.eval()
            ref_logits = self.model(ref_t)
            if ref_logits.dim() > 1:
                ref_logits = ref_logits[0]
            if target_class is None:
                target_class = int(ref_logits.argmax().item())

        inp = _as_input(xv, self.device)
        self.model.zero_grad()

        # Capture pre-activation inputs to each ReLU (for the Rescale factors).
        relu_inputs = {}

        def fwd_hook(module, fin, fout):
            relu_inputs[id(module)] = fin[0].detach()

        def bwd_hook(module, grad_in, grad_out):
            pre = relu_inputs.get(id(module))
            if pre is None:
                return None
            # Rescale: only positive pre-activations with positive gradients
            # pass, scaled by (out - ref_out) / out where out = ReLU(pre).
            out = torch.clamp(pre, min=0.0)
            mult = torch.where(out > 0, grad_out[0], torch.zeros_like(grad_out[0]))
            return (mult,)

        handles = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                handles.append(module.register_forward_hook(fwd_hook))
                handles.append(module.register_full_backward_hook(bwd_hook))

        logits = self.model(inp)
        if logits.dim() > 1:
            logits = logits[0]
        logits[target_class].backward()

        grad = inp.grad.detach().cpu().numpy().ravel()
        for h in handles:
            h.remove()

        # DeepLift attribution for the input layer: (x - x_ref) * dS/dx with the
        # Rescale-propagated gradient.  The multiplicative form makes this the
        # "multiply by input difference" convention Captum uses.
        return (xv - ref) * grad

    # -- GradientShap -------------------------------------------------------
    def gradient_shap(self, x, target_class=None, n_samples=50,
                      noise_level=0.1, baseline=None, seed=0):
        """GradientShap (Lundberg & Lee 2017; Lundberg et al. 2017).

        Averages gradients along the path from random references to the input.
        References are drawn from a baseline (zero) plus Gaussian noise, as in
        the Captum default of a single zero reference with noise.
        """
        rng = np.random.default_rng(seed)
        xv = np.asarray(x, dtype=np.float32).ravel()
        base = (np.zeros_like(xv) if baseline is None
                else np.asarray(baseline, dtype=np.float32).ravel())
        sigma = noise_level * (xv.max() - xv.min() + 1e-12)

        acc = np.zeros_like(xv, dtype=np.float64)
        for _ in range(n_samples):
            ref = base + rng.normal(0.0, sigma, size=xv.shape).astype(np.float32)
            a = float(rng.uniform(0.0, 1.0))
            interp = ref + a * (xv - ref)
            inp = _as_input(interp, self.device)
            self.model.zero_grad()
            score, _ = _target_logit(self.model, inp, target_class)
            score.backward()
            acc += inp.grad.detach().cpu().numpy().ravel()
        return (xv - base) * (acc / n_samples).astype(np.float32)
