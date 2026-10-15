"""XAI baseline methods for reproduction."""

import numpy as np
from scipy import ndimage


def sobel_explanation(image_2d):
    """Sobel edge detection baseline."""
    gx = ndimage.sobel(image_2d, axis=0)
    gy = ndimage.sobel(image_2d, axis=1)
    return np.sqrt(gx**2 + gy**2).ravel()


def laplace_explanation(image_2d):
    """Laplace edge detection baseline."""
    return np.abs(ndimage.laplace(image_2d)).ravel()


class GradientBaselines:
    """Gradient-based XAI baselines using PyTorch."""

    def __init__(self, model):
        self.model = model

    def saliency(self, x, target_class=None):
        """Plain gradient saliency map."""
        import torch
        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        x_tensor.requires_grad_(True)
        self.model.zero_grad()
        output = self.model(x_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        output[0, target_class].backward()
        return x_tensor.grad.data.abs().squeeze(0).numpy().ravel()

    def integrated_gradients(self, x, target_class=None, n_steps=50):
        """Integrated Gradients."""
        import torch
        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        baseline = torch.zeros_like(x_tensor)
        total_grad = torch.zeros_like(x_tensor)
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            interpolated = baseline + alpha * (x_tensor - baseline)
            interpolated.requires_grad_(True)
            self.model.zero_grad()
            output = self.model(interpolated)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            output[0, target_class].backward()
            total_grad += interpolated.grad.data
        avg_grad = total_grad / n_steps
        ig = (x_tensor - baseline) * avg_grad
        return ig.abs().squeeze(0).numpy().ravel()

    def guided_backprop(self, x, target_class=None):
        """Guided Backpropagation."""
        import torch
        activations = {}

        def forward_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[id(output)] = output

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output[0], torch.Tensor):
                grad_output = (torch.clamp(grad_output[0], min=0),)
            return grad_output

        hooks = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                hooks.append(module.register_forward_hook(forward_hook))
                hooks.append(module.register_full_backward_hook(backward_hook))

        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        x_tensor.requires_grad_(True)
        self.model.zero_grad()
        output = self.model(x_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        output[0, target_class].backward()

        grad = x_tensor.grad.data.squeeze(0).numpy()
        for hook in hooks:
            hook.remove()
        return np.abs(grad.ravel())
