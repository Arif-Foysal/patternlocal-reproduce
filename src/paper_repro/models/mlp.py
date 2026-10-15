"""MLP model following Appendix D.1 of the paper.

Architecture: 4-layer perceptron with hidden dimensions {128, 64, 32, 16}
ReLU activations, batch normalization after every linear layer, 25% dropout.
"""

import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """MLP classifier for XAI-TRIS benchmark."""

    def __init__(self, input_dim=4096, hidden_dims=(128, 64, 32, 16), num_classes=2, dropout=0.25):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
