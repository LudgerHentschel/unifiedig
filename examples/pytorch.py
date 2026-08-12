"""Explain a raw scalar output from a PyTorch module through native autograd."""

import torch

import unifiedig as uig

model = torch.nn.Sequential(
    torch.nn.Linear(3, 8),
    torch.nn.Tanh(),
    torch.nn.Linear(8, 1),
)
data = torch.tensor([[0.5, -0.2, 0.8]])

explanation = uig.Explainer(model, baseline=torch.zeros(3), n_steps=64)(data)
print(explanation.values)
print(explanation.max_abs_completeness_error)
