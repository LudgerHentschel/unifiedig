"""Explain a raw scalar output from a PyTorch module through native autograd."""

import torch

import unifiedig as uig

model = torch.nn.Sequential(
    torch.nn.Linear(3, 8),
    torch.nn.Tanh(),
    torch.nn.Linear(8, 1),
)
data = torch.tensor([[0.5, -0.2, 0.8]])
background = torch.tensor(
    [[-0.4, 0.1, 0.3], [0.2, -0.3, 0.6], [0.1, 0.4, -0.2]]
)

explanation = uig.Explainer(model, background)(data)
print(explanation.values)
print(explanation.max_abs_completeness_error)
