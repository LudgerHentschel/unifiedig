"""Explain a raw scalar output from a PyTorch module through native autograd."""

import numpy as np
import torch

from cbaseline import background
import unifiedig as uig

torch.manual_seed(6)
model = torch.nn.Sequential(
    torch.nn.Linear(3, 8),
    torch.nn.Tanh(),
    torch.nn.Linear(8, 1),
)
data = torch.tensor([[0.5, -0.2, 0.8]])
reference = torch.randn(200, 3)
with torch.no_grad():
    reference_predictions = model(reference).numpy()[:, 0]
f0 = float(reference_predictions.mean())
bg = background(
    predictions=reference_predictions,
    f0=f0,
    features=reference.numpy(),
    weighting="calibrated",
)

# A deliberate single starting point is also valid: uig.Explainer(model, x0).
explanation = uig.Explainer(model, bg)(data)
print(explanation.values)
print(explanation.max_abs_completeness_error)
