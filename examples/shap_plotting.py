"""Plot UnifiedIG attributions with SHAP; optionally save a gallery."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from cbaseline import background
from sklearn.linear_model import Ridge

import unifiedig as uig

# Start explanation
rng = np.random.default_rng(17)
X = rng.normal(size=(180, 4))
y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]
model = Ridge(alpha=0.5).fit(X[:120], y[:120])
reference_predictions = model.predict(X[:120])
bg = background(
    predictions=reference_predictions,
    f0=float(reference_predictions.mean()),
    features=X[:120],
    weighting="calibrated",
)
result = uig.Explainer(model, bg)(X[120:])
plot_values = result.to_shap()
plot_values.feature_names = ["Feature A", "Feature B", "Feature C", "Feature D"]
# End explanation

np.testing.assert_allclose(plot_values.values, result.values)
np.testing.assert_allclose(
    plot_values.base_values + plot_values.values.sum(axis=1),
    model.predict(X[120:]),
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()
if args.output_dir:
    args.output_dir.mkdir(parents=True, exist_ok=True)


def finish(name):
    if args.output_dir:
        plt.savefig(args.output_dir / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close("all")


shap.plots.waterfall(plot_values[0], show=False)
plt.title("One prediction: Integrated Gradients contributions")
finish("waterfall")

shap.plots.beeswarm(plot_values, show=False)
plt.xlabel("Integrated Gradients contribution")
finish("beeswarm")

shap.plots.bar(plot_values, show=False)
plt.xlabel("Mean absolute Integrated Gradients contribution")
finish("bar")

shap.plots.scatter(plot_values[:, "Feature A"], show=False)
plt.ylabel("Integrated Gradients contribution for Feature A")
finish("scatter")
