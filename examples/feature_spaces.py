"""Choose original, standardized, or PCA feature attributions explicitly."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import unifiedig as uig

rng = np.random.default_rng(4)
X = rng.normal(size=(100, 3)) * [1, 5, 10]
y = X[:, 0] - 0.3 * X[:, 1] + 0.1 * X[:, 2]
model = Pipeline([
    ("scale", StandardScaler()), ("pca", PCA(2)), ("regressor", Ridge())
]).fit(X, y)
baseline = X[10:15]
evaluation = X[:4]

# Both calls accept ORIGINAL observations and ORIGINAL baseline rows.
original = uig.Explainer(model, baseline)(evaluation)
standardized = uig.Explainer(model, baseline, attribute_after="scale")(evaluation)
components = uig.Explainer(model, baseline, attribute_after="pca")(evaluation)

# Featurewise affine scaling changes gradient units but not IG contributions.
np.testing.assert_allclose(original.values, standardized.values, atol=1e-12)
np.testing.assert_allclose(standardized.data, model[:1].transform(evaluation))
assert original.values.shape == (4, 3)
assert components.values.shape == (4, 2)
assert components.feature_names == ["pca0", "pca1"]
for result in (original, standardized, components):
    np.testing.assert_allclose(result.base_values + result.values.sum(axis=1),
                               model.predict(evaluation), atol=1e-12)
    print("After:", result.attribute_after, "features:", result.feature_names,
          "shape:", result.values.shape)
