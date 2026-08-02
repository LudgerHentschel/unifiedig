# Changelog

## 0.1.0.dev1

- Add completeness-error diagnostics and numerical-accuracy warnings.
- Test every sklearn MLP hidden activation and per-sample baselines.
- Verify compatibility with SHAP waterfall and beeswarm plots.
- Add model-specific examples and an MLP gradient benchmark.
- Harden releases with version-tag validation and installed-wheel checks.

## 0.1.0.dev0

- Establish the `Explainer` and `Explanation` public API.
- Add closed-form sklearn linear and binary logistic regression backends.
- Add analytic-gradient sklearn MLP regression and binary classification.
- Add optional conversion to `shap.Explanation`.
