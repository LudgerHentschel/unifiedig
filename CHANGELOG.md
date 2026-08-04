# Changelog

## 0.1.1.dev0

- Delegate analytic scikit-learn MLP values and input Jacobians to skgrad.
- Recognize every smooth estimator supported by skgrad through one capability
  check, including Ridge, Lasso, ElasticNet, and RidgeClassifier.
- Document the complete supported-model inventory and output scales.
- Add an opt-in, batched finite-difference fallback for other smooth sklearn
  regressors and binary decision-score classifiers.
- Reject known discontinuous estimator families from the numerical fallback.
- Support explicit weighted baseline distributions across every backend and
  consume CBaseline `Background` objects directly through their public
  `rows` and `weights` properties.
- Attribute multiclass models as one centered, zero-sum decision-score vector
  across skgrad, TreeIG, PyTorch, and numerical backends.
- Add `Explanation.contrast()` for derived pairwise score-margin attribution.
- Add optional JAX automatic-gradient attribution through the explicit
  `JaxModel` adapter, including parameter pytrees, closures, single-sample
  vectorization, weighted backgrounds, and centered multiclass scores.
- Add a JAX example, optional dependency extra, and dedicated CI job.
- Add an explicit `fallback="tree_numeric"` route for CatBoost and recognized
  piecewise-constant tree models through TreeIGNumeric, with weighted
  backgrounds, multiclass raw scores, configurable path grids, and CatBoost CI.
- Expand the README with computation routes, model coverage, baseline guidance,
  SHAP plotting, numerical diagnostics, and JAX usage.

## 0.1.0

- Stabilize the `Explainer` and `Explanation` V1 API.
- Define baseline matrices as equally weighted distributions shared by every input.
- Provide exact TreeIG, closed-form linear, analytic sklearn MLP, and optional
  PyTorch/Captum backends behind one dispatch interface.
- Improve optional-dependency errors, package metadata, and installed type information.

## 0.1.0.dev3

- Add an optional exact tree-model backend powered entirely by TreeIG.
- Treat baseline matrices as shared distributions for every model family.
- Keep binary tree classification attribution on the raw decision-score scale.

## 0.1.0.dev2

- Add optional PyTorch support through Captum Integrated Gradients.
- Preserve model device, floating-point dtype, and training/evaluation state.
- Support tabular and structured single-tensor inputs with scalar outputs.
- Keep PyTorch and Captum out of the core installation.

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
