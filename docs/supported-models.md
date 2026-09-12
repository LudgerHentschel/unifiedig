---
myst:
  html_meta:
    description: "Check UnifiedIG model support, exact and numerical backends, optional framework installations, and classification output restrictions."
---

# Supported models and backend selection

This matrix describes the release dependency series: skgrad 0.1.5+, TreeIG
0.2.x, and CBaseline 0.1.2+. All three install with UnifiedIG. Frameworks in
the last column are separate optional installations. All models must be fitted
or otherwise ready for inference. Inputs must permit numeric interpolation.

| Models / interface | Backend and integration | Explained output / limits | Extra installation |
|---|---|---|---|
| `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LinearSVR` | skgrad, exact affine | Regression predictions, including supported multi-output estimators | None |
| `LogisticRegression`, `RidgeClassifier`, `LinearSVC` | skgrad, exact affine | Binary decision margin; multiclass centered decision scores | None |
| `SVR`, `NuSVR`, binary `SVC`, `NuSVC` | skgrad, analytic derivatives and quadrature; linear kernels exact | Built-in linear, polynomial, RBF, sigmoid kernels; no multiclass kernel SVM or precomputed/callable kernels | None |
| `MLPRegressor`, `MLPClassifier` | skgrad and quadrature | Identity regression output; binary logits / centered multiclass logits; identity, logistic, tanh, ReLU hidden activations | None |
| sklearn `Pipeline` ending in a skgrad-supported estimator | skgrad chain rule; exact affine/polynomial shortcuts when applicable | StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler, PCA, PolynomialFeatures, supported fitted feature selectors, nested pipelines; no arbitrary transformer derivatives | None |
| `DecisionTreeRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor` | TreeIG, exact split crossings | Scalar regression prediction; numeric splits | None |
| `GradientBoostingClassifier` | TreeIG, exact split crossings | Binary margin / centered multiclass margins | None |
| `XGBRegressor`, `XGBClassifier`, native XGBoost `Booster` | TreeIG, exact split crossings | Supported numeric tree objectives; regression output or raw class margins; TreeIG parser restrictions apply | xgboost |
| `LGBMRegressor`, `LGBMClassifier`, native LightGBM `Booster` | TreeIG, exact split crossings | Supported numeric tree objectives; regression output or raw class margins; TreeIG parser restrictions apply | lightgbm |
| `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier`; numeric CatBoost models | TreeIG numerical event detector, explicit `fallback="tree_numeric"` | Prediction / raw class scores; approximate crossing detection | catboost for CatBoost |
| `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier` | TreeIG numerical event detector, explicit `fallback="tree_numeric"` | Log odds / centered log probabilities of the aggregated model; explicit `probability_floor` required for zero probabilities | None |
| `torch.nn.Module` | Native PyTorch autograd and quadrature | Scalar, raw class scores, or declared multi-output regression; one input tensor per sample | torch |
| `uig.JaxModel(function, ...)` | Native JAX autodiff and quadrature | Differentiable batched function; optional explicit parameters | jax |
| `uig.TensorFlowModel(function, ...)`, TensorFlow-backed Keras | Native TensorFlow gradients and quadrature | Differentiable model output | tensorflow |
| Keras 3 model | Native configured TensorFlow / JAX / PyTorch backend | Visible sigmoid/softmax classification heads rejected; return logits | keras and its selected backend |
| Other smooth sklearn regressors / decision-score classifiers | Central finite differences and quadrature, explicit `fallback="finite_difference"` | Requires predict / decision_function; no probability-only classifiers, multiclass pairwise SVM scores, or tree fallback | Model's own dependencies |

Specialized backends take precedence over an explicitly requested fallback.
Unsupported models raise an error; fallback is never silently enabled. TreeIG
support depends on objective, split type, and model configuration, not only the
Python class. Native categorical inputs and arbitrary discontinuous callables
are outside the supported interpolation contract.

For framework vectors, classification is the default: two scores become
`score[1] - score[0]`, while three or more are centered across classes. Use
`output_kind="regression"` for multi-output regression. A scalar framework
output is explained as supplied; callers must supply a score rather than a
probability for classification. UnifiedIG cannot detect every probability
transformation hidden inside arbitrary user functions.

A baseline vector is one point; a matrix is a shared distribution, even when
its row count equals the input batch size. CBaseline objects supply rows and
weights directly. Completeness reconstructs the explained output from the
weighted mean baseline output plus summed attributions. Numerical completeness
does not guarantee accurate individual allocations; refine quadrature or tree
resolution when needed. See [semantics](semantics.md) for shapes and loss output
conventions, and [feature spaces](feature-spaces.md) for `attribute_after`.
