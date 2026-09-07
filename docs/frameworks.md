# Framework adapters

Native frameworks supply gradients; UnifiedIG supplies baseline handling,
path integration, output conventions, and completeness checks. Install the
framework appropriate for your model and device separately, or use the
`torch`, `jax`, or `tensorflow` extras for their standard dependency.

## PyTorch

Pass a `torch.nn.Module` directly. Inputs must be a single tensor per sample,
with one scalar or output vector per sample. UnifiedIG respects the module's
device and floating dtype and restores its prior training/evaluation state.

```{literalinclude} ../examples/pytorch.py
:language: python
:lines: 3-
```

## JAX

Wrap a differentiable prediction function in `uig.JaxModel`. The function is
called with a batch by default. With explicit parameters, it receives
`(params, X)`; otherwise it receives `X`. Use `vectorize=True` for a function
written for a single observation. Flax, NNX, Equinox, and Haiku models can be
exposed through this calling convention; no additional UnifiedIG adapter is
needed for each library.

```{literalinclude} ../examples/jax_model.py
:language: python
:lines: 3-
```

## TensorFlow and Keras

TensorFlow-backed Keras models work directly. Wrap an arbitrary differentiable
TensorFlow function with `uig.TensorFlowModel`. Keras 3 uses its configured
TensorFlow, JAX, or PyTorch backend; configure it before importing Keras.

```{literalinclude} ../examples/tensorflow_keras.py
:language: python
:lines: 3-
```

## Output contract

Supply raw scores for classification. Visible sigmoid/softmax Keras heads are
rejected, but UnifiedIG cannot inspect every transformation hidden in a custom
function. Vector outputs default to class scores; use
`output_kind="regression"` for multiple regression outputs. Multiple input
tensors, dictionaries of outputs, and arbitrary structured outputs are outside
the supported contract. See [output interpretation](explanations.md).

The docs example checker runs framework examples when their dependencies are
installed. Optional-framework CI jobs also check these examples alongside the
backend tests; the documentation build itself needs only core dependencies.
