import builtins

import numpy as np
import pandas as pd
import pytest

import unifiedig as uig


torch = pytest.importorskip("torch")
pytest.importorskip("captum")


def test_torch_linear_regression_is_complete_and_numpy_backed():
    model = torch.nn.Linear(3, 1, bias=True).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.25], dtype=torch.float64))
    data = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.5, 2.0]], dtype=torch.float64)
    baseline = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

    result = uig.Explainer(model, baseline)(data)

    assert isinstance(result.values, np.ndarray)
    assert result.values.dtype == np.float64
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model(data).detach().numpy()[:, 0],
        rtol=1e-8,
        atol=1e-8,
    )
    assert result.max_abs_completeness_error < 1e-8


def test_torch_nonlinear_binary_logit_is_complete():
    torch.manual_seed(2)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)
    )
    data = torch.tensor([[0.4, -0.2], [-0.5, 0.7]])
    baselines = torch.tensor([[0.0, 0.0], [0.1, -0.1], [-0.2, 0.3]])

    result = uig.Explainer(model, baselines, n_steps=64)(data)

    logits = model(data).detach().numpy()[:, 0]
    expected_base_value = model(baselines).detach().numpy()[:, 0].mean()
    np.testing.assert_allclose(result.base_values, expected_base_value)
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        logits,
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.output_names is None


def test_structured_tensor_input_and_scalar_baseline():
    class ImageSum(torch.nn.Module):
        def forward(self, inputs):
            return inputs.flatten(start_dim=1).sum(dim=1)

    model = ImageSum()
    data = torch.arange(8, dtype=torch.float64).reshape(2, 1, 2, 2)

    result = uig.Explainer(model, 0.0)(data)

    assert result.data.shape == result.values.shape == (2, 1, 2, 2)
    assert result.values.dtype == np.float64
    np.testing.assert_allclose(result.values, data.numpy(), atol=1e-6)
    np.testing.assert_allclose(
        result.values.sum(axis=(1, 2, 3)) + result.base_values,
        data.flatten(start_dim=1).sum(dim=1).numpy(),
        atol=1e-5,
    )


def test_dataframe_input_preserves_feature_names_for_torch():
    model = torch.nn.Linear(2, 1)
    data = pd.DataFrame([[0.5, -0.25]], columns=["age", "income"])

    result = uig.Explainer(model, [0.0, 0.0])(data)

    assert result.feature_names == ["age", "income"]
    assert result.data.shape == result.values.shape == (1, 2)


def test_model_module_training_states_are_restored():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2), torch.nn.Dropout(), torch.nn.Linear(2, 1)
    )
    model.train()
    model[1].eval()
    states_before = [module.training for module in model.modules()]

    uig.Explainer(model, [0.0, 0.0])([[0.5, -0.25]])

    assert [module.training for module in model.modules()] == states_before


def test_multi_output_torch_model_is_rejected():
    model = torch.nn.Linear(2, 3)
    with pytest.raises(ValueError, match="one raw scalar output"):
        uig.Explainer(model, [0.0, 0.0])([[1.0, 2.0]])


def test_missing_captum_error_is_actionable(monkeypatch):
    real_import = builtins.__import__

    def without_captum(name, *args, **kwargs):
        if name.startswith("captum"):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_captum)
    with pytest.raises(ImportError, match=r"unifiedig\[torch\]"):
        uig.Explainer(torch.nn.Linear(1, 1), [0.0])
