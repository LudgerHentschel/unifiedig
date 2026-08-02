"""Compare analytic-gradient IG with feature-wise finite differences."""

import argparse
import time

import numpy as np
from sklearn.neural_network import MLPRegressor

import unifiedig as uig


def finite_difference_ig(model, data, baseline, *, n_steps, epsilon=1e-5):
    difference = data - baseline
    gradients = np.zeros_like(data)
    nodes, weights = np.polynomial.legendre.leggauss(n_steps)
    for alpha, quadrature_weight in zip((nodes + 1.0) / 2.0, weights / 2.0):
        path = baseline + alpha * difference
        for feature in range(data.shape[1]):
            offset = np.zeros_like(data)
            offset[:, feature] = epsilon
            gradients[:, feature] += quadrature_weight * (
                model.predict(path + offset) - model.predict(path - offset)
            ) / (2.0 * epsilon)
    return difference * gradients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    model = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=0)
    layer_sizes = [args.features, 64, 32, 1]
    model.coefs_ = [
        rng.normal(scale=0.5 / np.sqrt(input_size), size=(input_size, output_size))
        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:])
    ]
    model.intercepts_ = [np.zeros(size) for size in layer_sizes[1:]]
    model.n_features_in_ = args.features
    model.n_layers_ = len(layer_sizes)
    model.n_outputs_ = 1
    model.out_activation_ = "identity"
    data = rng.normal(size=(args.samples, args.features))
    baseline = np.zeros_like(data)

    # Some NumPy/BLAS combinations emit spurious matmul floating-point warnings
    # while still returning finite results; validate the results explicitly.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        start = time.perf_counter()
        analytic = uig.Explainer(
            model, baseline, n_steps=args.steps, check_completeness=False
        )(data).values
        analytic_seconds = time.perf_counter() - start

        start = time.perf_counter()
        finite = finite_difference_ig(model, data, baseline, n_steps=args.steps)
        finite_seconds = time.perf_counter() - start

    if not np.isfinite(analytic).all() or not np.isfinite(finite).all():
        raise RuntimeError("benchmark produced non-finite attributions")

    print(f"analytic gradients: {analytic_seconds:.3f}s")
    print(f"finite differences: {finite_seconds:.3f}s")
    print(f"speedup: {finite_seconds / analytic_seconds:.1f}x")
    print(f"maximum attribution difference: {np.max(np.abs(analytic - finite)):.3g}")


if __name__ == "__main__":
    main()
