"""Execute the examples embedded in the documentation, without a plotting UI."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = (
    "quickstart.py", "logistic_regression.py", "multiclass_classification.py",
    "feature_spaces.py", "loss_attribution.py", "loss_classification.py",
    "numerical_fallback.py",
)
OPTIONAL = {"shap": "shap_plotting.py", "torch": "pytorch.py", "jax": "jax_model.py", "tensorflow": "tensorflow_keras.py"}


def main():
    examples = list(CORE)
    for dependency, example in OPTIONAL.items():
        if importlib.util.find_spec(dependency) is not None:
            examples.append(example)
        else:
            print(f"SKIP {example}: optional {dependency} is not installed", flush=True)
    env = dict(os.environ, MPLBACKEND="Agg")
    for example in examples:
        print(f"CHECK {example}", flush=True)
        result = subprocess.run(
            [sys.executable, str(ROOT / "examples" / example)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
        )
        if result.returncode:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)
    print(f"Passed {len(examples)} documentation examples.")


if __name__ == "__main__":
    main()
