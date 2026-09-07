# Public release checklist

## Current prerequisites

- Publish skgrad 0.1.5 or a compatible newer 0.1.x version before UnifiedIG.
  The feature-space API requires `pipeline_view`; do not lower the requirement
  to bypass installation failures. PyPI had only skgrad 0.1.1 at the 2026-09-07 audit.
- Run CI against published dependencies. Local sibling package validation is
  useful but does not demonstrate a successful public one-install workflow.
- Replace `0.1.1.dev1` with the intended release version in `pyproject.toml`,
  finalize the Unreleased changelog, and review the release contents.
- Configure the GitHub `pypi` environment and PyPI trusted publisher for this
  repository's `release.yml` workflow. Environment protection may require a
  maintainer's approval. These remote settings are not changed by this audit.

## Validation and publishing

Pushes and pull requests run Python 3.10–3.13 tests, separate optional framework
jobs, and wheel/sdist validation. The tree job installs XGBoost and LightGBM.
The package job installs the wheel into a fresh environment using public
package resolution, runs `pip check` and the runnable README example, and
rebuilds a wheel from the sdist.

Only a **published GitHub release** triggers the PyPI workflow; a tag push alone
never publishes. The release tag must equal `v` plus the package version.
Development and local versions are rejected. The release job tests all optional
backends, validates build metadata, rebuilds from the sdist, and smoke-tests a
fresh wheel installation before the OIDC publishing job can run.

When explicitly authorized to publish, create the matching tag and GitHub
release after CI passes. This workflow intentionally leaves version selection,
release creation, PyPI publication, and repository visibility to the maintainer.

## Dependency policy

CBaseline `>=0.1.2,<0.2`, skgrad `>=0.1.5,<0.2`, and TreeIG `>=0.2.0,<0.3`
are normal dependencies. The lower bounds identify the release-tested stack;
skgrad's feature-space API is required. Upper bounds avoid silently accepting
new minor API series while these dependencies remain pre-1.0. Broaden bounds
after compatibility testing. The `trees` extra remains a compatibility alias;
it is unnecessary for normal installation.

The paper, comprehensive SHAP benchmarks, conda-forge, GPU work, and expanded
plotting are not prerequisites for this release.
