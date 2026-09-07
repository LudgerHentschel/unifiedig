# Building the documentation

Use Python 3.12 for the documentation toolchain, matching CI:

```console
python -m pip install -e ".[docs,shap]"
python scripts/check_docs_examples.py
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to preview the site. The site uses Sphinx,
MyST Markdown, and the PyData Sphinx Theme with the same complete-sidebar
navigation as CBaseline, skgrad, and TreeIG. API documentation is generated from
the public classes; worked examples are included from executable source files.
Warnings fail the build, including unresolved documentation references.

## GitHub Pages

In repository **Settings → Pages**, choose **GitHub Actions** as the build
source. The **Documentation** workflow checks examples and builds the site on
pushes and pull requests. Successful builds on `main` deploy through the
`github-pages` environment; pull requests never deploy.

If Pages was enabled after the push, open **Actions → Documentation → Run
workflow**, select `main`, and run it again. The workflow supports manual runs,
so no version bump or release tag is needed for documentation updates. Confirm
that the `github-pages` environment permits main-branch deployments.

The documentation address is
[https://ludgerhentschel.github.io/unifiedig/](https://ludgerhentschel.github.io/unifiedig/).
The workflow does not enable Pages settings, change repository visibility, or
publish a package to PyPI. Deploying Pages can expose documentation publicly,
depending on the repository's Pages access configuration.

## Maintain the guide

Add new guide pages to the `index.md` table of contents. Keep runnable examples
in `examples/`; add them to `scripts/check_docs_examples.py` when they appear in
the guide. Framework examples are checked when the relevant optional dependency
is installed, including in the existing framework test jobs.

For package publishing, use the separate [release checklist](releasing.md).
