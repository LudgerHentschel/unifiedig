# References and citation

## Cite UnifiedIG

Please cite the software, together with the paper for the construction your
work relies on. Record the UnifiedIG version and the background settings used,
since attributions depend on the reference distribution.

```bibtex
@software{hentschel_unifiedig,
  author = {Hentschel, Ludger},
  title  = {{UnifiedIG}: A unified, model-agnostic interface for Integrated Gradients},
  url    = {https://github.com/LudgerHentschel/unifiedig},
}
```

See [CITATION.cff](https://github.com/LudgerHentschel/unifiedig/blob/main/CITATION.cff)
for the machine-readable record.

## The stack papers

- Hentschel, Ludger. 2026a.
  ["Canonical Integrated Gradients: Expectations over Neutral Prediction Baselines."](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26h.pdf)
  *www.ludgerhentschel.com/Research.html*
  Develops the neutral-manifold reference distribution, expected Integrated
  Gradients, kernel localization, and exponential calibration. Implemented in
  [CBaseline](https://github.com/LudgerHentschel/cbaseline).

- Hentschel, Ludger. 2026b.
  ["TreeIG: Exact Integrated Gradients for Tree-Based Models."](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26g.pdf)
  *www.ludgerhentschel.com/Research.html*
  Develops the distributional reading of the path integral for piecewise-constant
  models. Implemented in [TreeIG](https://github.com/LudgerHentschel/treeig).

See [the IG stack](ig-stack.md) for how CBaseline, skgrad, TreeIG, and UnifiedIG
divide the work.

## Value theory

- Aumann, Robert J., and Lloyd S. Shapley. 1974.
  *Values of Non-Atomic Games.* Princeton University Press.

- Friedman, Eric J. 2004.
  ["Paths and Consistency in Additive Cost Sharing."](https://doi.org/10.1007/s001820400173)
  *International Journal of Game Theory* 32 (4): 501–518.
  Establishes the uniqueness of the straight-line path among additive path
  methods, which fixes the path used by Integrated Gradients.

- Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. 2017.
  ["Axiomatic Attribution for Deep Networks."](https://proceedings.mlr.press/v70/sundararajan17a.html)
  *Proceedings of the 34th International Conference on Machine Learning*,
  PMLR 70:3319–3328.

- Sundararajan, Mukund, and Amir Najmi. 2020.
  ["The Many Shapley Values for Model Explanation."](https://proceedings.mlr.press/v119/sundararajan20b.html)
  *Proceedings of the 37th International Conference on Machine Learning*,
  PMLR 119:9269–9278.

## Shapley attribution in practice

- Chen, Hugh, Joseph D. Janizek, Scott Lundberg, and Su-In Lee. 2020.
  ["True to the Model or True to the Data?"](https://arxiv.org/abs/2006.16234)
  arXiv:2006.16234.
  Compares the interventional and observational conditional-expectation
  conventions for the value function, which give different attributions for the
  same model.

- Lundberg, Scott M., and Su-In Lee. 2017.
  "A Unified Approach to Interpreting Model Predictions."
  *Advances in Neural Information Processing Systems (NeurIPS).*

- Lundberg, Scott M., Gabriel Erion, and Su-In Lee. 2020.
  "From Local Explanations to Global Understanding with Explainable AI for Trees."
  *Nature Machine Intelligence.*

## Integrated Gradients BibTeX

```bibtex
@inproceedings{sundararajan2017axiomatic,
  title = {Axiomatic Attribution for Deep Networks},
  author = {Sundararajan, Mukund and Taly, Ankur and Yan, Qiqi},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {70},
  pages = {3319--3328},
  year = {2017},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v70/sundararajan17a.html},
}

@article{friedman2004paths,
  author  = {Friedman, Eric J.},
  title   = {Paths and consistency in additive cost sharing},
  journal = {International Journal of Game Theory},
  volume  = {32},
  number  = {4},
  pages   = {501--518},
  year    = {2004},
  doi     = {10.1007/s001820400173},
}

@inproceedings{sundararajan2020many,
  title     = {The Many {S}hapley Values for Model Explanation},
  author    = {Sundararajan, Mukund and Najmi, Amir},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {9269--9278},
  year      = {2020},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v119/sundararajan20b.html},
}

@misc{hentschel2026canonical,
  author = {Hentschel, Ludger},
  title = {Canonical Integrated Gradients: Expectations over Neutral Prediction Baselines},
  year = {2026},
  url = {https://www.ludgerhentschel.com/PDFs/Hentschel%20'26h.pdf},
}

@misc{hentschel2026treeig,
  author = {Hentschel, Ludger},
  title  = {{TreeIG}: Exact Integrated Gradients for Tree-Based Models},
  year   = {2026},
  url    = {https://www.ludgerhentschel.com/PDFs/Hentschel%20'26g.pdf},
}
```

## License

UnifiedIG is distributed under the
[BSD 3-Clause License](https://github.com/LudgerHentschel/unifiedig/blob/main/LICENSE).
