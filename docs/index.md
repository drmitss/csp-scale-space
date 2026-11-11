# csp-scalespace

**Anisotropic spectral scale-space for multiscale nonlinear ODEs via Computational Singular
Perturbation (CSP)** - reference implementation of the paper by D. M. Manias and L. Najman.

Give the package a model (equations, parameters, initial conditions); it returns the figures of the
paper computed for *your* model, a numerical report, and the scale-space operators as a library.

```bash
pip install -e .
csp-scalespace template my_model.yaml
csp-scalespace run my_model.yaml -o figs
```

```{toctree}
:maxdepth: 2
:caption: User guide

installation
quickstart
model_file
workflow
figures
settings
```

```{toctree}
:maxdepth: 2
:caption: Background

theory
paper
limitations
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
