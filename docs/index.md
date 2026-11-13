# csp-scalespace

**Anisotropic spectral scale-space for multiscale nonlinear ODEs via Computational Singular
Perturbation (CSP).** Reference implementation of the paper by D. M. Manias and L. Najman.

Classical CSP reduces a stiff system by choosing an integer number $M$ of exhausted (fast) modes
and projecting them out: a binary switch. Here the switch is replaced by a continuous dial, the
*observation scale* $\sigma$. Every CSP mode is attenuated by a smooth weight
$w_i=\exp[-(\sigma/\tau_i)^2]$ that depends on the ratio of $\sigma$ to the timescale $\tau_i$ of
the mode, so a single model moves continuously from the full dynamics ($\sigma\to0$) through every
slow invariant manifold (SIM) of the system.

```{figure} _static/gallery/three_scale/12_realtime_pointers.png
:width: 70%

Sequential exhaustion of the modes of a three-timescale network: the number of exhausted modes
climbs $0\to1\to2\to3$ as the observation scale grows.
```

## What the package does

You provide **a model** - equations, parameters, initial conditions - and obtain

- the **figures of the paper computed for your model**: the step-by-step theory figures (shown in
  the paper for the Davis-Skodje system) and the application figures (shown for the
  Michaelis-Menten mechanism and a three-scale network);
- a **report** with timescales, cutoff scales, CSP pointers and reduced-model errors;
- the **operators as a library**: $\mathcal S_\sigma$, scale-dependent pointers, Formulation A,
  Formulation B, the two-stage reduced model and the first-order CSP refinements.

```bash
pip install -e .
csp-scalespace template my_model.yaml      # edit the equations
csp-scalespace run my_model.yaml -o figs   # figures + report.md
```

## Where to go

| you want to ... | read |
|---|---|
| install and run something in two minutes | {doc}`installation`, {doc}`quickstart` |
| analyse your own system | {doc}`own_model`, {doc}`model_file`, {doc}`settings` |
| understand each figure | {doc}`figures` |
| see complete worked analyses | {doc}`tutorial_davis_skodje`, {doc}`tutorial_michaelis_menten`, {doc}`tutorial_three_scale` |
| connect the formulas of the paper to functions | {doc}`theory`, {doc}`api` |
| regenerate the manuscript figures | {doc}`paper` |

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
cli
```

```{toctree}
:maxdepth: 2
:caption: Your own model

own_model
model_file
settings
workflow
figures
faq
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorial_davis_skodje
tutorial_michaelis_menten
tutorial_three_scale
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
