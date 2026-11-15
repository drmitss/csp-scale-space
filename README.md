# csp-scalespace

[![tests](https://github.com/drmitss/csp-scale-space/actions/workflows/tests.yml/badge.svg)](https://github.com/drmitss/csp-scale-space/actions/workflows/tests.yml)
[![docs](https://readthedocs.org/projects/csp-scale-space/badge/?version=latest)](https://csp-scale-space.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Anisotropic spectral scale-space for multiscale nonlinear ODEs via Computational Singular Perturbation (CSP).**

📖 **Manual:** <https://csp-scale-space.readthedocs.io>

Reference implementation of

> D. M. Manias and L. Najman, *Anisotropic Spectral Scale-Space Formulation for Multiscale Nonlinear
> Ordinary Differential Equations via Computational Singular Perturbation*.

Classical CSP reduces a stiff system by choosing an integer number $`M`$ of exhausted (fast) modes and
projecting them out - a binary switch. This package replaces the switch with a continuous dial: an
observation scale $`\sigma`$. Each CSP mode is attenuated by a smooth weight

```math
w_i(\mathbf y;\sigma)=\exp\!\left[-\left(\frac{\sigma}{\tau_i(\mathbf y)}\right)^{2}\right]
```

that depends on the ratio of $`\sigma`$ to the mode's timescale $`\tau_i`$, so one global model moves
continuously from the full dynamics ($`\sigma\to0`$) through every slow invariant manifold (SIM) of the system.

You give the package **a model** (equations, parameters, initial conditions) and it gives you

* the **figures of the paper computed for your model** - both the step-by-step theory figures
  (shown in the paper for the Davis-Skodje system) and the application figures (shown for the
  Michaelis-Menten mechanism and a three-scale network);
* a **report** with timescales, cutoff scales, CSP pointers and reduced-model errors;
* a small, documented **library** of the operators themselves ($`\mathcal S_\sigma`$, pointers, Formulation A,
  Formulation B, the two-stage reduced model, first-order CSP refinements).

It also reproduces **every figure of the paper** exactly (`paper/`).

| Fig. 6a of the paper | Fig. 12c of the paper | Fig. 10a of the paper |
|---|---|---|
| ![](docs/_static/paper/fig1a_phase_space.png) | ![](docs/_static/paper/fig_3d_dynamic_pointers.png) | ![](docs/_static/paper/fig_mm_comparison_sqssa_adaptive.png) |
| Two-stage operator: fibre relaxation, then tangent flow (Davis-Skodje) | Sequential mode exhaustion with $`\sigma=t`$ (three-scale network) | Scale-space reduced model across a regime shift (Michaelis-Menten) |

---

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Describing your model](#describing-your-model)
4. [What you get: the figure catalogue](#what-you-get-the-figure-catalogue)
5. [The method in one page](#the-method-in-one-page)
6. [Using the library directly](#using-the-library-directly)
7. [Reproducing the paper](#reproducing-the-paper)
8. [Settings reference](#settings-reference)
9. [Assumptions and limitations](#assumptions-and-limitations)
10. [Repository layout](#repository-layout) · [Tests](#tests) · [Documentation](#documentation) · [Citation](#citation)

---

## Installation

```bash
git clone https://github.com/drmitss/csp-scale-space.git
cd csp-scale-space
pip install -e .            # add ".[test]" for pytest, ".[docs]" for the documentation
```

Requires Python ≥ 3.10 with NumPy, SciPy, Matplotlib, SymPy and PyYAML (installed automatically).

## Quick start

**From the command line** - no Python needed:

```bash
csp-scalespace template my_model.yaml      # write a commented template, then edit the equations
csp-scalespace run my_model.yaml -o figs   # all figures + figs/report.md
csp-scalespace run davis_skodje            # or try a built-in model
```

**From Python:**

```python
import csp_scalespace as css

model = css.ODEModel.from_equations(
    equations={"y1": "-y1",
               "y2": "-(y2 - y1/(1 + y1))/eps - y1/(1 + y1)**2"},
    parameters={"eps": 0.08},
    initial_conditions={"y1": 1.5, "y2": 0.2},
)
study = css.ScaleSpaceStudy(model, t_end=4.0)
study.make_figures("figs")                 # PNG files + report.md
print(study.summary())                     # timescales, cutoff scales, pointers
```

The Jacobian and all other derivatives are obtained symbolically from the text of the equations -
you never write them by hand.

## Describing your model

A model is an autonomous system $`\dot{\mathbf y}=\mathbf g(\mathbf y)`$, $`\mathbf y(0)=\mathbf y_0`$. There are three equivalent ways to
provide it.

### 1. A model file (YAML or JSON)

```yaml
name: three_scale
description: "3D three-timescale network"

parameters:  {eps1: 1.0e-3, eps2: 5.0e-2}

definitions:                         # optional intermediate quantities
  R1: "(y1*y2 - y3)/eps1"
  R2: "(y1/(1 + y1) - y2)/eps2"

equations:                           # d(variable)/dt = expression
  y1: "-0.4*R1 - R2 - 0.5*y1"
  y2: "-0.6*R1 + R2 - 0.2*y2**2"
  y3: "R1 - 0.1*y3"

initial_conditions: {y1: 1.0, y2: 0.78, y3: 0.18}

settings:                            # analysis options; only t_end is required
  t_end: 20.0
  time_scale: log
  t_start: 1.0e-5
```

Expressions use Python syntax (`**` for powers; `exp`, `log`, `sqrt`, `Max`, ... are available).
Optional blocks: `labels` (axis labels), `exact_sim` (an analytically known SIM),
`reference_models` (classical reduced models such as QSSA/PEA to overlay), `settings`.
Complete, commented examples: [`src/csp_scalespace/models/`](src/csp_scalespace/models) (the three
models of the paper and `template.yaml`).

### 2. Equations as text, in Python

`ODEModel.from_equations(equations, initial_conditions, parameters, definitions, labels, exact_sim)` -
the same content as the model file, as dictionaries (see the quick start).

### 3. Python functions

For models that cannot be written as formulas (table look-ups, external libraries):

```python
def rhs(y, p):                                   # g(y); p is the parameter dictionary
    return [-p["k"] * y[0], p["k"] * y[0] - y[1]]

model = css.ODEModel(rhs, y0=[1.0, 0.0], params={"k": 50.0})      # jac=..., jac_dot=... optional
```

Missing derivatives are approximated by finite differences. See
[`examples/02_model_from_python_functions.py`](examples/02_model_from_python_functions.py)
(Robertson kinetics).

## What you get: the figure catalogue

`study.make_figures(outdir)` / `csp-scalespace run` writes the files below. Every curve is computed
from your model; nothing is schematic. The *paper* column gives the corresponding figure of the
manuscript, where the same content is shown for the benchmark models.

**Theory group** - the construction of the method, step by step

| file | paper | content |
|---|---|---|
| `01_time_evolution` | Fig. 1a, 7a | state variables in time, fast initial layer shaded |
| `02_phase_portrait` | Fig. 1b, 7b | trajectories collapsing onto the slow invariant manifold, fast-fibre directions |
| `03_filter_weights` | Fig. 2 | survival weights $`w_i(\sigma)`$ of all modes, timescales $`\tau_i`$, cutoff scales $`\sigma_c`$, SIM windows |
| `04_isotropic_vs_anisotropic` | Fig. 3 | naive Gaussian smoothing drifts off the SIM; anisotropic CSP filtering returns perturbed states to it |
| `05_geometric_duality` | Fig. 4 | Formulation A (filter the velocity) vs. Formulation B (relax the state) at the initial state |
| `06_refinement_geometry` | Fig. 5 | basis vectors before/after the $`\mathbf B`$- and $`\mathbf A`$-refinements; tangency error of both bases |
| `07_two_stage` | Fig. 6a, 13 | Stage 1 fibre relaxation + Stage 2 tangent flow in phase space (2-D or 3-D) |
| `08_refinement_convergence` | Fig. 6b | error orders vs. the spectral gap (needs `gap_parameter`) |

**Applications group** - diagnostics and reduced models along the trajectory

| file | paper | content |
|---|---|---|
| `09_timescales` | Fig. 8a, 12b | $`\tau_i(\mathbf y(t))`$ and adaptive cutoff scales $`\sigma_{c,M}(t)=\sqrt{\tau_M\,\tau_{M+1}}`$ |
| `10_pointers_vs_scale_t*` | Fig. 9a-c, 11a | fast/slow CSP pointers of each variable against $`\sigma`$ at the snapshot times |
| `11_pointers_trajectory` | Fig. 8b, 9d | pointers at the adaptive scale along the trajectory: which variable is slaved, and when |
| `12_realtime_pointers` | Fig. 12c | pointers with $`\sigma=t`$: staircase $`M(t)=0\to1\to2\to\dots`$ |
| `13_fiber_relaxation` | Fig. 11b | initial state relaxed across all scales: one plateau per manifold |
| `14_model_comparison_*` | Fig. 10, 12a | scale-space reduced models (every level $`M`$, adaptive or fixed $`\sigma`$) vs. full and reference models |

The corresponding figures of the paper, for its three benchmark models:

| Davis-Skodje | Michaelis-Menten | Three-scale network |
|---|---|---|
| ![](docs/_static/paper/fig_geom2_filter_weights.png) | ![](docs/_static/paper/fig_mm_adaptive_timescales.png) | ![](docs/_static/paper/fig2b_3d_fiber_relaxation.png) |
| Fig. 2 (`03_filter_weights`) | Fig. 8a (`09_timescales`) | Fig. 11b (`13_fiber_relaxation`) |
| ![](docs/_static/paper/fig1b_asymptotic_convergence.png) | ![](docs/_static/paper/fig_mm_pointers_trajectory.png) | ![](docs/_static/paper/fig_3d_time_comparison.png) |
| Fig. 6b (`08_refinement_convergence`) | Fig. 8b (`11_pointers_trajectory`) | Fig. 12a (`14_model_comparison`) |

The versions produced by the generic pipeline for the same three models are in
[`docs/_static/gallery/`](docs/_static/gallery).

## The method in one page

Notation: $`N`$ variables, modes ordered from fastest to slowest,
$`\tau_1<\tau_2<\dots<\tau_N`$; $`M`$ is the number of exhausted (fast) modes.

**CSP basis** (Eqs. 19, 26) - `csp_basis(model, y)`. Right and left eigenvectors of the Jacobian
$`\mathbf J=\partial\mathbf g/\partial\mathbf y`$:

```math
\mathbf J\,\mathbf a_i=\lambda_i\,\mathbf a_i,\qquad
\mathbf b^i\,\mathbf J=\lambda_i\,\mathbf b^i,\qquad
\mathbf b^i\cdot\mathbf a_j=\delta^i_j,\qquad
\tau_i=\frac{1}{\lvert\lambda_i\rvert}
```

**Modal amplitudes** (Eq. 12) - `basis.amplitudes(g)`:

```math
\mathbf g(\mathbf y)=\sum_{i=1}^{N}\mathbf a_i\,f^i,\qquad f^i=\mathbf b^i\cdot\mathbf g
```

**Survival weights** (Eqs. 36-37) - `ScaleFilter.weights`. Exponential family (Gaussian for
$`p=2`$), rational filter, and the sharp limit $`p\to\infty`$:

```math
w_i=\exp\!\left[-\left(\frac{\sigma}{\tau_i}\right)^{p}\right],\qquad
w_i=\frac{1}{1+(\sigma/\tau_i)^2},\qquad
w_i=\Theta(\tau_i-\sigma)
```

**Scale-space operator and Formulation A** (Eqs. 42-43) - `scale_space_operator`,
`filtered_field`. The velocity is filtered, the state is held fixed:

```math
\mathcal S_\sigma=\mathbf A\,\mathbf W\,\mathbf B=\sum_{i=1}^{N}w_i\,\mathbf a_i\,\mathbf b^i,\qquad
\mathbf g_\sigma=\mathcal S_\sigma\,\mathbf g
```

**Formulation B** (Eq. 49) - `fiber_relaxation`. The state is relaxed along the fast fibres, with
$`d_i=-\partial_\sigma\ln w_i`$:

```math
\frac{\partial\mathbf y_\sigma}{\partial\sigma}
=-\sum_{i=1}^{N}\mathbf a_i(\mathbf y_\sigma)\,
\frac{d_i(\mathbf y_\sigma;\sigma)}{\lambda_i(\mathbf y_\sigma)}\,f^i(\mathbf y_\sigma),
\qquad \mathbf y_{\sigma=0}=\mathbf y
```

**Cutoff scale** (Theorem 4.2) - `cutoff_scale`. The centre of the spectral gap between modes
$`M`$ and $`M+1`$:

```math
\sigma_{c,M}=\sqrt{\tau_M\,\tau_{M+1}}
```

**Scale-dependent pointers** (Eqs. 56-57) - `pointers`. Fast pointer $`D^k`$ and slow pointer
$`S^k`$ of variable $`k`$:

```math
S^k(\mathbf y;\sigma)=\sum_{i=1}^{N}w_i\,a_i^k\,b^i_k,\qquad
D^k=1-S^k,\qquad \sum_{k=1}^{N}D^k\approx M(\sigma)
```

**First-order refinement** (Eqs. 66-73) - `refine_basis`, `csp_basis(..., order=1, M=M)`. One
$`\mathbf B`$- and one $`\mathbf A`$-refinement account for the curvature of the manifold
(subscripts $`f`$, $`s`$: fast and slow blocks):

```math
\mathbf B_f^{(1)}=\mathbf B_f+\mathbf P_0\,\mathbf B_s,\qquad
\mathbf A_s^{(1)}=\mathbf A_s-\mathbf A_f\,\mathbf P_0,\qquad
\mathbf A_f^{(1)}=\mathbf A_f-\mathbf A_s^{(1)}\,\mathbf Q_0,\qquad
\mathbf B_s^{(1)}=\mathbf B_s+\mathbf Q_0\,\mathbf B_f^{(1)}
```

**Two-stage reduced model** (Eq. 50) - `two_stage_rhs`, `study.reduced(M)`. Stage 1 relaxes the
state onto the manifold, Stage 2 integrates the filtered field from there:

```math
\mathbf y\ \longmapsto\ \mathbf y_\sigma,\qquad
\dot{\mathbf y}_\sigma=\mathcal S_\sigma(\mathbf y_\sigma)\,\mathbf g(\mathbf y_\sigma)
```

With the sharp filter $`\mathcal S_\sigma`$ is exactly the classical CSP slow projector with
$`M(\sigma)`$ exhausted modes (Eq. 45). With a smooth filter and $`\sigma=\sigma_c`$ it agrees with
it up to $`O(\epsilon)`$, where $`\epsilon=\tau_M/\tau_{M+1}`$ (Theorem 4.2), and with the
first-order basis the filtered field is tangent to the SIM up to $`O(\epsilon^2)`$ (Theorem 5.3).
Both statements are checked in the test-suite.

## Using the library directly

```python
import csp_scalespace as css
from csp_scalespace.models import three_scale

model = three_scale()
basis = css.csp_basis(model, model.y0)                 # λ_i, τ_i, A, B
s1 = css.cutoff_scale(basis.tau, 1)                    # centre of the first spectral gap
D, S = css.pointers(basis, s1)                         # fast / slow pointers of y1, y2, y3
g_s = css.filtered_field(model, model.y0, s1)          # Formulation A
path = css.fiber_relaxation(model, model.y0, "adaptive", M=1)   # Formulation B -> point on the 2D SIM
rhs = css.two_stage_rhs(model, M=1, sigma="adaptive")  # reduced model, ready for any ODE solver
y = css.integrate(rhs, path.final, [0.0, 1.0, 5.0])
```

More in [`examples/`](examples): `01_model_from_equations.py`, `02_model_from_python_functions.py`,
`03_low_level_operators.py`.

## Reproducing the paper

```bash
python paper/reproduce_paper_figures.py            # all 13 figures (28 panels) -> paper_figures/*.pdf, *.png
python paper/reproduce_paper_figures.py --only 6 12
```

The scripts in [`paper/`](paper) are the original figure code of the manuscript with the numerics
routed through this library. `PAPER_FIGURES` in `reproduce_paper_figures.py` maps each figure and
panel to its function and file name. Figures 3, 4 and 5 of the paper are *schematic illustrations*;
their computed counterparts are figures `04`-`06` of the catalogue above.

| Fig. 1b | Fig. 9b | Fig. 13 |
|---|---|---|
| ![](docs/_static/paper/fig_ds_intro_phase.png) | ![](docs/_static/paper/fig_mm_pointers_t10.png) | ![](docs/_static/paper/fig_3d_sim_trajectories.png) |

## Settings reference

All settings can be given in the `settings:` block of a model file or as keyword arguments of
`ScaleSpaceStudy`. Only `t_end` is required.

| setting | default | meaning |
|---|---|---|
| `t_end`, `n_points` | -, 800 | final time and number of output times |
| `time_scale`, `t_start` | `linear`, 0 | `log` for dynamics spanning decades (then `t_start > 0` is the first output time) |
| `time_unit` | `""` | unit shown on axes |
| `filter`, `p` | `gaussian`, 2 | scale filter: `gaussian`, `exponential` (exponent `p`), `rational` |
| `fiber_filter` | same as `filter` | filter used for Formulation B (`rational` is the most robust over many decades of $`\sigma`$) |
| `order` | 1 | CSP basis: 0 = Jacobian eigenvectors, 1 = one B- and one A-refinement |
| `levels` | all gaps | numbers $`M`$ of exhausted modes to analyse |
| `fixed_sigmas` | `[]` | additional reduced models at constant $`\sigma`$ |
| `stabilize`, `tau_relax_max` | `true`, none | keep the reduced model on the manifold (continuous Stage 1 term) |
| `snapshots` | `[0]` | times of the pointer-vs-scale figures |
| `sigma_range` | automatic | range of the scale axis |
| `phase_variables`, `phase_log` | first 2-3 variables, `false` | variables and axis type of phase-space figures |
| `extra_initial_conditions`, `extra_t_end` | automatic | additional trajectories of the phase portrait |
| `log_states`, `state_floor` | `false`, none | logarithmic state axes |
| `gap_parameter`, `gap_values`, `gap_reference_state` | none | parameter sweep for `08_refinement_convergence` |
| `reference_models` | `{}` | reduced models to overlay in `14_model_comparison` |
| `rtol`, `atol` | 1e-10, 1e-12 | integrator tolerances (Radau) |

## Assumptions and limitations

The implementation follows the assumptions of the paper (Section 3.3) and reports when they fail:

* **Real spectrum.** Complex-conjugate eigenvalues are outside the theory (Section 8.5); a
  `ComplexSpectrumWarning` is issued and results at such states should not be trusted.
* **Ordered, non-crossing timescales.** Modes are sorted by $`\lvert\lambda_i\rvert`$ at every state.
  Where two eigenvalues cross, the refinement (which divides by $`\lambda_m-\lambda_n`$) is
  ill-conditioned.
* **Dissipative fast modes.** Formulation B relaxes only modes with $`\lambda_i<0`$; conserved
  ($`\lambda=0`$) and growing modes are left untouched.
* **Stage 2 stabilisation.** For long integrations the reduced model contains, by default, a
  continuous Stage-1 term that returns the state to the manifold on the time scale $`\sigma`$
  (`stabilize`); this is the form used for the Michaelis-Menten and three-scale results of the paper.
  Set `stabilize: false` for the pure tangent flow of Eq. (50).
* **Refined timescales.** The first-order basis keeps the eigenvalue timescales $`\tau_i`$; the
  refined diagonal entries $`\Lambda^{i,(1)}_i`$ differ from them at higher order only.
* Dense linear algebra (one eigen-decomposition per evaluation): intended for small and medium $`N`$.

## Repository layout

```
src/csp_scalespace/
    model.py        ODEModel: equations / model files / Python functions
    csp.py          CSP basis: eigenbasis and first-order B-, A-refinement
    filters.py      scale filters (weights, attenuation rates)
    scalespace.py   operators: S_sigma, pointers, Formulation A and B, two-stage model
    study.py        ScaleSpaceStudy: trajectory, diagnostics, reduced models, settings
    figures.py      model-independent figure catalogue
    cli.py          the csp-scalespace command
    models/         built-in models (YAML) and the model-file template
paper/              exact reproduction of the manuscript figures
examples/           three worked examples and a model-file template
tests/              unit tests, regression tests against numbers quoted in the paper
docs/               Sphinx documentation sources (Read the Docs ready) and figure gallery
```

## Tests

```bash
pip install -e ".[test]" && pytest
```

`tests/test_paper_values.py` compares the library against values printed in the manuscript
(Stage-1 state of Section 5.3, the error orders of Fig. 6b, Table 2, Eq. 104, Table 3).

## Documentation

The manual is hosted at <https://csp-scale-space.readthedocs.io> (tutorials for the three benchmark
models, a walkthrough for your own model, figure catalogue, settings, theory-to-code map, API).
Its sources are in [`docs/`](docs) (Sphinx + MyST Markdown); Read the Docs builds them on every push
using `.readthedocs.yaml`. To preview locally:

```bash
pip install -e ".[docs]" && sphinx-build -b html docs docs/_build/html
```

## Citation

If you use this software, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{ManiasNajman_CSPScaleSpace,
  author  = {Manias, Dimitris M. and Najman, Laurent},
  title   = {Anisotropic Spectral Scale-Space Formulation for Multiscale Nonlinear Ordinary
             Differential Equations via Computational Singular Perturbation},
  year    = {2026},
  note    = {Manuscript}
}
```

## License

MIT - see [`LICENSE`](LICENSE).
