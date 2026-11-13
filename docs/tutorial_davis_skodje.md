# Tutorial 1: Davis-Skodje (the theory, step by step)

The Davis-Skodje system is the running example of Sections 3-5 of the paper: two variables, one
fast and one slow mode, and an exactly known curved slow invariant manifold.

$$\dot y^1=-y^1,\qquad
\dot y^2=-\frac1\epsilon\Big(y^2-\frac{y^1}{1+y^1}\Big)-\frac{y^1}{(1+y^1)^2},\qquad
\mathcal M_{SIM}:\ y^2=\frac{y^1}{1+y^1}.$$

```bash
csp-scalespace run davis_skodje -o figs_ds
```

or, in Python,

```python
import csp_scalespace as css
from csp_scalespace.models import davis_skodje, builtin_model_path

model = davis_skodje(eps=0.08)                     # y0 = (1.5, 0.2)
study = css.ScaleSpaceStudy.from_file(builtin_model_path("davis_skodje"))
```

## The model and its two scales

```{figure} _static/gallery/davis_skodje/01_time_evolution.png
:width: 65%

`01_time_evolution`: $y^2$ joins the SIM value within the fast layer $t\lesssim3\tau_1$.
```

```{figure} _static/gallery/davis_skodje/02_phase_portrait.png
:width: 65%

`02_phase_portrait`: trajectories fall along the fast direction $\mathbf a_1$ onto the SIM.
```

## CSP basis

```python
basis = css.csp_basis(model, model.y0)
basis.tau                         # [0.08, 1.0]            tau_1 = eps, tau_2 = 1
basis.A                           # columns a_1 = (0, 1), a_2 = (0.985, 0.174)
basis.amplitudes(model.g(model.y0))   # f = (5.025, -1.523)
```

The fast vector $\mathbf a_1=(0,1)^T$ is the vertical fibre; the large fast amplitude $f^1$ says
that the initial state is far from the manifold.

## Survival weights and the cutoff scale

```{figure} _static/gallery/davis_skodje/03_filter_weights.png
:width: 70%

`03_filter_weights`: between $\tau_1$ and $\tau_2$ the fast mode is gone and the slow mode survives.
```

```python
sigma_c = css.cutoff_scale(basis.tau, 1)           # sqrt(tau_1 tau_2) = 0.2828
css.ScaleFilter("gaussian").weights(sigma_c, basis.tau)   # [3.7e-06, 0.9231]
D, S = css.pointers(basis, sigma_c)                # D = [0.077, 1.000]
```

At $\sigma_c$ the fast pointer of $y^2$ is one: $y^2$ is the slaved variable. The value 0.077 for
$y^1$ is $1-e^{-\epsilon}$, the $O(\epsilon)$ attenuation of the slow mode discussed in the
{doc}`faq`.

## Why the filtering must be anisotropic

```{figure} _static/gallery/davis_skodje/04_isotropic_vs_anisotropic.png
:width: 100%

`04_isotropic_vs_anisotropic`. Left: Gaussian smoothing of the trajectory in time leaves the
manifold. Right: Formulation B moves perturbed states along the fast fibres only.
```

## Formulation A and Formulation B

```{figure} _static/gallery/davis_skodje/05_geometric_duality.png
:width: 80%

`05_geometric_duality`.
```

Formulation A keeps the state and filters the velocity:

```python
model.g(model.y0)                                  # [-1.5, 4.76]
css.filtered_field(model, model.y0, sigma_c)       # [-1.385, -0.245]: the fast restoring part is gone
```

Integrating this field from the *unrelaxed* state gives the orange curve: it moves parallel to the
manifold and never reaches it. Formulation B moves the state instead:

```python
path = css.fiber_relaxation(model, model.y0, sigma_c, order=1, M=1)
path.final                                         # [1.3847, 0.5807]   (paper, Section 5.3)
```

## First-order refinement

The eigenvectors of the Jacobian ignore the curvature of the manifold. One $\mathbf B$- and one
$\mathbf A$-refinement correct this:

```python
refined = css.csp_basis(model, model.y0, order=1, M=1)
```

```{figure} _static/gallery/davis_skodje/06_refinement_geometry.png
:width: 100%

`06_refinement_geometry`. Right: fraction of the on-manifold velocity wrongly assigned to the fast
subspace, for the eigenbasis (blue) and the refined basis (red).
```

```{figure} _static/gallery/davis_skodje/08_refinement_convergence.png
:width: 65%

`08_refinement_convergence`: the fitted slopes reproduce Theorem 5.3 - velocity errors
$O(\epsilon)$ and $O(\epsilon^2)$, manifold errors $O(\epsilon^2)$ and $O(\epsilon^3)$.
```

This last figure needs to know which parameter controls the gap; in the model file:

```yaml
settings:
  gap_parameter: eps
  gap_values: {start: -3.5, stop: -0.8, num: 50, scale: log}
  gap_reference_state: [1.0, 0.5]
```

## The two-stage operator

```{figure} _static/gallery/davis_skodje/07_two_stage.png
:width: 70%

`07_two_stage`: Stage 1 (dash-dotted) relaxes the initial state onto the manifold, Stage 2 (solid
red) follows the manifold; the unrelaxed Formulation A flow (orange) keeps its offset.
```

```python
study.stage1(1).final        # [1.3847, 0.5807]
study.reduced(1)["error"]    # 0.062
```

The error of 0.062 is not a tangency error: with $\epsilon=0.08$ the Gaussian filter slows the slow
mode by $1-e^{-\epsilon}\approx7.7\,\%$. Try

```bash
csp-scalespace run davis_skodje --set eps=0.01
```

or a sharper filter (`filter: exponential`, `p: 4`), and compare with the table in the {doc}`faq`.
