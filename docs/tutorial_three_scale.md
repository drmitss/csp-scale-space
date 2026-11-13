# Tutorial 3: three-scale network (nested manifolds)

Section 6.2 of the paper. Three species, an ultra-fast reversible reaction $R_1$, an intermediate
conversion $R_2$ and slow degradation:

$$R_1=\frac{y^1y^2-y^3}{\epsilon_1},\quad R_2=\frac1{\epsilon_2}\Big(\frac{y^1}{1+y^1}-y^2\Big),\qquad
\begin{aligned}\dot y^1&=-0.4R_1-R_2-0.5y^1\\ \dot y^2&=-0.6R_1+R_2-0.2(y^2)^2\\ \dot y^3&=R_1-0.1y^3\end{aligned}$$

with $\epsilon_1=10^{-3}$, $\epsilon_2=5\times10^{-2}$, $\mathbf y_0=(1,\,0.78,\,0.18)$. There are
two spectral gaps, hence two reduction levels: $M=1$ (a 2-D manifold, $R_1\approx0$) and $M=2$ (a
1-D manifold, $R_1\approx R_2\approx0$).

```bash
csp-scalespace run three_scale -o figs_3s
```

## Spectrum at the initial state

```python
import csp_scalespace as css
from csp_scalespace.models import three_scale

model = three_scale()
basis = css.csp_basis(model, model.y0)
basis.lam                              # [-1913.4, -24.22, -0.298]
basis.tau                              # [5.23e-4, 4.13e-2, 3.35]
css.cutoff_scale(basis.tau, 1)         # 4.65e-3   centre of the first gap
css.cutoff_scale(basis.tau, 2)         # 0.372     centre of the second gap
```

```{figure} _static/gallery/three_scale/03_filter_weights.png
:width: 70%

`03_filter_weights`: three weights, two SIM windows.
```

## Pointers across the scale axis

```{figure} _static/gallery/three_scale/10_pointers_vs_scale_t1e-05.png
:width: 70%

`10_pointers_vs_scale`: in the first window the fast mode is shared by all three species
(it is the reaction $R_1$); in the second window $y^2$ is the most strongly slaved variable.
```

```python
D, S = css.pointers(basis, [1e-5, 4.65e-3, 0.372, 20.0])
D.sum(axis=1)                          # [0.000, 1.013, 2.012, 3.000]
```

| | $\sigma_{c,1}$ | $\sigma_{c,2}$ |
|---|---|---|
| $D^{1}$ | 0.165 | 0.570 |
| $D^{2}$ | 0.324 | 0.889 |
| $D^{3}$ | 0.523 | 0.554 |
| $\sum_kD^k$ | 1.013 | 2.012 |

## Formulation B: one plateau per manifold

```{figure} _static/gallery/three_scale/13_fiber_relaxation.png
:width: 70%

`13_fiber_relaxation`: the initial state relaxed over the whole scale axis, without any time
integration.
```

```python
path = css.fiber_relaxation(model, model.y0, 20.0, sigma_min=1e-5, filt="rational")
path.at(6e-3)                          # [0.874, 0.582, 0.504]   on the 2-D manifold: y3 = y1*y2
path.at(0.40)                          # [0.959, 0.497, 0.477]   on the 1-D manifold
```

## Time: sequential exhaustion

```{figure} _static/gallery/three_scale/09_timescales.png
:width: 65%

`09_timescales` (`time_scale: log`).
```

```{figure} _static/gallery/three_scale/12_realtime_pointers.png
:width: 65%

`12_realtime_pointers`: with $\sigma=t$ the number of exhausted modes climbs $0\to1\to2\to3$.
```

## Two reduced models

```{figure} _static/gallery/three_scale/14_model_comparison_adaptive.png
:width: 70%

`14_model_comparison_adaptive`: level 1 (dashed) starts on the 2-D manifold and is accurate after
the first layer; level 2 (dotted) starts on the 1-D manifold and is accurate after the second.
```

```{figure} _static/gallery/three_scale/07_two_stage.png
:width: 75%

`07_two_stage`: for three variables the two-stage figure is a 3-D view.
```

| level | Stage-1 state | max. error once the first $M$ modes are exhausted |
|---|---|---|
| $M=1$ | (0.874, 0.583, 0.503) | 5.8e-3 |
| $M=2$ | (0.960, 0.498, 0.478) | 1.1e-2 |

```python
from csp_scalespace.models import builtin_model_path

study = css.ScaleSpaceStudy.from_file(builtin_model_path("three_scale"))
study.levels                           # [1, 2]
study.reduced(2)["y"]                  # trajectory of the 1-D reduced model, shape (3, T)
```

## Settings used

| setting | value | reason |
|---|---|---|
| `time_scale`, `t_start` | `log`, `1e-5` | three layers spread over six decades of time |
| `levels` | `[1, 2]` | both gaps (this is also the automatic choice) |
| `fiber_filter` | `rational` | as in the paper for the fibre relaxation of this model |
| `phase_variables` | `[y1, y2, y3]` | three names give 3-D phase-space figures |
