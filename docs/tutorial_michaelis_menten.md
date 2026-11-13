# Tutorial 2: Michaelis-Menten (a regime shift)

Section 6.1 of the paper. The mechanism $S+E\rightleftharpoons C\to E+P$ with state
$\mathbf y=(c,s)$:

$$\dot c=k_{1f}(e_0-c)s-k_{1b}c-k_2c,\qquad \dot s=-k_{1f}(e_0-c)s+k_{1b}c .$$

For the parameters of the paper ($k_{1f}=1000$, $k_{1b}=0.01$, $k_2=0.1$, $e_0=10$,
$c(0)=9.999999$, $s(0)=10$) the roles of the variables invert at $t\approx10$ s: before, $c$ is
slaved (standard QSSA); after, $s$ is slaved (reverse QSSA). No single classical reduction with a
fixed fast variable covers both regimes.

```bash
csp-scalespace run michaelis_menten -o figs_mm
```

## Model file

Reaction rates are written once under `definitions`; the PEA model of the literature is added as a
reference:

```{literalinclude} ../src/csp_scalespace/models/michaelis_menten.yaml
:language: yaml
:lines: 5-34
```

## Timescales move by orders of magnitude

```{figure} _static/gallery/michaelis_menten/01_time_evolution.png
:width: 65%

`01_time_evolution` (`log_states: true`).
```

```{figure} _static/gallery/michaelis_menten/09_timescales.png
:width: 65%

`09_timescales`: $\tau_2$ falls from about $10^6$ s to 10 s; the adaptive cutoff
$\sigma_c(t)=\sqrt{\tau_1\tau_2}$ stays in the middle of the gap.
```

From `report.md`:

| quantity | t = 2 s | t = 10 s | t = 15 s |
|---|---|---|---|
| $\tau_1$ | 1.25e-4 | 1.87e-2 | 2.54e-4 |
| $\tau_2$ | 5.82e+5 | 1.93e+1 | 1.00e+1 |
| $\sigma_c$ | 8.53 | 0.601 | 0.0504 |
| $D^c(\sigma_c)$ | 1.0000 | 0.4819 | 0.0000 |
| $D^s(\sigma_c)$ | 0.0000 | 0.5190 | 1.0000 |
| $\sum_kD^k$ | 1.0000 | 1.0010 | 1.0000 |

## Pointers identify the slaved variable

`snapshots: [2.0, 10.0, 15.0]` gives one pointer-versus-scale figure per regime:

```{figure} _static/gallery/michaelis_menten/10_pointers_vs_scale_t2.png
:width: 65%

`10_pointers_vs_scale_t2`: inside the SIM window the fast pointer of $c$ is one.
```

```{figure} _static/gallery/michaelis_menten/10_pointers_vs_scale_t10.png
:width: 65%

`10_pointers_vs_scale_t10`: at the corner the fast mode is shared almost equally by $c$ and $s$.
```

```{figure} _static/gallery/michaelis_menten/10_pointers_vs_scale_t15.png
:width: 65%

`10_pointers_vs_scale_t15`: now $s$ is the slaved variable.
```

```{figure} _static/gallery/michaelis_menten/11_pointers_trajectory.png
:width: 65%

`11_pointers_trajectory`: the inversion along the trajectory, with $\sum_kD^k=1$ throughout.
```

In Python the same curves are

```python
import csp_scalespace as css
from csp_scalespace.models import builtin_model_path

study = css.ScaleSpaceStudy.from_file(builtin_model_path("michaelis_menten"))
D, S = study.pointers_at(study.sigma_c(1))     # shape (T, 2): columns c, s
```

## One reduced model for both regimes

```{figure} _static/gallery/michaelis_menten/14_model_comparison_adaptive.png
:width: 70%

`14_model_comparison_adaptive`: full model (solid), scale-space model (dashed) and the PEA model
(markers).
```

| scale | Stage-1 state | max. error |
|---|---|---|
| adaptive $\sigma_c(\mathbf y)$ | (9.99989, 10.00001) | 5.9e-3 |
| fixed $\sigma=0.25$ s | (9.99989, 10.00011) | 5.9e-3 |
| fixed $\sigma=0.1$ s | (9.99989, 10.00011) | 5.9e-3 |

The fixed scales work here because they lie inside the gap at all times
($\max_t\tau_1\approx0.019$ s, $\min_t\tau_2=10$ s); the adaptive scale needs no such knowledge.
`fixed_sigmas: [0.25, 0.1]` requests the two additional comparisons.

```{note}
The PEA markers are obtained by integrating the PEA equations (Eq. 94 of the paper) from the same
initial state. After the corner their substrate value decays to zero (shown at the plotting floor),
whereas the full model keeps a small quasi-steady value of $s$.
```

## Settings used

| setting | value | reason |
|---|---|---|
| `order` | 0 | zeroth-order basis, as in the paper for this example |
| `fiber_filter` | `rational` | Formulation B over 13 decades of $\sigma$ |
| `tau_relax_max` | 0.1 | the adaptive $\sigma_c$ reaches several seconds; cap the stabilisation time |
| `phase_log`, `log_states` | `true` | states span six decades |
| `sigma_range` | `[1e-6, 3e7]` | covers $\tau_1$ and $\tau_2$ at all times |
