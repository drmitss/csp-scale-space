# The analysis workflow

`ScaleSpaceStudy(model, **settings)` computes lazily and caches:

| attribute / method | content | paper |
|---|---|---|
| `study.t`, `study.full` | time grid and trajectory of the full model | - |
| `study.diagnostics` | `lam`, `tau`, `mode_pointers`, `u_fast` along the trajectory | Eqs. 19, 41, 54 |
| `study.levels` | reduction levels `M` (number of exhausted modes) analysed | - |
| `study.sigma_c(M)` | adaptive cutoff $\sigma_{c,M}(t)=\sqrt{\tau_M\tau_{M+1}}$ | Eq. 97 |
| `study.pointers_at(sigma)` | fast and slow pointers along the trajectory | Eqs. 56-57 |
| `study.fiber` | Formulation B over the whole scale axis | Eq. 49 |
| `study.stage1(M, sigma)` | Stage 1: fibre relaxation to the level-`M` manifold | Eq. 50 |
| `study.reduced(M, sigma)` | Stage 2: reduced-model trajectory, start state, error | Eq. 50 |
| `study.reference(label)` | trajectory of a user reference model | - |
| `study.summary()` | Markdown table of timescales, scales and pointers | Tables 2-3 |
| `study.make_figures(outdir)` | the figure catalogue and `report.md` | Figs. 1-13 |

## How the reduced model is built

1. **Stage 1.** The initial state is relaxed with Formulation B from $\sigma\approx0$ up to the
   self-consistent cutoff $\sigma=\sigma_{c,M}(\mathbf y_\sigma)$ (or to a fixed $\sigma$). The
   cutoff is evaluated at the *relaxed* state because the timescales of an off-manifold state can
   differ by orders of magnitude from those on the manifold.
2. **Stage 2.** From the relaxed state the system
   $\dot{\mathbf y}=\mathcal S_\sigma(\mathbf y)\mathbf g(\mathbf y)$ is integrated with the basis,
   timescales and (for `sigma="adaptive"`) the scale re-evaluated at every state. With
   `stabilize=True` a continuous Stage-1 term keeps the state on the manifold, see
   {func}`csp_scalespace.scalespace.two_stage_rhs`.

The reported `error` is the maximum absolute deviation from the full model over the times at which
the first `M` modes of the full model are exhausted (`study.on_manifold(M)`).

## Choosing the options

* `order: 1` (default) uses the curvature-corrected basis; use `order: 0` to reproduce classical
  zeroth-order CSP or when $(\mathbf g\cdot\nabla)\mathbf J$ is only available by finite differences
  and noisy.
* `time_scale: log` when the transients span several decades (three-scale network).
* `fiber_filter: rational` when Formulation B has to be integrated over many decades of $\sigma$:
  its rates per unit $\ln\sigma$ are bounded, whereas the Gaussian rates grow like $(\sigma/\tau_i)^2$.
* `levels` to restrict the analysis to the spectral gaps of interest.
