# Figure catalogue

`study.make_figures(outdir, formats=("png",), which=None)` or `csp-scalespace run` produce the figures below for any model. `which`/`--only` accepts figure keys (with or without the number prefix) and the group names `theory` and `applications`.

Complete sets for the three built-in models are in `docs/_static/gallery/<model>/` together with their `report.md`.

## Theory group

### `01_time_evolution`

*Paper: Fig. 1a / 7a. Function:* {func}`csp_scalespace.figures.fig_time_evolution`

Time histories of all state variables. On a linear time axis the fast initial layer $t\lesssim3\tau_1(\mathbf y_0)$ is shaded; if the model has an `exact_sim`, its value is overlaid.

```{figure} _static/gallery/davis_skodje/01_time_evolution.png
:width: 85%

`01_time_evolution` for the built-in model `davis_skodje`.
```

### `02_phase_portrait`

*Paper: Fig. 1b / 7b. Function:* {func}`csp_scalespace.figures.fig_phase_portrait`

Reference trajectory (red), additional off-manifold trajectories (blue) and the slow invariant manifold, drawn as the part of the reference trajectory on which the fast modes are exhausted (green band) and, when available, the exact SIM (black). Grey dashes: directions of the fast basis vector $\mathbf a_1$.

```{figure} _static/gallery/davis_skodje/02_phase_portrait.png
:width: 85%

`02_phase_portrait` for the built-in model `davis_skodje`.
```

### `03_filter_weights`

*Paper: Fig. 2. Function:* {func}`csp_scalespace.figures.fig_filter_weights`

Survival weights $w_i(\sigma)$ of every mode at the initial state, the sharp CSP cutoff, the timescales $\tau_i$ (dotted), the cutoff scales $\sigma_{c,M}$ (dash-dotted) and the SIM windows $3\tau_M<\sigma<\tau_{M+1}/3$ (shaded).

```{figure} _static/gallery/three_scale/03_filter_weights.png
:width: 85%

`03_filter_weights` for the built-in model `three_scale`.
```

### `04_isotropic_vs_anisotropic`

*Paper: Fig. 3. Function:* {func}`csp_scalespace.figures.fig_isotropic_vs_anisotropic`

Left: the trajectory convolved in time with a Gaussian kernel (`smoothing_width`) leaves the manifold. Right: states perturbed transversally to the flow (`perturbation`) are returned to the manifold by Formulation B, each along its fast fibre.

```{figure} _static/gallery/davis_skodje/04_isotropic_vs_anisotropic.png
:width: 85%

`04_isotropic_vs_anisotropic` for the built-in model `davis_skodje`.
```

### `05_geometric_duality`

*Paper: Fig. 4. Function:* {func}`csp_scalespace.figures.fig_geometric_duality`

At the initial state: the velocity $\mathbf g$, its half-filtered and fully filtered versions (Formulation A; arrow lengths compressed with exponent 0.4), the continuous fibre path of Formulation B, the one-step classical CSP projection, the flow obtained by filtering without relaxing, and the Stage 2 flow.

```{figure} _static/gallery/davis_skodje/05_geometric_duality.png
:width: 85%

`05_geometric_duality` for the built-in model `davis_skodje`.
```

### `06_refinement_geometry`

*Paper: Fig. 5. Function:* {func}`csp_scalespace.figures.fig_refinement_geometry`

Left: eigenvector basis (dashed) and first-order refined basis (solid) at a point of the SIM, with the manifolds $f^{1,(0)}=0$ and $f^{1,(1)}=0$ (two-dimensional models). Right: the fraction of the on-manifold velocity that each basis wrongly assigns to the fast subspace, $\|\mathbf P^{(k)}_{fast}\mathbf g\|/\|\mathbf g\|$; it is $O(\epsilon)$ for $k=0$ and $O(\epsilon^2)$ for $k=1$.

```{figure} _static/gallery/davis_skodje/06_refinement_geometry.png
:width: 85%

`06_refinement_geometry` for the built-in model `davis_skodje`.
```

### `07_two_stage`

*Paper: Fig. 6a / 13. Function:* {func}`csp_scalespace.figures.fig_two_stage`

Full trajectory, Stage 1 fibre relaxation and Stage 2 flow for every level $M$; for two-dimensional models also the CSP manifolds and the unrelaxed Formulation A flow, for three or more variables a 3-D view.

```{figure} _static/gallery/three_scale/07_two_stage.png
:width: 85%

`07_two_stage` for the built-in model `three_scale`.
```

### `08_refinement_convergence`

*Paper: Fig. 6b. Function:* {func}`csp_scalespace.figures.fig_refinement_convergence`

For a sweep of `gap_parameter`: velocity error $\|\mathbf P_{fast}^{(k)}\mathbf g\|$ and manifold-position error of the zeroth- and first-order bases at a point of the true SIM, with fitted slopes (expected 1, 2 and 2, 3).

```{figure} _static/gallery/davis_skodje/08_refinement_convergence.png
:width: 85%

`08_refinement_convergence` for the built-in model `davis_skodje`.
```

## Applications group

### `09_timescales`

*Paper: Fig. 8a / 12b. Function:* {func}`csp_scalespace.figures.fig_timescales`

Timescales $\tau_i(\mathbf y(t))$ and the adaptive cutoff scales $\sigma_{c,M}(t)$ that follow the centre of each spectral gap.

```{figure} _static/gallery/michaelis_menten/09_timescales.png
:width: 85%

`09_timescales` for the built-in model `michaelis_menten`.
```

### `10_pointers_vs_scale`

*Paper: Fig. 9a-c / 11a. Function:* {func}`csp_scalespace.figures.fig_pointers_vs_scale`

At each snapshot time: fast pointers $D^k(\sigma)$ (solid) and slow pointers $S^k(\sigma)$ (dashed) of every variable. Inside a SIM window the fast pointers identify the slaved variables.

```{figure} _static/gallery/three_scale/10_pointers_vs_scale_t1e-05.png
:width: 85%

`10_pointers_vs_scale` for the built-in model `three_scale`.
```

### `11_pointers_trajectory`

*Paper: Fig. 8b / 9d. Function:* {func}`csp_scalespace.figures.fig_pointers_trajectory`

Fast pointers evaluated at $\sigma_{c,M}(t)$ along the trajectory, one panel per level; their sum equals $M$. Changes of the pointers in time reveal changes of the dynamical role of the variables.

```{figure} _static/gallery/michaelis_menten/11_pointers_trajectory.png
:width: 85%

`11_pointers_trajectory` for the built-in model `michaelis_menten`.
```

### `12_realtime_pointers`

*Paper: Fig. 12c. Function:* {func}`csp_scalespace.figures.fig_realtime_pointers`

Pointers with the scale tied to the elapsed time, $\sigma=t$: the sum counts the modes exhausted so far.

```{figure} _static/gallery/three_scale/12_realtime_pointers.png
:width: 85%

`12_realtime_pointers` for the built-in model `three_scale`.
```

### `13_fiber_relaxation`

*Paper: Fig. 11b. Function:* {func}`csp_scalespace.figures.fig_fiber_relaxation`

The initial state relaxed by Formulation B across the whole scale axis; each plateau is the point of one of the nested manifolds.

```{figure} _static/gallery/three_scale/13_fiber_relaxation.png
:width: 85%

`13_fiber_relaxation` for the built-in model `three_scale`.
```

### `14_model_comparison`

*Paper: Fig. 10 / 12a. Function:* {func}`csp_scalespace.figures.fig_model_comparison`

Full model (solid), scale-space reduced models for every level (dashed, dotted, ...) and reference models (markers). One figure for the adaptive scale and one per entry of `fixed_sigmas`.

```{figure} _static/gallery/three_scale/14_model_comparison_adaptive.png
:width: 85%

`14_model_comparison` for the built-in model `three_scale`.
```
