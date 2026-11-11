# Settings

Settings are given in the `settings:` block of a model file or as keyword arguments of
{class}`csp_scalespace.study.ScaleSpaceStudy`. Unknown names raise an error listing the valid ones.

```{eval-rst}
.. autoclass:: csp_scalespace.study.Settings
   :no-members:
   :no-index:
```

| setting | default | meaning |
|---|---|---|
| `t_end` | required | final time |
| `n_points` | 800 | number of output times |
| `time_scale`, `t_start` | `linear`, 0 | `log`: logarithmic output grid starting at `t_start` |
| `time_unit` | `""` | unit shown on axes |
| `rtol`, `atol` | 1e-10, 1e-12 | tolerances of the Radau integrator |
| `filter`, `p` | `gaussian`, 2 | scale filter and exponent |
| `fiber_filter` | `filter` | filter of Formulation B |
| `order` | 1 | CSP basis order (0 or 1) |
| `levels` | all gaps | list of numbers `M` of exhausted modes |
| `fixed_sigmas` | `[]` | constant scales for additional reduced models |
| `stabilize` | `true` | continuous Stage-1 term in the reduced model |
| `tau_relax_max` | none | cap of the stabilisation time |
| `snapshots` | `[0]` | times of `10_pointers_vs_scale` |
| `sigma_range` | automatic | `[min, max]` of the scale axis |
| `phase_variables` | first 2-3 | variables of phase-space figures (two, or three for a 3-D view) |
| `phase_log` | `false` | logarithmic phase-plane axes |
| `extra_initial_conditions` | automatic | further initial states for `02_phase_portrait` |
| `extra_t_end` | `t_end` | integration time of those trajectories |
| `perturbation` | 0.12 | size of the perturbations in `04`, relative to the range of each variable |
| `smoothing_width` | half the slow timescale | kernel width (time) of the naive smoothing in `04` |
| `log_states`, `state_floor` | `false`, none | logarithmic state axes and lower clip value |
| `gap_parameter` | none | parameter controlling the spectral gap (enables `08`) |
| `gap_values` | none | list, or `{start, stop, num, scale}` (exponents when `scale: log`) |
| `gap_reference_state` / `gap_reference_time` | initial state / `t_end/4` | where the errors of `08` are measured |
| `reference_models` | `{}` | `{label: g(y)}`; from a model file use the `reference_models` block |
