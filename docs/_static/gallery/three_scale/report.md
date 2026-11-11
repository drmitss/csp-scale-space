# CSP scale-space report: three_scale

3D three-timescale network: sequential exhaustion of two fast modes (2D SIM, then 1D SIM).

- variables: y1, y2, y3
- parameters: {'eps1': 0.001, 'eps2': 0.05}
- initial state: [1.0, 0.78, 0.18]
- reduction levels M: [1, 2]
- filter: gaussian (p = 2), CSP basis order: 0

## Figures

| file | group | paper figure | content |
|---|---|---|---|
| `01_time_evolution.png` | theory | Fig. 1a / 7a | State variables in time; the fast initial layer (t < 3 tau_1) is shaded. |
| `02_phase_portrait.png` | theory | Fig. 1b / 7b | Phase portrait with off-manifold trajectories collapsing onto the slow invariant manifold. |
| `03_filter_weights.png` | theory | Fig. 2 | Modal survival weights w_i(sigma) at the initial state, timescales and cutoff scales. |
| `04_isotropic_vs_anisotropic.png` | theory | Fig. 3 | Naive Gaussian smoothing of the trajectory (left) against anisotropic fibre relaxation of perturbed states (right). |
| `05_geometric_duality.png` | theory | Fig. 4 | Formulation A (velocity filtering at fixed state) vs. Formulation B (state relaxation along fibres) at the initial state. Arrow lengths are compressed (power 0.4) for visibility. |
| `06_refinement_geometry.png` | theory | Fig. 5 | CSP basis vectors before/after the first-order B- and A-refinements and the resulting tangency error along the trajectory. |
| `07_two_stage.png` | theory | Fig. 6a / 13 | Unified two-stage operator: Stage 1 fibre relaxation followed by the Stage 2 tangent flow. |
| `09_timescales.png` | applications | Fig. 8a / 12b | Local timescales tau_i(y(t)) and adaptive cutoff scales sigma_c(t) along the trajectory. |
| `10_pointers_vs_scale_t1e-05.png` | applications | Fig. 9a-c / 11a | Scale-dependent fast/slow CSP pointers at the snapshot times. |
| `11_pointers_trajectory.png` | applications | Fig. 8b / 9d | Fast pointers evaluated at the adaptive cutoff scale along the trajectory. |
| `12_realtime_pointers.png` | applications | Fig. 12c | Sequential mode exhaustion: pointers with sigma = t and the number of exhausted modes. |
| `13_fiber_relaxation.png` | applications | Fig. 11b | Formulation B across all scales: plateaus of the relaxed state on the nested manifolds. |
| `14_model_comparison_adaptive.png` | applications | Fig. 10 / 12a | Scale-space reduced models vs. the full model (and reference reduced models). |

## Timescales, cutoff scales and pointers

| quantity | t = 1e-05 |
|---|---|
| tau_1 | 5.234e-04 |
| tau_2 | 4.129e-02 |
| tau_3 | 3.352e+00 |
| sigma_c (M=1) | 4.649e-03 |
| D^y1(sigma_c, M=1) | 0.1653 |
| D^y2(sigma_c, M=1) | 0.3239 |
| D^y3(sigma_c, M=1) | 0.5234 |
| sum_k D^k (M=1) | 1.0126 |
| sigma_c (M=2) | 3.720e-01 |
| D^y1(sigma_c, M=2) | 0.5701 |
| D^y2(sigma_c, M=2) | 0.8887 |
| D^y3(sigma_c, M=2) | 0.5535 |
| sum_k D^k (M=2) | 2.0122 |

## Accuracy of the scale-space reduced models

Maximum absolute deviation from the full model once the first M modes are exhausted.

| M | sigma | Stage-1 state | max error |
|---|---|---|---|
| 1 | adaptive | [0.87389, 0.58334, 0.50274] | 5.828e-03 |
| 2 | adaptive | [0.96032, 0.49766, 0.47809] | 1.105e-02 |
