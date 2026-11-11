# CSP scale-space report: michaelis_menten

Michaelis-Menten kinetics, Case 3: sQSSA -> PEA -> rQSSA regime shift at t ~ 10 s.

- variables: c, s
- parameters: {'k1f': 1000.0, 'k1b': 0.01, 'k2': 0.1, 'e0': 10.0}
- initial state: [9.999999, 10.0]
- reduction levels M: [1]
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
| `10_pointers_vs_scale_t2.png` | applications | Fig. 9a-c / 11a | Scale-dependent fast/slow CSP pointers at the snapshot times. |
| `10_pointers_vs_scale_t10.png` | applications | Fig. 9a-c / 11a | Scale-dependent fast/slow CSP pointers at the snapshot times. |
| `10_pointers_vs_scale_t15.png` | applications | Fig. 9a-c / 11a | Scale-dependent fast/slow CSP pointers at the snapshot times. |
| `11_pointers_trajectory.png` | applications | Fig. 8b / 9d | Fast pointers evaluated at the adaptive cutoff scale along the trajectory. |
| `12_realtime_pointers.png` | applications | Fig. 12c | Sequential mode exhaustion: pointers with sigma = t and the number of exhausted modes. |
| `13_fiber_relaxation.png` | applications | Fig. 11b | Formulation B across all scales: plateaus of the relaxed state on the nested manifolds. |
| `14_model_comparison_adaptive.png` | applications | Fig. 10 / 12a | Scale-space reduced models vs. the full model (and reference reduced models). |
| `14_model_comparison_sigma0.25.png` | applications | Fig. 10 / 12a | Scale-space reduced models vs. the full model (and reference reduced models). |
| `14_model_comparison_sigma0.1.png` | applications | Fig. 10 / 12a | Scale-space reduced models vs. the full model (and reference reduced models). |

## Timescales, cutoff scales and pointers

| quantity | t = 2.001 | t = 10 | t = 15 |
|---|---|---|---|
| tau_1 | 1.250e-04 | 1.873e-02 | 2.542e-04 |
| tau_2 | 5.818e+05 | 1.927e+01 | 1.000e+01 |
| sigma_c (M=1) | 8.528e+00 | 6.008e-01 | 5.042e-02 |
| D^c(sigma_c, M=1) | 1.0000 | 0.4819 | 0.0000 |
| D^s(sigma_c, M=1) | 0.0000 | 0.5190 | 1.0000 |
| sum_k D^k (M=1) | 1.0000 | 1.0010 | 1.0000 |

## Accuracy of the scale-space reduced models

Maximum absolute deviation from the full model once the first M modes are exhausted.

| M | sigma | Stage-1 state | max error |
|---|---|---|---|
| 1 | adaptive | [9.99989, 10.00001] | 5.901e-03 |
| 1 | 0.25 | [9.99989, 10.00011] | 5.895e-03 |
| 1 | 0.1 | [9.99989, 10.00011] | 5.895e-03 |
