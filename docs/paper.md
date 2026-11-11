# Reproducing the paper

```bash
python paper/reproduce_paper_figures.py              # -> paper_figures/*.pdf and *.png
python paper/reproduce_paper_figures.py --only 6 12  # selected figure numbers
```

| figure | file(s) | function (module in `paper/`) |
|---|---|---|
| 1a, 1b | `fig_ds_intro_time`, `fig_ds_intro_phase` | `make_fig_ds_intro_time`, `make_ds_phase_space` (`fig_davis_skodje`) |
| 2 | `fig_geom2_filter_weights` | `make_fig2b_filter_weights` |
| 3a, 3b | `fig_geom1_isotropic`, `fig_geom1_anisotropic` | `make_fig1a_isotropic_euclidean`, `make_fig1b_anisotropic_csp` |
| 4 | `fig_geom2_duality_phase_space` | `make_fig2a_geometric_duality` |
| 5 | `fig_geom3_refinements_2d` | `make_fig3a_curvature_refinements` |
| 6a, 6b | `fig1a_phase_space`, `fig1b_asymptotic_convergence` | `generate_figure_1a`, `generate_figure_1b` |
| 7a, 7b | `fig_mm_intro_time`, `fig_mm_intro_phase` | `make_fig_mm_intro_time`, `make_fig_mm_intro_phase` (`fig_michaelis_menten`) |
| 8a, 8b | `fig_mm_adaptive_timescales`, `fig_mm_pointers_trajectory` | `make_fig_mm_adaptive_timescales`, `make_fig_mm_pointers_trajectory` |
| 9a-c | `fig_mm_pointers_t2`, `_t10`, `_t15` | `make_fig_mm_pointers_and_timescales` |
| 10a-f | `fig_mm_comparison_{sqssa,rqssa}_{adaptive,0.25,0.1}` | `make_fig_mm_model_comparison` |
| 11a, 11b | `fig2a_3d_continuous_pointers`, `fig2b_3d_fiber_relaxation` | `generate_figure_2a`, `generate_figure_2b` (`fig_three_scale`) |
| 12a-c | `fig_3d_time_comparison`, `fig_3d_dynamic_timescales`, `fig_3d_dynamic_pointers` | `plot_figure_1_time_comparison`, ... |
| 13 | `fig_3d_sim_trajectories` | `plot_figure_4_sim_trajectories_3d` |

Figures 3, 4 and 5 are schematic illustrations in the paper. Their computed counterparts for any
model are `04_isotropic_vs_anisotropic`, `05_geometric_duality` and `06_refinement_geometry` of the
{doc}`figure catalogue <figures>`.

PNG versions of all paper figures are stored in `docs/_static/paper/`.

```{figure} _static/paper/fig1a_phase_space.png
:width: 70%

Paper Fig. 6a.
```

## Numbers checked against the manuscript

`tests/test_paper_values.py`:

| quantity | paper | test |
|---|---|---|
| Stage-1 state, Davis-Skodje, $\epsilon=0.08$ | $(1.3847,\,0.5807)$ | `test_davis_skodje_stage1_state` |
| error orders of Fig. 6b | 1, 2, 2, 3 | `test_davis_skodje_asymptotic_orders` |
| Table 2 ($\tau_1,\tau_2,\sigma_c,D^c,D^s$) | four columns | `test_michaelis_menten_table2` |
| maximum error of the MM reduced model | $5.9\times10^{-3}$ | `test_michaelis_menten_reduced_model_error` |
| Eq. (104), cutoff scales, Table 3 plateaus | three-scale network | `test_three_scale_spectrum_and_plateaus` |
