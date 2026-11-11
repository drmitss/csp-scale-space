#!/usr/bin/env python
"""Reproduce every figure of the paper.

Usage::

    python paper/reproduce_paper_figures.py                 # all figures -> paper_figures/
    python paper/reproduce_paper_figures.py --only 6 12     # selected figure numbers
    python paper/reproduce_paper_figures.py -o out --formats pdf --dpi 300

The table ``PAPER_FIGURES`` maps each figure/panel of the manuscript to the
function that draws it and to the output file name.
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).parent))

import _common  # noqa: E402
import fig_davis_skodje as ds  # noqa: E402
import fig_michaelis_menten as mm  # noqa: E402
import fig_three_scale as n3  # noqa: E402

_SIM = {}


def _fig13():
    if "data" not in _SIM:
        _SIM["data"] = n3.compute_3d_network_simulations()
    n3.plot_figure_4_sim_trajectories_3d(_SIM["data"])


#: (figure number, panel, callable, output file stem(s), description)
PAPER_FIGURES = [
    (1, "a", ds.make_fig_ds_intro_time, "fig_ds_intro_time", "Davis-Skodje: solution in time"),
    (1, "b", ds.make_ds_phase_space, "fig_ds_intro_phase", "Davis-Skodje: phase portrait and SIM"),
    (2, "", ds.make_fig2b_filter_weights, "fig_geom2_filter_weights", "Modal survival weights vs. scale"),
    (3, "a", ds.make_fig1a_isotropic_euclidean, "fig_geom1_isotropic", "Naive isotropic scale-space (schematic)"),
    (3, "b", ds.make_fig1b_anisotropic_csp, "fig_geom1_anisotropic", "Anisotropic CSP scale-space (schematic)"),
    (4, "", ds.make_fig2a_geometric_duality, "fig_geom2_duality_phase_space", "Formulation A vs. B (schematic)"),
    (5, "", ds.make_fig3a_curvature_refinements, "fig_geom3_refinements_2d", "B- and A-refinements (schematic)"),
    (6, "a", ds.generate_figure_1a, "fig1a_phase_space", "Two-stage filtration in phase space"),
    (6, "b", ds.generate_figure_1b, "fig1b_asymptotic_convergence", "Asymptotic convergence in epsilon"),
    (7, "a", mm.make_fig_mm_intro_time, "fig_mm_intro_time", "Michaelis-Menten: solution in time"),
    (7, "b", mm.make_fig_mm_intro_phase, "fig_mm_intro_phase", "Michaelis-Menten: phase portrait"),
    (8, "a", mm.make_fig_mm_adaptive_timescales, "fig_mm_adaptive_timescales", "Timescales and adaptive scale"),
    (8, "b", mm.make_fig_mm_pointers_trajectory, "fig_mm_pointers_trajectory", "Pointers at adaptive scale"),
    (9, "a-c", mm.make_fig_mm_pointers_and_timescales, "fig_mm_pointers_t{2,10,15}", "Pointers vs. scale"),
    (10, "a-b", lambda: mm.make_fig_mm_model_comparison("adaptive"), "fig_mm_comparison_*_adaptive",
     "Model comparison, adaptive sigma_c"),
    (10, "c-d", lambda: mm.make_fig_mm_model_comparison(0.25), "fig_mm_comparison_*_0.25",
     "Model comparison, sigma_c = 0.25 s"),
    (10, "e-f", lambda: mm.make_fig_mm_model_comparison(0.1), "fig_mm_comparison_*_0.1",
     "Model comparison, sigma_c = 0.1 s"),
    (11, "a", n3.generate_figure_2a, "fig2a_3d_continuous_pointers", "Three-scale network: pointers vs. scale"),
    (11, "b", n3.generate_figure_2b, "fig2b_3d_fiber_relaxation", "Three-scale network: fibre relaxation"),
    (12, "a", n3.plot_figure_1_time_comparison, "fig_3d_time_comparison", "Multi-resolution time comparison"),
    (12, "b", n3.plot_figure_2_dynamic_timescales, "fig_3d_dynamic_timescales", "Timescales along trajectory"),
    (12, "c", n3.plot_figure_3_dynamic_pointers, "fig_3d_dynamic_pointers", "Real-time pointers, sigma = t"),
    (13, "", _fig13, "fig_3d_sim_trajectories", "3D phase space: SIMs and scale-space flows"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--outdir", default="paper_figures", help="output directory")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"], help="file formats")
    ap.add_argument("--dpi", type=int, default=300, help="resolution of raster output")
    ap.add_argument("--only", nargs="+", type=int, help="figure numbers to produce (default: all)")
    args = ap.parse_args(argv)
    _common.OUTDIR, _common.FORMATS = Path(args.outdir), tuple(args.formats)
    import matplotlib.pyplot as plt

    plt.rcParams["savefig.dpi"] = args.dpi
    for num, panel, func, stem, desc in PAPER_FIGURES:
        if args.only and num not in args.only:
            continue
        t0 = time.time()
        func()
        plt.close("all")
        print(f"Fig. {num:>2}{panel:<4s} {desc:<45s} -> {stem}  ({time.time() - t0:.1f} s)")
    print(f"figures written to {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
