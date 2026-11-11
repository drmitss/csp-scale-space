"""Example 2 - a model given as plain Python functions (no symbolic equations).

The Robertson chemical kinetics problem: three species, rate constants spanning
nine orders of magnitude and one conservation law (y1 + y2 + y3 = 1), hence one
zero eigenvalue.  The Jacobian is supplied analytically; the convective
derivative (g.grad)J needed by the first-order refinement is obtained by finite
differences automatically.

Note: a ``ComplexSpectrumWarning`` is printed once.  At the initial state
(1, 0, 0) two eigenvalues of the Jacobian coincide (both are zero), which
violates the strict timescale ordering assumed by the method (Assumption 3.3);
the package warns instead of failing silently.

Run:  python examples/02_model_from_python_functions.py
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np

import csp_scalespace as css


def rhs(y, p):
    r1, r2, r3 = p["k1"] * y[0], p["k2"] * y[1] ** 2, p["k3"] * y[1] * y[2]
    return np.array([-r1 + r3, r1 - r2 - r3, r2])


def jac(y, p):
    k1, k2, k3 = p["k1"], p["k2"], p["k3"]
    return np.array([[-k1, k3 * y[2], k3 * y[1]],
                     [k1, -2 * k2 * y[1] - k3 * y[2], -k3 * y[1]],
                     [0.0, 2 * k2 * y[1], 0.0]])


model = css.ODEModel(rhs, y0=[1.0, 0.0, 0.0], jac=jac, names=["y1", "y2", "y3"],
                     labels=["$y_1$", "$y_2$", "$y_3$"],
                     params={"k1": 0.04, "k2": 3.0e7, "k3": 1.0e4}, name="robertson",
                     description="Robertson kinetics (stiff, one conservation law)")

study = css.ScaleSpaceStudy(
    model, t_end=1.0e4, t_start=1.0e-6, time_scale="log", n_points=400,
    order=0, levels=[1],            # one exhausted mode: y2 is the quasi-steady species
    fiber_filter="rational", sigma_range=[1e-7, 1.0], snapshots=[1.0], log_states=True, state_floor=1e-12,
)
print("levels analysed:", study.levels)
print(study.summary())
files = study.make_figures("figures_example2", which=["applications", "time_evolution", "filter_weights"])
print(f"{len(files)} figures written to figures_example2/")
