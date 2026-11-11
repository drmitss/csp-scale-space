"""Example 1 - define a model with text equations and produce every figure.

Run:  python examples/01_model_from_equations.py
"""
import matplotlib

matplotlib.use("Agg")

import csp_scalespace as css

# 1. The model: equations, parameters and initial conditions.
#    (This is the Davis-Skodje system of the paper; replace it with your own.)
model = css.ODEModel.from_equations(
    equations={
        "y1": "-y1",
        "y2": "-(y2 - y1/(1 + y1))/eps - y1/(1 + y1)**2",
    },
    parameters={"eps": 0.08},
    initial_conditions={"y1": 1.5, "y2": 0.2},
    labels={"y1": "$y^1$", "y2": "$y^2$"},
    name="my_first_model",
)

# 2. The study: only the final time is mandatory.
study = css.ScaleSpaceStudy(model, t_end=4.0, order=1)

# 3. Numbers ...
print("timescales at y0      :", study.basis0.tau)
print("adaptive cutoff sigma_c:", study.sigma_c(1)[0])
print("Stage-1 relaxed state  :", study.stage1(1).final)
print("reduced-model error    :", study.reduced(1)["error"])
print(study.summary())

# 4. ... and figures (PNG files + report.md).
files = study.make_figures("figures_example1", formats=("png",))
print(f"{len(files)} figures written to figures_example1/")
