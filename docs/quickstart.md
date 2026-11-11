# Quick start

## Command line

```bash
csp-scalespace template my_model.yaml       # commented template
# ... edit equations, parameters, initial conditions, t_end ...
csp-scalespace run my_model.yaml -o figs --formats png pdf
```

The output directory contains the figures of the {doc}`catalogue <figures>` and `report.md` (figure
list, timescales, cutoff scales, pointers and reduced-model errors).

Other commands:

| command | effect |
|---|---|
| `csp-scalespace run davis_skodje` | run a built-in model (`davis_skodje`, `michaelis_menten`, `three_scale`) |
| `csp-scalespace run m.yaml --only theory` | only one group (`theory`, `applications`) or individual figure keys |
| `csp-scalespace run m.yaml --set eps=0.01` | override parameter values |
| `csp-scalespace figures` | list the figure catalogue |
| `csp-scalespace models` | list built-in models and the location of their YAML files |

## Python

```python
import csp_scalespace as css

model = css.ODEModel.from_equations(
    equations={"y1": "-y1",
               "y2": "-(y2 - y1/(1 + y1))/eps - y1/(1 + y1)**2"},
    parameters={"eps": 0.08},
    initial_conditions={"y1": 1.5, "y2": 0.2},
)
study = css.ScaleSpaceStudy(model, t_end=4.0)

study.basis0.tau              # timescales at the initial state        -> [0.08, 1.0]
study.sigma_c(1)              # adaptive cutoff scale along y(t)
study.stage1(1).final         # Stage 1: initial state relaxed onto the SIM
study.reduced(1)["y"]         # Stage 2: trajectory of the reduced model
study.make_figures("figs")    # everything
```

The scripts in `examples/` are complete, runnable versions:

| script | shows |
|---|---|
| `01_model_from_equations.py` | model from text equations, numbers and figures |
| `02_model_from_python_functions.py` | model from Python callables (Robertson kinetics, conservation law) |
| `03_low_level_operators.py` | each operator of the paper called directly |
