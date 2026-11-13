# Analysing your own model

This page is a checklist from "I have a set of ODEs" to "I have the figures and I trust them".

## 1. Write the model file

```bash
csp-scalespace template my_model.yaml
```

Edit four blocks: `parameters`, `equations`, `initial_conditions` and, under `settings`, the final
time `t_end`. Everything else is optional. A minimal file is

```yaml
parameters: {k1: 100.0, k2: 1.0}
equations:
  a: "-k1*a*b + k2*c"
  b: "-k1*a*b + k2*c"
  c: "k1*a*b - k2*c - 0.1*c"
initial_conditions: {a: 1.0, b: 0.5, c: 0.0}
settings: {t_end: 50.0}
```

Details and all optional blocks: {doc}`model_file`.

## 2. Run it

```bash
csp-scalespace run my_model.yaml -o figs
```

## 3. Read `figs/report.md` first

The report lists, in this order,

1. the variables, parameters and initial state that were actually used;
2. the **reduction levels** `M` that were analysed (number of exhausted modes);
3. every figure with its group and the figure of the paper it corresponds to;
4. a table of timescales $\tau_i$, cutoff scales $\sigma_{c,M}$ and fast pointers $D^k$ at the
   snapshot times;
5. the accuracy of each scale-space reduced model against the full model.

Three checks tell you whether the analysis is meaningful:

| check | where | what you want to see |
|---|---|---|
| spectral gaps exist | `09_timescales`, `03_filter_weights` | $\tau_M\ll\tau_{M+1}$ for the levels analysed; shaded SIM windows of visible width |
| pointers count the exhausted modes | row `sum_k D^k` of the report | close to the integer $M$ |
| the reduced model follows the full model | `14_model_comparison`, last table of the report | an error that is small compared with the states |

## 4. Adjust the settings

Typical first adjustments, all in the `settings:` block ({doc}`settings`):

| symptom | setting |
|---|---|
| the interesting transients are squeezed against the left axis | `time_scale: log` and `t_start: <small time>` |
| states span many decades | `log_states: true`, `state_floor: 1.0e-12` |
| the wrong variables are shown in phase portraits | `phase_variables: [a, c]` |
| only some gaps are physically relevant | `levels: [1]` |
| pointer snapshots at other times | `snapshots: [0.1, 5.0, 20.0]` |
| the fibre relaxation is slow or fails over many decades of $\sigma$ | `fiber_filter: rational` |
| the reduced model lags slightly behind the full model | `filter: exponential` with `p: 4` (see {doc}`faq`) |
| compare with a classical reduction (QSSA, PEA) | add a `reference_models:` block |

## 5. Go further in Python

Everything the figures show is available as arrays:

```python
import csp_scalespace as css

study = css.ScaleSpaceStudy.from_file("my_model.yaml")
t, Y = study.t, study.full                 # full trajectory, shape (N, T)
tau = study.diagnostics["tau"]             # timescales along it, shape (T, N)
D, S = study.pointers_at(study.sigma_c(1)) # pointers at the adaptive scale, shape (T, N)
red = study.reduced(1)                     # {"y0": ..., "y": ..., "error": ...}
```

and the operators can be applied at any state, see {doc}`quickstart` and {doc}`theory`.

## When the model is not a formula

If the right-hand side cannot be typed as text (tabulated data, a call to another library), build
the model from Python functions and pass it to the study directly:

```python
model = css.ODEModel(rhs, y0=[...], names=[...], params={...}, jac=my_jacobian)
study = css.ScaleSpaceStudy(model, t_end=..., order=0)
study.make_figures("figs")
```

Use `order=0` unless you can also supply `jac_dot`; the first-order refinement needs
$(\mathbf g\cdot\nabla)\mathbf J$, and its finite-difference approximation is less accurate than
the symbolic one. A complete example is `examples/02_model_from_python_functions.py`.
