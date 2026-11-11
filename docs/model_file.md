# Describing a model

A model is an autonomous system $\dot{\mathbf y}=\mathbf g(\mathbf y)$, $\mathbf y(0)=\mathbf y_0$.

## Model file (YAML or JSON)

| key | required | content |
|---|---|---|
| `equations` | yes | `variable: "right-hand side"`; the order defines the state vector |
| `initial_conditions` | yes | `variable: value` |
| `parameters` | no | `name: value` for every constant in the equations |
| `definitions` | no | intermediate quantities (rates, fluxes); may use variables, parameters and earlier definitions |
| `labels` | no | Matplotlib labels, e.g. `y1: "$y^1$"` |
| `exact_sim` | no | analytically known SIM, `fast variable: "expression of the others"` |
| `reference_models` | no | reduced models to overlay: `LABEL: {equations: {...}, definitions: {...}}` |
| `settings` | `t_end` | analysis options, see {doc}`settings` |
| `name`, `description` | no | free text for the report |

Expressions are parsed by SymPy: use `**` for powers and the usual function names (`exp`, `log`,
`sqrt`, `sin`, `tanh`, `Max`, `Min`, ...). Every symbol must be a variable, a parameter or a
definition - otherwise an error names the undefined symbol.

Note for YAML: write floats in exponent form with a dot and a signed exponent (`1.0e-3`, not `1e-3`).

### Example: Michaelis-Menten (paper Section 6.1)

```{literalinclude} ../src/csp_scalespace/models/michaelis_menten.yaml
:language: yaml
```

### Template

```{literalinclude} ../src/csp_scalespace/models/template.yaml
:language: yaml
```

## Equations in Python

```python
model = css.ODEModel.from_equations(
    equations={...}, initial_conditions={...}, parameters={...},
    definitions={...}, labels={...}, exact_sim={...}, name="...", description="...")
```

The symbolic expressions are kept in `model.symbolic` (`g`, `J`, `Jdot`).

## Python functions

```python
model = css.ODEModel(rhs, y0, names=None, params={}, jac=None, jac_dot=None, labels=None, exact_sim=None)
```

`rhs(y, p)`, `jac(y, p)` and `jac_dot(y, p)` receive the state and the parameter dictionary.
`jac_dot` is the convective derivative $(\mathbf g\cdot\nabla)\mathbf J$ used by the first-order
refinement. Anything not supplied is computed by central finite differences.

## Changing parameters or initial conditions

```python
stiffer = model.with_parameters(eps=0.01)
other   = model.with_initial_condition([2.0, 0.1])
```
