# Installation

```bash
git clone https://github.com/drmitss/csp-scale-space.git
cd csp-scale-space
pip install -e .
```

| extra | command | adds |
|---|---|---|
| tests | `pip install -e ".[test]"` | pytest |
| documentation | `pip install -e ".[docs]"` | Sphinx, MyST, Furo |

Dependencies: Python ≥ 3.9, NumPy, SciPy, Matplotlib, SymPy, PyYAML.

Check the installation:

```bash
csp-scalespace models        # lists the three built-in models
pytest                       # about 20 s
```
