"""Model definition: how a user hands an ODE system to the package.

The whole package works on autonomous systems

.. math::  \\frac{d\\mathbf{y}}{dt} = \\mathbf{g}(\\mathbf{y}), \\qquad \\mathbf{y}(0)=\\mathbf{y}_0

(Eq. 4 of the paper).  An :class:`ODEModel` bundles everything the CSP
scale-space machinery needs from such a system:

* the vector field :math:`\\mathbf{g}(\\mathbf{y})`,
* its Jacobian :math:`\\mathbf{J}(\\mathbf{y})` (Eq. 7),
* the convective derivative of the Jacobian
  :math:`\\dot{\\mathbf{J}} = (\\mathbf{g}\\cdot\\nabla)\\mathbf{J}` (Remark 5.2),
  needed only for the first-order CSP refinement,
* initial conditions, parameter values and variable names/labels.

There are three ways to build a model:

1. **From equations written as text** (recommended) with
   :meth:`ODEModel.from_equations`.  The equations are parsed with SymPy, so the
   Jacobian and :math:`(\\mathbf{g}\\cdot\\nabla)\\mathbf{J}` are exact.
2. **From a model file** (YAML or JSON) with :func:`load_model_file` -- the
   same information as (1) stored in a text file, plus analysis settings.
3. **From Python callables** with the plain constructor; missing derivatives
   are approximated by finite differences.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

__all__ = ["ODEModel", "load_model_file"]


def _fd_jacobian(g: Callable[[np.ndarray], np.ndarray], y: np.ndarray) -> np.ndarray:
    """Central finite-difference Jacobian of ``g`` at ``y``."""
    n = y.size
    J = np.empty((n, n))
    for j in range(n):
        h = np.cbrt(np.finfo(float).eps) * max(1.0, abs(y[j]))
        e = np.zeros(n)
        e[j] = h
        J[:, j] = (g(y + e) - g(y - e)) / (2.0 * h)
    return J


@dataclass
class ODEModel:
    """An autonomous ODE system :math:`\\dot{\\mathbf{y}} = \\mathbf{g}(\\mathbf{y})`.

    Parameters
    ----------
    rhs : callable
        ``rhs(y, p) -> array`` returning :math:`\\mathbf{g}(\\mathbf{y})`.  ``y``
        is a 1-D array of length ``N`` and ``p`` is the parameter dictionary
        :attr:`params`.
    y0 : array_like
        Initial condition :math:`\\mathbf{y}_0`.
    names : sequence of str, optional
        Variable names (default ``y1, y2, ...``).
    params : dict, optional
        Parameter values passed to ``rhs`` as ``p``.
    jac : callable, optional
        ``jac(y, p) -> (N, N) array``.  Finite differences are used if omitted.
    jac_dot : callable, optional
        ``jac_dot(y, p) -> (N, N) array`` returning
        :math:`(\\mathbf{g}\\cdot\\nabla)\\mathbf{J}`.  A directional finite
        difference of ``jac`` is used if omitted.
    labels : sequence of str, optional
        Matplotlib/LaTeX labels for the variables, e.g. ``r"$y^1$"``.
    name, description : str
        Free-text identification used in figure titles and reports.
    exact_sim : callable, optional
        ``exact_sim(y, p) -> y_on_sim``: projects a state onto the *exactly
        known* slow invariant manifold (keeping the slow coordinates).  Only
        needed for the asymptotic-convergence figure (paper Fig. 6b) and for
        drawing the exact SIM in phase portraits.

    Notes
    -----
    Use :meth:`from_equations` whenever the equations can be typed as text;
    it fills ``jac`` and ``jac_dot`` with exact symbolic derivatives.
    """

    rhs: Callable
    y0: Sequence[float]
    names: Sequence[str] | None = None
    params: dict = field(default_factory=dict)
    jac: Callable | None = None
    jac_dot: Callable | None = None
    labels: Sequence[str] | None = None
    name: str = "model"
    description: str = ""
    exact_sim: Callable | None = None
    _spec: dict | None = field(default=None, repr=False)

    def __post_init__(self):
        self.y0 = np.asarray(self.y0, dtype=float).copy()
        if self.names is None:
            self.names = [f"y{i + 1}" for i in range(self.n)]
        self.names = list(self.names)
        if self.labels is None:
            self.labels = [rf"${nm}$" for nm in self.names]
        self.labels = list(self.labels)
        if len(self.names) != self.n or len(self.labels) != self.n:
            raise ValueError("names/labels must have one entry per state variable")

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        """Number of state variables :math:`N`."""
        return int(np.size(self.y0))

    def g(self, y) -> np.ndarray:
        """Vector field :math:`\\mathbf{g}(\\mathbf{y})`."""
        return np.asarray(self.rhs(np.asarray(y, dtype=float), self.params), dtype=float).ravel()

    def J(self, y) -> np.ndarray:
        """Jacobian :math:`J^i_j = \\partial g^i/\\partial y^j` (Eq. 7)."""
        y = np.asarray(y, dtype=float)
        if self.jac is not None:
            return np.asarray(self.jac(y, self.params), dtype=float)
        return _fd_jacobian(self.g, y)

    def Jdot(self, y) -> np.ndarray:
        """Convective derivative :math:`(\\mathbf{g}\\cdot\\nabla)\\mathbf{J}` (Remark 5.2)."""
        y = np.asarray(y, dtype=float)
        if self.jac_dot is not None:
            return np.asarray(self.jac_dot(y, self.params), dtype=float)
        gv = self.g(y)
        norm = np.linalg.norm(gv)
        if norm == 0.0:
            return np.zeros((self.n, self.n))
        h = np.cbrt(np.finfo(float).eps) * max(1.0, np.linalg.norm(y))
        d = gv / norm
        return (self.J(y + h * d) - self.J(y - h * d)) / (2.0 * h) * norm

    def index(self, name: str) -> int:
        """Position of the variable called ``name``."""
        return self.names.index(name)

    def project_to_exact_sim(self, y) -> np.ndarray:
        """Project ``y`` onto the exact SIM (requires :attr:`exact_sim`)."""
        if self.exact_sim is None:
            raise ValueError(f"model '{self.name}' has no exact_sim")
        return np.asarray(self.exact_sim(np.asarray(y, dtype=float), self.params), dtype=float)

    # ----------------------------------------------------------- re-parametrise
    def with_parameters(self, **changes) -> "ODEModel":
        """Return a copy of the model with some parameter values replaced.

        Used e.g. to sweep the stiffness parameter :math:`\\epsilon` for the
        asymptotic-convergence figure.
        """
        unknown = set(changes) - set(self.params)
        if unknown:
            raise KeyError(f"unknown parameter(s): {sorted(unknown)}")
        if self._spec is not None:
            spec = dict(self._spec)
            spec["parameters"] = {**spec["parameters"], **changes}
            return ODEModel.from_equations(**spec)
        new = ODEModel(self.rhs, self.y0, self.names, {**self.params, **changes}, self.jac,
                       self.jac_dot, self.labels, self.name, self.description, self.exact_sim)
        return new

    def with_initial_condition(self, y0) -> "ODEModel":
        """Return a copy of the model with a different initial condition."""
        import copy

        new = copy.copy(self)
        new.y0 = np.asarray(y0, dtype=float).copy()
        if new._spec is not None:
            new._spec = {**new._spec, "initial_conditions": dict(zip(self.names, map(float, new.y0)))}
        return new

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_equations(
        cls,
        equations: Mapping[str, str],
        initial_conditions: Mapping[str, float],
        parameters: Mapping[str, float] | None = None,
        definitions: Mapping[str, str] | None = None,
        labels: Mapping[str, str] | None = None,
        exact_sim: Mapping[str, str] | None = None,
        name: str = "model",
        description: str = "",
    ) -> "ODEModel":
        """Build a model from equations given as text (parsed by SymPy).

        Parameters
        ----------
        equations : dict
            ``{variable: "right-hand side"}``, one entry per state variable, in
            the order the variables should appear in the state vector.  Python
            syntax is used (``**`` for powers; ``exp``, ``log``, ``sqrt``,
            ``sin`` ... are available).
        initial_conditions : dict
            ``{variable: value}``.
        parameters : dict, optional
            ``{parameter: value}`` for every constant used in the equations.
        definitions : dict, optional
            Intermediate quantities, e.g. reaction rates
            ``{"R1": "(y1*y2 - y3)/eps1"}``.  They may use variables,
            parameters and *earlier* definitions, and can be used in the
            equations.
        labels : dict, optional
            ``{variable: matplotlib label}``.
        exact_sim : dict, optional
            Exactly known SIM written as ``{fast variable: expression in the
            other variables}``, e.g. ``{"y2": "y1/(1+y1)"}``.
        name, description : str
            Identification strings.

        Examples
        --------
        >>> m = ODEModel.from_equations(
        ...     equations={"y1": "-y1",
        ...                "y2": "-(y2 - y1/(1+y1))/eps - y1/(1+y1)**2"},
        ...     parameters={"eps": 0.08},
        ...     initial_conditions={"y1": 1.5, "y2": 0.2})
        >>> m.J([1.0, 0.5]).shape
        (2, 2)
        """
        import sympy as sp

        parameters = dict(parameters or {})
        definitions = dict(definitions or {})
        names = list(equations)
        missing = [nm for nm in names if nm not in initial_conditions]
        if missing:
            raise ValueError(f"initial_conditions missing for: {missing}")
        clash = (set(names) & set(parameters)) | (set(names) & set(definitions))
        if clash:
            raise ValueError(f"name(s) used twice: {sorted(clash)}")

        ysym = [sp.Symbol(nm, real=True) for nm in names]
        psym = [sp.Symbol(nm, real=True) for nm in parameters]
        ns = {s.name: s for s in ysym + psym}
        for key, expr in definitions.items():
            ns[key] = sp.sympify(str(expr), locals=ns)

        def parse(expr):
            out = sp.sympify(str(expr), locals=ns)
            free = {s.name for s in out.free_symbols} - set(names) - set(parameters)
            if free:
                raise ValueError(
                    f"undefined symbol(s) {sorted(free)} in '{expr}'. Declare them under "
                    "'parameters' or 'definitions'.")
            return out

        gsym = sp.Matrix([parse(equations[nm]) for nm in names])
        Jsym = gsym.jacobian(ysym)
        Jdsym = sp.zeros(len(names), len(names))
        for k, yk in enumerate(ysym):
            Jdsym += gsym[k] * Jsym.diff(yk)

        args = ysym + psym
        pvals = [float(parameters[k]) for k in parameters]
        n = len(names)
        g_f = sp.lambdify(args, list(gsym), "numpy", cse=True)
        J_f = sp.lambdify(args, Jsym.tolist(), "numpy", cse=True)
        Jd_f = sp.lambdify(args, Jdsym.tolist(), "numpy", cse=True)

        def rhs(y, p):
            return np.array(g_f(*y, *pvals), dtype=float)

        def jac(y, p):
            return np.array(J_f(*y, *pvals), dtype=float).reshape(n, n)

        def jac_dot(y, p):
            return np.array(Jd_f(*y, *pvals), dtype=float).reshape(n, n)

        sim_fun = None
        if exact_sim:
            sim_idx = [names.index(k) for k in exact_sim]
            sim_f = sp.lambdify(args, [parse(v) for v in exact_sim.values()], "numpy")

            def sim_fun(y, p):
                out = np.array(y, dtype=float)
                out[sim_idx] = sim_f(*y, *pvals)
                return out

        spec = dict(equations=dict(equations), initial_conditions=dict(initial_conditions),
                    parameters=parameters, definitions=definitions,
                    labels=dict(labels) if labels else None,
                    exact_sim=dict(exact_sim) if exact_sim else None,
                    name=name, description=description)
        lab = [labels.get(nm, rf"${nm}$") for nm in names] if labels else None
        model = cls(rhs, [float(initial_conditions[nm]) for nm in names], names, parameters,
                    jac, jac_dot, lab, name, description, sim_fun, spec)
        model.symbolic = {"g": gsym, "J": Jsym, "Jdot": Jdsym, "variables": ysym, "parameters": psym}
        return model

    @classmethod
    def from_file(cls, path) -> "ODEModel":
        """Read a model from a YAML/JSON model file (see :func:`load_model_file`)."""
        return load_model_file(path)[0]


def _read_mapping(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


def load_model_file(path):
    """Load a model file and return ``(model, settings_dict)``.

    The file (YAML or JSON) has the following top-level keys; only
    ``equations`` and ``initial_conditions`` are mandatory:

    ``name``, ``description``
        Free text.
    ``equations``
        Mapping ``variable: right-hand side`` (see
        :meth:`ODEModel.from_equations`).
    ``parameters``, ``definitions``, ``initial_conditions``, ``labels``, ``exact_sim``
        As in :meth:`ODEModel.from_equations`.
    ``reference_models``
        Optional reduced models to overlay in the comparison figures, e.g.
        ``PEA: {equations: {c: "...", s: "..."}}`` (same variables/parameters).
    ``settings``
        Analysis and plotting options, see
        :class:`csp_scalespace.study.Settings`.

    Returns
    -------
    model : ODEModel
    settings : dict
        The ``settings`` block, with ``reference_models`` converted into
        callables under the key ``reference_models``.
    """
    cfg = _read_mapping(Path(path))
    for key in ("equations", "initial_conditions"):
        if key not in cfg:
            raise ValueError(f"{path}: missing mandatory key '{key}'")
    common = dict(parameters=cfg.get("parameters"), definitions=cfg.get("definitions"))
    model = ODEModel.from_equations(
        cfg["equations"], cfg["initial_conditions"], labels=cfg.get("labels"),
        exact_sim=cfg.get("exact_sim"), name=cfg.get("name", Path(path).stem),
        description=cfg.get("description", ""), **common)
    settings = dict(cfg.get("settings") or {})
    refs = {}
    for label, spec in (cfg.get("reference_models") or {}).items():
        ref = ODEModel.from_equations(spec["equations"], cfg["initial_conditions"],
                                      parameters=cfg.get("parameters"),
                                      definitions={**(cfg.get("definitions") or {}),
                                                   **(spec.get("definitions") or {})})
        refs[label] = (lambda r: (lambda y: r.g(y)))(ref)
    if refs:
        settings["reference_models"] = refs
    return model, settings
