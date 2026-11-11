"""Built-in benchmark models of the paper.

==========================  ===========================================  ===============
function                    system                                       paper
==========================  ===========================================  ===============
:func:`davis_skodje`        2-D Davis-Skodje benchmark                   Sec. 3.4, Eq. 21
:func:`michaelis_menten`    Michaelis-Menten mechanism (Case 3)          Sec. 6.1, Eq. 90
:func:`three_scale`         3-D three-timescale reaction network         Sec. 6.2, Eq. 103
==========================  ===========================================  ===============

Each function returns an :class:`~csp_scalespace.model.ODEModel`; keyword
arguments override parameter values or initial conditions.  The same models
are stored as YAML model files next to this module and can be used as
templates for your own systems (``csp-scalespace models`` lists them).
"""
from __future__ import annotations

from pathlib import Path

from ..model import load_model_file

__all__ = ["davis_skodje", "michaelis_menten", "three_scale", "builtin_model_path", "load_builtin",
           "BUILTIN_MODELS"]

_HERE = Path(__file__).parent
BUILTIN_MODELS = ("davis_skodje", "michaelis_menten", "three_scale")


def builtin_model_path(name: str) -> Path:
    """Path of the YAML file of a built-in model."""
    path = _HERE / f"{name}.yaml"
    if not path.exists():
        raise KeyError(f"unknown built-in model '{name}'; available: {BUILTIN_MODELS}")
    return path


def load_builtin(name: str, y0=None, **params):
    """Return ``(model, settings)`` for a built-in model, optionally modified."""
    model, settings = load_model_file(builtin_model_path(name))
    if params:
        model = model.with_parameters(**params)
    if y0 is not None:
        model = model.with_initial_condition(y0)
    return model, settings


def davis_skodje(eps: float = 0.08, y0=(1.5, 0.2)):
    """Davis-Skodje system (Eq. 21) with stiffness ``eps`` = :math:`\\tau_1/\\tau_2`."""
    return load_builtin("davis_skodje", y0, eps=eps)[0]


def michaelis_menten(k1f: float = 1000.0, k1b: float = 0.01, k2: float = 0.1, e0: float = 10.0,
                     y0=(9.999999, 10.0)):
    """Michaelis-Menten system (Eq. 90); state ``[c, s]``. Defaults: Case 3 of the paper (Eq. 95)."""
    return load_builtin("michaelis_menten", y0, k1f=k1f, k1b=k1b, k2=k2, e0=e0)[0]


def three_scale(eps1: float = 1e-3, eps2: float = 5e-2, y0=(1.0, 0.78, 0.18)):
    """Three-scale reaction network (Eqs. 102-103)."""
    return load_builtin("three_scale", y0, eps1=eps1, eps2=eps2)[0]
