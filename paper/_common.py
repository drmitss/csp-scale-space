"""Shared set-up of the scripts that reproduce the figures of the paper.

* publication style (identical to the original notebook),
* the three benchmark models, taken from :mod:`csp_scalespace.models`,
* thin wrappers that expose the library routines under the names used in the
  figure functions (``csp_decomp_mm``, ``net3d_twostage_scalespace_rhs`` ...),
* the closed-form Davis-Skodje expressions of the manuscript (Eqs. 23, 30, 76,
  77, 87, 88).  ``tests/test_paper_values.py`` checks that the general-purpose
  library reproduces these closed forms.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from csp_scalespace import ScaleFilter, csp_basis, two_stage_rhs
from csp_scalespace.models import davis_skodje, michaelis_menten, three_scale

OUTDIR = Path("paper_figures")
FORMATS = ("pdf", "png")

STYLE = {
    "font.family": "serif", "font.size": 15, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 11.0, "xtick.labelsize": 11, "ytick.labelsize": 11, "figure.dpi": 100,
    "savefig.dpi": 300, "savefig.bbox": "tight", "mathtext.fontset": "cm",
}
plt.rcParams.update(STYLE)


def save(fig, name, **kwargs):
    """Save ``fig`` as ``OUTDIR/<name>.<fmt>`` for every format in ``FORMATS``."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = Path(str(name)).stem
    kwargs.pop("dpi", None)
    for fmt in FORMATS:
        fig.savefig(OUTDIR / f"{stem}.{fmt}", **kwargs)


GAUSSIAN = ScaleFilter("gaussian")
RATIONAL = ScaleFilter("rational")

# --------------------------------------------------------------- Davis-Skodje
eps_ds = 0.08
_DS = davis_skodje(eps=eps_ds)


def ds_rhs(t, y):
    """Davis-Skodje vector field (Eq. 21) for eps = 0.08."""
    return _DS.g(y)


def g_ds(y, eps=eps_ds):
    y1, y2 = y
    return np.array([-y1, -(y2 - y1 / (1.0 + y1)) / eps - y1 / ((1.0 + y1) ** 2)])


def alpha_ds(y1, eps=eps_ds):
    """alpha(y1; eps) of Eq. (23)."""
    return (1.0 + y1 + eps * (y1 - 1.0)) / ((1.0 - eps) * ((1.0 + y1) ** 3))


def alpha1_ds(y1, eps=eps_ds):
    """Refined curvature function alpha^(1)(y1; eps) of Eqs. (76)-(77)."""
    num = (1.0 + eps) * ((1.0 + y1) ** 3) - 3.0 * (1.0 + y1 + eps * (y1 - 1.0)) * ((1.0 + y1) ** 2)
    return alpha_ds(y1, eps) + eps * y1 * num / ((1.0 - eps) * ((1.0 + y1) ** 6))


def ds_fiber_rhs(sigma, y, eps=eps_ds, refined=True):
    """Closed-form Formulation B fibre ODE, Eq. (51) / (88)."""
    y1, y2 = y
    g1, g2 = g_ds(y, eps)
    alp = alpha1_ds(y1, eps) if refined else alpha_ds(y1, eps)
    c1 = (2.0 * sigma / eps) * (-alp * g1 + g2)
    c2 = (2.0 * sigma / 1.0) * g1
    return [c2, c1 + c2 * alp]


# ----------------------------------------------------------- Michaelis-Menten
_MM = michaelis_menten()
k1f, k1b, k2, e0 = (_MM.params[k] for k in ("k1f", "k1b", "k2", "e0"))
KR, K = k1b / k1f, k2 / k1f
KM = KR + K
y0_mm = _MM.y0.copy()


def mm_rhs(t, y):
    """Michaelis-Menten vector field (Eq. 90), state [c, s]."""
    return _MM.g(y)


def csp_decomp_mm(y):
    """Eigenvalues, timescales and zeroth-order CSP basis (fastest mode first)."""
    b = csp_basis(_MM, y)
    return b.lam, b.tau, b.A, b.B


def pea_rhs(t, y):
    """PEA reduced model, Eq. (94)."""
    c, s = y
    nu = (e0 - c) / (KR + max(s, 0.0))
    return np.array([-(nu / (1.0 + nu)) * k2 * c, -(1.0 / (1.0 + nu)) * k2 * c])


_MM_RHS = {}


def mm_twostage_scalespace_rhs(t, y, sigma_c="adaptive"):
    """Unified two-stage scale-space model (Eq. 50) at adaptive or fixed sigma_c."""
    if sigma_c not in _MM_RHS:
        _MM_RHS[sigma_c] = two_stage_rhs(_MM, 1, sigma_c, GAUSSIAN, order=0, stabilize=True,
                                         tau_relax_max=0.10 if sigma_c == "adaptive" else None)
    return _MM_RHS[sigma_c](t, y)


# ------------------------------------------------------- three-scale network
_N3 = three_scale()
eps1, eps2 = _N3.params["eps1"], _N3.params["eps2"]
y0_3d = _N3.y0.copy()


def g_rev(t, y):
    """Three-scale network vector field (Eq. 103)."""
    return _N3.g(y)


def jac_rev(y):
    return _N3.J(y)


def csp_decomp_3d(y):
    b = csp_basis(_N3, y)
    return b.lam, b.tau, b.A, b.B


def net3d_fiber_log_rhs(u, y_s):
    """Formulation B (Eq. 49) in u = ln(sigma) with the rational filter (Eq. 37)."""
    b = csp_basis(_N3, y_s)
    coef = RATIONAL.log_rates(np.exp(u), b.tau) * b.tau
    return b.A @ (coef * b.amplitudes(_N3.g(y_s)))


_N3_RHS = {M: two_stage_rhs(_N3, M, "adaptive", GAUSSIAN, order=0, stabilize=True) for M in (1, 2)}


def net3d_twostage_scalespace_rhs(t, y, level=1):
    """Two-stage scale-space model at sigma_{c,level}(y) (level = number of exhausted modes)."""
    return _N3_RHS[level](t, y)


def shade_regimes(ax):
    """Background shading of the three dynamical regimes of the three-scale network (Fig. 12)."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

    spans = [(1e-5, 1.3e-3, "mistyrose", "#d9e8f5"), (1.3e-3, 1e-1, "#d9e8f5", "#e2f0d9"),
             (1e-1, 9.0, "#e2f0d9", "lightgrey")]
    for i, (x_min, x_max, c_start, c_end) in enumerate(spans):
        a, b = to_rgba(c_start, alpha=0.45), to_rgba(c_end, alpha=0.45)
        cmap = LinearSegmentedColormap.from_list(f"fade_span_{i}", [(0.0, a), (0.85, a), (1.0, b)])
        x = np.geomspace(x_min, x_max, 1000)
        X, Y = np.meshgrid(x, np.array([0, 1]))
        Z = np.linspace(0, 1, len(x) - 1).reshape(1, -1)
        ax.pcolormesh(X, Y, Z, cmap=cmap, shading="flat", transform=ax.get_xaxis_transform(), zorder=0)
    ax.axvspan(9.0, 20.0, color="lightgrey", alpha=0.45, zorder=0)
