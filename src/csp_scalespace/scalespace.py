"""The CSP scale-space operators.

Everything in this module is a direct transcription of the formulas of the
paper into functions that act on an :class:`~csp_scalespace.model.ODEModel`:

=====================================  =========================================  ==========
function                               object                                     paper
=====================================  =========================================  ==========
:func:`cutoff_scale`                   :math:`\\sigma_c=\\sqrt{\\tau_M\\tau_{M+1}}`      Thm 4.2, Eq. 97
:func:`active_dimension`               :math:`M(\\sigma)`                           Eq. 45
:func:`scale_space_operator`           :math:`\\mathcal S_\\sigma=\\mathbf{AWB}`        Eq. 43 / 82
:func:`diffusion_tensor`               :math:`\\mathcal D_\\sigma=\\mathbf{ADB}`        Eq. 44
:func:`pointers`                       :math:`D^k(\\sigma),\\,S^k(\\sigma)`            Eqs. 56-57
:func:`filtered_field`                 Formulation A, :math:`\\mathbf g_\\sigma`       Eq. 42 / 82
:func:`fiber_relaxation`               Formulation B, :math:`\\mathbf y_\\sigma`       Eq. 49 / 83
:func:`radical_correction`             classical CSP projection onto the SIM       Sec. 4.3
:func:`two_stage_rhs`                  unified two-stage reduced model             Eq. 50
:func:`integrate`                      stiff time integration helper
=====================================  =========================================  ==========
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .csp import CSPBasis, csp_basis
from .filters import ScaleFilter, as_filter

__all__ = [
    "cutoff_scale", "active_dimension", "scale_space_operator", "diffusion_tensor", "pointers",
    "filtered_field", "FiberPath", "fiber_relaxation", "radical_correction", "two_stage_rhs",
    "integrate",
]


# --------------------------------------------------------------------- scales
def cutoff_scale(tau, M: int) -> float:
    """State-adaptive cutoff scale :math:`\\sigma_{c,M}=\\sqrt{\\tau_M\\,\\tau_{M+1}}`.

    This is the geometric mean of the slowest *fast* timescale and the fastest
    *slow* timescale, i.e. the centre (on a log axis) of the spectral gap that
    separates ``M`` exhausted modes from the rest (Theorem 4.2, Eq. 97).

    Parameters
    ----------
    tau : (..., N) array
        Timescales ordered fastest first.
    M : int
        Number of exhausted modes, ``1 <= M <= N-1``.
    """
    tau = np.asarray(tau, dtype=float)
    if not 1 <= M <= tau.shape[-1] - 1:
        raise ValueError(f"M must satisfy 1 <= M <= N-1, got {M}")
    return np.sqrt(tau[..., M - 1] * tau[..., M])


def active_dimension(tau, sigma) -> int:
    """Number of modes faster than the scale: :math:`M(\\sigma)=\\#\\{i:\\tau_i<\\sigma\\}`."""
    return int(np.sum(np.asarray(tau) < sigma))


# ------------------------------------------------------------------ operators
def scale_space_operator(basis: CSPBasis, sigma, filt=None) -> np.ndarray:
    """Scale-space operator :math:`\\mathcal S_\\sigma(\\mathbf y)=\\mathbf{A\\,W\\,B}` (Eq. 43).

    Multiplying the vector field by this matrix attenuates each CSP mode by
    its survival weight.  With the sharp filter it reduces to the classical
    CSP slow projector (Eq. 45).
    """
    w = as_filter(filt).weights(sigma, basis.tau)
    return (basis.A * w) @ basis.B


def diffusion_tensor(basis: CSPBasis, sigma, filt=None) -> np.ndarray:
    """Anisotropic spectral diffusion tensor :math:`\\mathcal D_\\sigma=\\mathbf{A\\,D\\,B}` (Eq. 44)."""
    d = as_filter(filt).rates(sigma, basis.tau)
    return (basis.A * d) @ basis.B


def pointers(basis: CSPBasis, sigma, filt=None):
    """Scale-dependent fast and slow CSP pointers (Eqs. 56-57).

    Parameters
    ----------
    basis : CSPBasis
    sigma : float or (S,) array
        Observation scale(s).
    filt : filter specification, optional

    Returns
    -------
    D, S : arrays of shape ``(N,)`` or ``(S, N)``
        ``D[..., k]`` is the fast pointer of variable ``k`` (how much of the
        variable has been exhausted at scale ``sigma``) and ``S[..., k] =
        1 - D[..., k]`` its slow pointer.  ``D.sum(-1)`` counts the exhausted
        modes.
    """
    sigma = np.asarray(sigma, dtype=float)
    w = as_filter(filt).weights(sigma[..., None], basis.tau)  # (..., N modes)
    S = w @ basis.mode_pointers()
    return 1.0 - S, S


def filtered_field(model, y, sigma, filt=None, order: int = 0, M: int | None = None) -> np.ndarray:
    """Formulation A: filtered vector field :math:`\\mathbf g_\\sigma=\\mathcal S_\\sigma\\mathbf g` (Eqs. 42, 82).

    The state is held fixed; only the velocity is filtered.

    Parameters
    ----------
    model : ODEModel
    y : array_like
    sigma : float
    filt : filter specification, optional
    order : {0, 1}
        CSP basis order.
    M : int, optional
        Fast dimension used by the first-order refinement.  Defaults to
        :math:`M(\\sigma)`.
    """
    basis = _basis_for(model, y, sigma, order, M)
    return scale_space_operator(basis, sigma, filt) @ model.g(y)


def _basis_for(model, y, sigma, order, M):
    if order == 0:
        return csp_basis(model, y)
    basis0 = csp_basis(model, y)
    if M is None:
        M = active_dimension(basis0.tau, sigma)
    from .csp import refine_basis

    return refine_basis(basis0, model.Jdot(y), M)


# ------------------------------------------------------------- Formulation B
@dataclass
class FiberPath:
    """Result of :func:`fiber_relaxation`: the homotopy :math:`\\sigma\\mapsto\\mathbf y_\\sigma`.

    Attributes
    ----------
    sigma : (S,) array
        Scales at which the state was stored.
    y : (N, S) array
        Relaxed states :math:`\\mathbf y_\\sigma`.
    """

    sigma: np.ndarray
    y: np.ndarray
    _sol: object = None
    _log: bool = True

    def at(self, sigma) -> np.ndarray:
        """Relaxed state at an arbitrary scale inside the integrated range."""
        s = np.log(sigma) if self._log else sigma
        return self._sol.sol(s)

    @property
    def final(self) -> np.ndarray:
        """State at the largest scale."""
        return self.y[:, -1]


def fiber_relaxation(model, y0, sigma_max, *, sigma_min=None, n: int = 400, filt=None,
                     order: int = 0, M: int | None = None, param: str = "log",
                     rtol: float = 1e-10, atol: float = 1e-12) -> FiberPath:
    """Formulation B: relax a state along the fast fibres (Eqs. 49, 83).

    Integrates the scale-space ODE

    .. math::
       \\frac{\\partial\\mathbf y_\\sigma}{\\partial\\sigma}
       = -\\sum_i \\mathbf a_i(\\mathbf y_\\sigma)\\,
         \\frac{d_i(\\mathbf y_\\sigma;\\sigma)}{\\lambda_i(\\mathbf y_\\sigma)}\\,f^i(\\mathbf y_\\sigma),
       \\qquad \\mathbf y_{\\sigma=0}=\\mathbf y_0 ,

    re-evaluating the basis, eigenvalues and amplitudes at every step.  As
    :math:`\\sigma` passes the timescale of a mode, the state slides along the
    corresponding fibre until that mode's amplitude vanishes: the result is
    the point of the :math:`\\sigma`-scale slow invariant manifold "below"
    :math:`\\mathbf y_0`.  Only dissipative modes (:math:`\\lambda_i<0`) are relaxed.

    Parameters
    ----------
    model : ODEModel
    y0 : array_like
        Starting (possibly off-manifold) state.
    sigma_max : float or "adaptive"
        Target scale.  With ``"adaptive"`` (requires ``M``) the integration
        stops at the self-consistent cutoff scale, i.e. at the first scale
        for which :math:`\\sigma=\\sigma_{c,M}(\\mathbf y_\\sigma)`: the centre of
        the ``M``-th spectral gap *evaluated at the relaxed state*.
    sigma_min : float, optional
        Scale at which the integration starts when ``param="log"``; default
        ``1e-3 * tau_1(y0)`` (where the state has not moved yet).
    n : int
        Number of stored points.
    filt : filter specification, optional
        Must be a smooth filter.
    order : {0, 1}
        CSP basis order.
    M : int, optional
        Number of fast modes for the refinement when ``order=1``; defaults to
        :math:`M(\\sigma_{max})` evaluated at ``y0``.
    param : {"log", "linear"}
        Integrate in :math:`u=\\ln\\sigma` (robust over many decades) or
        directly in :math:`\\sigma` from 0.
    rtol, atol : float
        Tolerances of the stiff integrator (Radau).

    Returns
    -------
    FiberPath
    """
    filt = as_filter(filt)
    y0 = np.asarray(y0, dtype=float)
    tau0 = csp_basis(model, y0).tau
    adaptive = isinstance(sigma_max, str)
    if adaptive:
        if M is None or param != "log":
            raise ValueError("sigma_max='adaptive' needs M and param='log'")
        finite = tau0[np.isfinite(tau0)]
        sigma_max = 1e3 * finite.max()
    if order == 1 and M is None:
        M = active_dimension(tau0, sigma_max)
    log = param == "log"

    def rhs(s, y):
        sigma = np.exp(s) if log else s
        basis = csp_basis(model, y, order, M)
        f = basis.amplitudes(model.g(y))
        coef = filt.log_rates(sigma, basis.tau) if log else filt.rates(sigma, basis.tau)
        # relax dissipative modes only (not conserved, lambda = 0, or growing ones)
        active = basis.lam < -1e-12 * np.abs(basis.lam).max()
        coef = np.where(active, coef, 0.0) * np.where(active, basis.tau, 0.0)
        return basis.A @ (coef * f)

    if log:
        if sigma_min is None:
            sigma_min = 1e-3 * np.min(tau0[np.isfinite(tau0)])
        sigma_min = min(sigma_min, 0.5 * sigma_max)
        span = (np.log(sigma_min), np.log(sigma_max))
    else:
        span = (0.0, float(sigma_max))
    events = None
    if adaptive:
        def reached(s, y):
            return s - np.log(cutoff_scale(csp_basis(model, y).tau, M))

        reached.terminal, reached.direction = True, 1.0
        events = reached
    sol = solve_ivp(rhs, span, y0, method="Radau", dense_output=True, events=events,
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"fiber relaxation failed: {sol.message}")
    grid = np.linspace(span[0], sol.t[-1], n)
    return FiberPath(np.exp(grid) if log else grid, sol.sol(grid), sol, log)


def radical_correction(model, y, M: int, order: int = 0, iterations: int = 1) -> np.ndarray:
    """Classical discrete CSP projection of ``y`` onto the ``M``-fast-mode SIM.

    One step is :math:`\\Delta\\mathbf y=-\\sum_{i\\le M}\\mathbf a_i\\lambda_i^{-1}f^i`
    (Section 4.3).  Formulation B is the continuous counterpart of this jump.
    """
    y = np.asarray(y, dtype=float).copy()
    for _ in range(iterations):
        basis = csp_basis(model, y, order, M)
        f = basis.amplitudes(model.g(y))
        y = y - basis.A[:, :M] @ (f[:M] / basis.lam[:M])
    return y


# ------------------------------------------------------------ two-stage model
def two_stage_rhs(model, M: int, sigma="adaptive", filt=None, order: int = 0,
                  stabilize: bool = True, tau_relax_max: float | None = None):
    """Right-hand side ``rhs(t, y)`` of the scale-space reduced model (Eq. 50).

    The returned function evaluates

    .. math::
       \\dot{\\mathbf y}=\\underbrace{\\mathcal S_{\\sigma}(\\mathbf y)\\,\\mathbf g(\\mathbf y)}_{\\text{Stage 2: tangent projection}}
       \\;\\underbrace{-\\sum_{i\\le M}\\big(1-w_i\\big)\\,\\mathbf a_i\\,
       \\frac{f^i}{\\lambda_i\\,\\tau_{relax}}}_{\\text{Stage 1 in continuous form}}

    with the basis, timescales and weights re-evaluated at the current state.

    * The first term is Formulation A at scale ``sigma``.
    * The second term (``stabilize=True``) is a continuous version of the
      fibre relaxation: it pulls the state back onto the manifold
      :math:`f^1=\\dots=f^M=0` on the time scale ``tau_relax`` (instead of the
      stiff :math:`\\tau_i`), so that the reduced model stays on
      :math:`\\mathcal M_\\sigma` during long integrations.  With
      ``stabilize=False`` the pure Stage-2 flow of Eq. (50) is returned.

    Parameters
    ----------
    model : ODEModel
    M : int
        Number of exhausted modes (level of reduction).
    sigma : float or "adaptive"
        Observation scale.  ``"adaptive"`` uses the state-dependent
        :func:`cutoff_scale` :math:`\\sigma_{c,M}(\\mathbf y)`.
    filt : filter specification, optional
    order : {0, 1}
        CSP basis order.
    stabilize : bool
        Include the continuous Stage-1 term.
    tau_relax_max : float, optional
        Upper bound on ``tau_relax`` (which otherwise equals ``sigma``).
    """
    filt = as_filter(filt)
    adaptive = isinstance(sigma, str)
    if adaptive and sigma != "adaptive":
        raise ValueError("sigma must be a number or 'adaptive'")

    def rhs(t, y):
        basis = csp_basis(model, y, order, M)
        sig = cutoff_scale(basis.tau, M) if adaptive else float(sigma)
        w = filt.weights(sig, basis.tau)
        f = basis.amplitudes(model.g(y))
        v = basis.A @ (w * f)
        if stabilize:
            tr = sig if tau_relax_max is None else min(sig, tau_relax_max)
            for i in range(M):
                if basis.lam[i] < 0:
                    v = v - (1.0 - w[i]) * basis.A[:, i] * f[i] / (basis.lam[i] * tr)
        return v

    return rhs


def integrate(rhs, y0, t_eval, *, jac=None, rtol: float = 1e-10, atol: float = 1e-12,
              method: str = "Radau"):
    """Integrate ``dy/dt = rhs(t, y)`` from ``t = 0`` and return the states at ``t_eval``.

    Thin wrapper around :func:`scipy.integrate.solve_ivp` with a stiff solver.

    Returns
    -------
    (N, T) array
    """
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), np.asarray(y0, dtype=float), method=method,
                    t_eval=t_eval, jac=jac, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"time integration failed: {sol.message}")
    return sol.y
