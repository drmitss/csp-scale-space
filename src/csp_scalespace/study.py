"""High-level workflow: from a model to all results and figures.

:class:`ScaleSpaceStudy` is the object most users interact with.  It takes an
:class:`~csp_scalespace.model.ODEModel` and a set of :class:`Settings`, and
lazily computes (and caches) everything the figures need:

* the reference trajectory :math:`\\mathbf y(t)` of the full model,
* the CSP diagnostics along it (eigenvalues, timescales, pointers),
* the fibre relaxation :math:`\\mathbf y_\\sigma` of the initial state
  (Formulation B / Stage 1),
* the scale-space reduced models for every requested reduction level ``M``
  (Stage 2),
* optional user-supplied reference reduced models.

``study.make_figures(outdir)`` then writes the complete figure set.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path

import numpy as np

from .csp import csp_basis
from .filters import ScaleFilter, as_filter
from .model import ODEModel
from .scalespace import cutoff_scale, fiber_relaxation, integrate, two_stage_rhs

__all__ = ["Settings", "ScaleSpaceStudy"]


@dataclass
class Settings:
    """Analysis and plotting options of a :class:`ScaleSpaceStudy`.

    Every field can be given in the ``settings:`` block of a model file or as
    a keyword argument of :class:`ScaleSpaceStudy`.

    Time integration
        ``t_end`` (required) final time; ``t_start`` first output time (only
        used with a logarithmic grid); ``n_points`` number of output times;
        ``time_scale`` ``"linear"`` or ``"log"`` (use ``"log"`` when the
        dynamics span several decades in time); ``time_unit`` unit string for
        axis labels; ``rtol``/``atol`` integrator tolerances.

    Scale-space method
        ``filter`` kind of scale filter (``"gaussian"``, ``"exponential"``,
        ``"rational"``) and ``p`` its exponent; ``fiber_filter`` filter used
        for the fibre relaxation (defaults to ``filter``); ``order`` CSP basis
        order (0 = Jacobian eigenvectors, 1 = one B- and A-refinement);
        ``levels`` list of reduction levels ``M`` (number of exhausted modes)
        to analyse - default: every ``M = 1..N-1`` with a finite spectral gap;
        ``fixed_sigmas`` additional constant scales at which reduced models
        are integrated (besides the adaptive :math:`\\sigma_c(\\mathbf y)`);
        ``stabilize`` keep the reduced model on the manifold (see
        :func:`~csp_scalespace.scalespace.two_stage_rhs`); ``tau_relax_max``
        cap of the stabilisation time.

    Diagnostics
        ``snapshots`` times at which pointers are plotted against the scale;
        ``sigma_range`` ``[min, max]`` of the scale axis (default: derived
        from the timescales met along the trajectory).

    Phase-space plots
        ``phase_variables`` two (or three) variable names; ``phase_log`` use
        logarithmic axes; ``extra_initial_conditions`` additional initial
        states for the phase portrait and ``extra_t_end`` their integration
        time; ``perturbation`` relative size of the off-manifold perturbations
        in the anisotropic-filtering figure; ``smoothing_width`` width (in
        time) of the naive Gaussian smoothing shown for comparison.

    Time-series plots
        ``log_states`` plot states on a log axis; ``state_floor`` lower clip
        value for such plots.

    Asymptotic study (optional, paper Fig. 6b)
        ``gap_parameter`` name of the model parameter that controls the
        spectral gap; ``gap_values`` list of values or
        ``{start, stop, num, scale}``; ``gap_reference_state`` state at which
        errors are measured (projected onto the exact SIM when the model has
        one) or ``gap_reference_time`` time along the trajectory.

    Comparison
        ``reference_models`` ``{label: g(y)}`` reduced vector fields
        integrated from the same initial condition and overlaid on the
        comparison figures.
    """

    t_end: float = None
    t_start: float = 0.0
    n_points: int = 800
    time_scale: str = "linear"
    time_unit: str = ""
    rtol: float = 1e-10
    atol: float = 1e-12
    filter: object = "gaussian"
    p: float = 2.0
    fiber_filter: object = None
    order: int = 1
    levels: list | None = None
    fixed_sigmas: list = field(default_factory=list)
    stabilize: bool = True
    tau_relax_max: float | None = None
    snapshots: list | None = None
    sigma_range: list | None = None
    phase_variables: list | None = None
    phase_log: bool = False
    extra_initial_conditions: list | None = None
    extra_t_end: float | None = None
    perturbation: float = 0.12
    smoothing_width: float | None = None
    log_states: bool = False
    state_floor: float | None = None
    gap_parameter: str | None = None
    gap_values: object = None
    gap_reference_state: list | None = None
    gap_reference_time: float | None = None
    reference_models: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown setting(s): {sorted(unknown)}; valid: {sorted(known)}")
        return cls(**d)


class ScaleSpaceStudy:
    """Complete CSP scale-space analysis of one model.

    Parameters
    ----------
    model : ODEModel
    settings : Settings or dict, optional
    **overrides
        Individual :class:`Settings` fields, e.g. ``t_end=4.0, order=0``.

    Examples
    --------
    >>> import csp_scalespace as css
    >>> from csp_scalespace.models import davis_skodje
    >>> study = css.ScaleSpaceStudy(davis_skodje(), t_end=4.0)
    >>> study.sigma_c(1)[0]            # adaptive cutoff sqrt(tau_1 tau_2) at t = 0
    0.2828...
    >>> files = study.make_figures("out")      # doctest: +SKIP
    """

    def __init__(self, model: ODEModel, settings=None, **overrides):
        if settings is None:
            settings = {}
        if isinstance(settings, Settings):
            settings = {f.name: getattr(settings, f.name) for f in fields(Settings)}
        self.settings = Settings.from_dict({**settings, **overrides})
        if self.settings.t_end is None:
            raise ValueError("the final time 't_end' must be given")
        self.model = model
        self._cache = {}

    @classmethod
    def from_file(cls, path, **overrides) -> "ScaleSpaceStudy":
        """Build the study from a YAML/JSON model file."""
        from .model import load_model_file

        model, settings = load_model_file(path)
        return cls(model, settings, **overrides)

    # ------------------------------------------------------------- basic data
    @property
    def filter(self) -> ScaleFilter:
        """Scale filter used for weights, pointers and Stage 2."""
        s = self.settings
        if isinstance(s.filter, str) and s.filter == "exponential":
            return ScaleFilter("exponential", s.p)
        return as_filter(s.filter)

    @property
    def fiber_filter(self) -> ScaleFilter:
        """Scale filter used for the fibre relaxation (Stage 1)."""
        return self.filter if self.settings.fiber_filter is None else as_filter(self.settings.fiber_filter)

    @cached_property
    def t(self) -> np.ndarray:
        """Output time grid."""
        s = self.settings
        if s.time_scale == "log":
            t0 = s.t_start if s.t_start > 0 else 1e-6 * s.t_end
            t = np.logspace(np.log10(t0), np.log10(s.t_end), s.n_points)
            t[-1] = s.t_end
            return t
        return np.linspace(0.0, s.t_end, s.n_points)

    def solve(self, rhs, y0, t=None, jac=None) -> np.ndarray:
        """Integrate ``rhs(t, y)`` from ``y0`` with the study's tolerances."""
        t = self.t if t is None else t
        return integrate(rhs, y0, t, jac=jac, rtol=self.settings.rtol, atol=self.settings.atol)

    @cached_property
    def full(self) -> np.ndarray:
        """Trajectory of the full model, shape ``(N, T)``."""
        m = self.model
        return self.solve(lambda t, y: m.g(y), m.y0, jac=lambda t, y: m.J(y))

    @cached_property
    def diagnostics(self) -> dict:
        """CSP data along the trajectory.

        Keys: ``lam`` and ``tau`` of shape ``(T, N)``; ``mode_pointers`` of
        shape ``(T, N, N)`` (mode, variable); ``u_fast`` of shape ``(T, N)``,
        the modal displacements :math:`u_i=f^i/\\lambda_i` (Eq. 41).
        """
        Y = self.full
        T, N = Y.shape[1], self.model.n
        lam, tau, mp, u = np.empty((T, N)), np.empty((T, N)), np.empty((T, N, N)), np.empty((T, N))
        for k in range(T):
            b = csp_basis(self.model, Y[:, k])
            lam[k], tau[k], mp[k] = b.lam, b.tau, b.mode_pointers()
            with np.errstate(divide="ignore", invalid="ignore"):
                u[k] = np.where(b.lam != 0, b.amplitudes(self.model.g(Y[:, k])) / b.lam, 0.0)
        return {"lam": lam, "tau": tau, "mode_pointers": mp, "u_fast": u}

    @cached_property
    def basis0(self):
        """Zeroth-order CSP basis at the initial state."""
        return csp_basis(self.model, self.model.y0)

    @cached_property
    def levels(self) -> list:
        """Reduction levels ``M`` analysed by the study."""
        if self.settings.levels is not None:
            return [int(M) for M in self.settings.levels]
        b = self.basis0
        return [M for M in range(1, self.model.n)
                if b.lam[M - 1] < 0 and np.isfinite(b.tau[M])]

    def sigma_c(self, M: int) -> np.ndarray:
        """Adaptive cutoff :math:`\\sigma_{c,M}(\\mathbf y(t))` along the trajectory."""
        return cutoff_scale(self.diagnostics["tau"], M)

    def pointers_at(self, sigma) -> tuple:
        """Fast and slow pointers along the trajectory at scale(s) ``sigma`` (scalar or ``(T,)``)."""
        d = self.diagnostics
        sig = np.broadcast_to(np.asarray(sigma, dtype=float), self.t.shape)
        w = self.filter.weights(sig[:, None], d["tau"])
        S = np.einsum("ti,tik->tk", w, d["mode_pointers"])
        return 1.0 - S, S

    def on_manifold(self, M: int = 1, tol: float = 1e-2) -> np.ndarray:
        """Boolean mask of the times at which the first ``M`` modes are exhausted.

        A mode counts as exhausted when its modal displacement
        :math:`|u_i|` is below ``tol`` times the norm of the state.
        """
        u = np.abs(self.diagnostics["u_fast"][:, :M]).max(axis=1)
        return u <= tol * np.linalg.norm(self.full, axis=0)

    @cached_property
    def sigma_grid(self) -> np.ndarray:
        """Logarithmic grid of observation scales used by the scale-axis figures."""
        if self.settings.sigma_range is not None:
            lo, hi = map(float, self.settings.sigma_range)
        else:
            tau = self.diagnostics["tau"]
            tau = tau[np.isfinite(tau)]
            lo, hi = 0.03 * tau.min(), 30.0 * tau.max()
        return np.logspace(np.log10(lo), np.log10(hi), 600)

    def snapshot_indices(self) -> list:
        """Indices of the time grid closest to the requested ``snapshots``."""
        snaps = self.settings.snapshots
        if snaps is None:
            snaps = [0.0]
        return [int(np.argmin(np.abs(self.t - float(ts)))) for ts in snaps]

    # ------------------------------------------------------ Stage 1 / Stage 2
    @cached_property
    def fiber(self):
        """Fibre relaxation of the initial state over the whole ``sigma_grid`` (zeroth-order basis)."""
        g = self.sigma_grid
        return fiber_relaxation(self.model, self.model.y0, g[-1], sigma_min=g[0], n=g.size,
                                filt=self.fiber_filter)

    def stage1(self, M: int, sigma="adaptive", y0=None):
        """Stage 1: relax ``y0`` (default: the model's initial state) onto the level-``M`` manifold.

        Returns the :class:`~csp_scalespace.scalespace.FiberPath` from
        :math:`\\sigma\\approx0` to the target scale (the self-consistent cutoff
        :math:`\\sigma=\\sigma_{c,M}(\\mathbf y_\\sigma)` when ``sigma="adaptive"``).
        """
        key = ("stage1", M, sigma, None if y0 is None else tuple(np.round(y0, 14)))
        if key not in self._cache:
            y0 = self.model.y0 if y0 is None else np.asarray(y0, dtype=float)
            target = "adaptive" if isinstance(sigma, str) else float(sigma)
            self._cache[key] = fiber_relaxation(
                self.model, y0, target, n=200, filt=self.fiber_filter, order=self.settings.order, M=M)
        return self._cache[key]

    def reduced(self, M: int, sigma="adaptive", relax: bool = True, stabilize=None) -> dict:
        """Scale-space reduced model of level ``M`` (two-stage operator, Eq. 50).

        Parameters
        ----------
        M : int
            Number of exhausted modes.
        sigma : float or "adaptive"
        relax : bool
            Start from the Stage-1 relaxed state (``True``) or directly from
            the model's initial state (``False``).
        stabilize : bool, optional
            Override :attr:`Settings.stabilize`.

        Returns
        -------
        dict
            ``y0`` starting state, ``y`` trajectory ``(N, T)``, ``error``
            maximum absolute deviation from the full model over the times at
            which the first ``M`` modes are exhausted.
        """
        s = self.settings
        stabilize = s.stabilize if stabilize is None else stabilize
        key = ("reduced", M, sigma, relax, stabilize)
        if key not in self._cache:
            y0 = self.stage1(M, sigma).final if relax else self.model.y0
            rhs = two_stage_rhs(self.model, M, sigma, self.filter, s.order, stabilize, s.tau_relax_max)
            y = self.solve(rhs, y0)
            mask = self.on_manifold(M)
            err = float(np.abs(y - self.full)[:, mask].max()) if mask.any() else float("nan")
            self._cache[key] = {"y0": y0, "y": y, "error": err}
        return self._cache[key]

    def reference(self, label: str) -> np.ndarray:
        """Trajectory of a user-supplied reference reduced model."""
        key = ("ref", label)
        if key not in self._cache:
            g = self.settings.reference_models[label]
            self._cache[key] = self.solve(lambda t, y: g(y), self.model.y0)
        return self._cache[key]

    # ----------------------------------------------------------------- output
    def summary(self) -> str:
        """Markdown table of timescales, cutoff scales and pointers at the snapshots."""
        d, names = self.diagnostics, self.model.names
        idx = self.snapshot_indices()
        rows = [["quantity"] + [f"t = {self.t[i]:.4g}" for i in idx]]
        rows += [[f"tau_{i + 1}"] + [f"{d['tau'][k, i]:.3e}" for k in idx] for i in range(self.model.n)]
        for M in self.levels:
            sc = self.sigma_c(M)
            D, _ = self.pointers_at(sc)
            rows.append([f"sigma_c (M={M})"] + [f"{sc[k]:.3e}" for k in idx])
            rows += [[f"D^{nm}(sigma_c, M={M})"] + [f"{D[k, j]:.4f}" for k in idx]
                     for j, nm in enumerate(names)]
            rows.append([f"sum_k D^k (M={M})"] + [f"{D[k].sum():.4f}" for k in idx])
        out = ["| " + " | ".join(r) + " |" for r in rows]
        out.insert(1, "|" + "---|" * len(rows[0]))
        return "\n".join(out)

    def make_figures(self, outdir="figures", formats=("png",), which=None, dpi: int = 200,
                     report: bool = True, verbose: bool = False) -> dict:
        """Create the figure set and save it to ``outdir``.

        Parameters
        ----------
        outdir : path
        formats : sequence of str
            File formats, e.g. ``("png", "pdf")``.
        which : sequence of str, optional
            Figure keys or groups (``"theory"``, ``"applications"``) to
            produce; default all.  See
            :data:`csp_scalespace.figures.FIGURES`.
        dpi : int
        report : bool
            Also write ``report.md`` (figure list with captions and the
            :meth:`summary` table).
        verbose : bool
            Print progress.

        Returns
        -------
        dict
            ``{figure key: [paths]}``.
        """
        from .figures import make_all

        return make_all(self, Path(outdir), formats, which, dpi, report, verbose)
