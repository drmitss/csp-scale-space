"""Model-independent figures.

Every function takes a :class:`~csp_scalespace.study.ScaleSpaceStudy` and
returns a :class:`matplotlib.figure.Figure` that is the counterpart, *for the
user's model*, of one figure of the paper.  Nothing here is schematic: every
curve, arrow and marker is computed from the model.

The catalogue :data:`FIGURES` lists the figures in two groups:

``theory``
    The step-by-step construction that the paper illustrates on the
    Davis-Skodje system (Figs. 1-6).
``applications``
    The diagnostics and reduced-model comparisons that the paper shows for
    the Michaelis-Menten mechanism and the three-scale network (Figs. 7-13).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

from .csp import csp_basis
from .scalespace import (cutoff_scale, fiber_relaxation, filtered_field, radical_correction,
                         two_stage_rhs)

__all__ = ["FIGURES", "FigureSpec", "make_all", "paper_style", "PALETTE"]

#: Colours assigned to the state variables (in order).
PALETTE = ["#1f4e78", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]
_LEVEL_COLORS = ["mediumpurple", "#ff7f0e", "#17becf", "#8c564b"]
_WINDOW_COLORS = ["#d9e8f5", "#e2f0d9", "#fde9d9", "#ece2f0"]

_RC = {
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 11, "axes.labelsize": 11,
    "axes.titlesize": 11, "legend.fontsize": 9, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "savefig.bbox": "tight", "figure.dpi": 110,
}


@contextmanager
def paper_style():
    """Context manager applying the plotting style of the paper."""
    with matplotlib.rc_context(_RC):
        yield


def _plt():
    import matplotlib.pyplot as plt

    return plt


# ------------------------------------------------------------------ helpers
def _unit(st):
    return rf" ($\mathrm{{{st.settings.time_unit}}}$)" if st.settings.time_unit else ""


def _phase_idx(st, dims=None):
    names = st.settings.phase_variables or st.model.names[: min(3, st.model.n)]
    idx = [st.model.index(nm) for nm in names]
    return idx[:dims] if dims else idx


def _grid(ax, both=False):
    ax.grid(True, which="both" if both else "major", ls=":", alpha=0.4)


def _time_axis(st, ax):
    if st.settings.time_scale == "log":
        ax.set_xscale("log")
    ax.set_xlim(st.t[0], st.t[-1])
    ax.set_xlabel("Physical time $t$" + _unit(st))


def _clip(st, y):
    return np.maximum(y, st.settings.state_floor) if st.settings.state_floor else y


def _windows(st, ax, tau, label=True):
    """Shade the spectral-gap windows 3 tau_M < sigma < tau_{M+1}/3 for the analysed levels."""
    for j, M in enumerate(st.levels):
        lo, hi = 3.0 * tau[M - 1], tau[M] / 3.0
        if np.isfinite(hi) and lo < hi:
            ax.axvspan(lo, hi, color=_WINDOW_COLORS[j % 4], alpha=0.55, zorder=0,
                       label=rf"{st.model.n - M}D SIM window ($\tau_{M}\ll\sigma\ll\tau_{M + 1}$)"
                       if label else None)


def _tau_lines(ax, tau, y=0.93):
    lo, hi = ax.get_xlim()
    for i, tv in enumerate(tau):
        if np.isfinite(tv) and lo < tv < hi:
            ax.axvline(tv, color="k", ls=":", lw=1.1, alpha=0.7)
            ax.text(tv * 1.12, y, rf"$\tau_{i + 1}$", fontsize=12, transform=ax.get_xaxis_transform(),
                    bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.85))


def _legend_below(ax, ncol=2, fontsize=8.5, y=-0.16):
    """Legend underneath the axes, so that it never hides data of an arbitrary model."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=True, fontsize=fontsize)


def _by_arclength(Y, fractions, scale):
    """Indices of the columns of Y located at the given fractions of its (scaled) arc length."""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Y, axis=1) / scale[:, None], axis=0))]
    return [int(np.searchsorted(d, f * d[-1])) for f in np.atleast_1d(fractions)]


def _span(st):
    sp = np.ptp(st.full, axis=1)
    return np.where(sp > 0, sp, 1.0)


def _arrow(ax, p, q, color, lw=1.6, ls="-"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", color=color, lw=lw, ls=ls))


def _phase_axes(st, ax, idx):
    lab = st.model.labels
    ax.set_xlabel(f"{lab[idx[0]]}")
    ax.set_ylabel(f"{lab[idx[1]]}")
    if len(idx) == 3:
        ax.set_zlabel(f"{lab[idx[2]]}")
    elif st.settings.phase_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    if len(idx) == 2:
        _grid(ax)


def _new_phase(st, figsize=(6.2, 4.6), dims=None):
    plt = _plt()
    idx = _phase_idx(st, dims)
    fig = plt.figure(figsize=(6.4, 6.0) if len(idx) == 3 else figsize)
    ax = fig.add_subplot(111, projection="3d" if len(idx) == 3 else None)
    return fig, ax, idx


def _sim_part(st, M=1):
    """Trajectory points on the level-M manifold (used to draw the SIM)."""
    mask = st.on_manifold(M)
    return st.full[:, mask] if mask.sum() > 3 else st.full


def _draw_sim(st, ax, idx, M=1, label=True):
    Y = _sim_part(st, M)
    ax.plot(*Y[idx], "-", color="#2ca02c", lw=6.0, alpha=0.25, solid_capstyle="round",
            label=r"Slow invariant manifold $\mathcal{M}_{\mathrm{SIM}}$ (attracting part of $\mathbf{y}(t)$)"
            if label else None)


def _manifold_contours(st, ax, idx, orders=(0, 1), n=71):
    """Contours f^{1,(k)} = 0 of the level-1 CSP manifold in a 2-D phase plane (N = 2 only)."""
    if st.model.n != 2 or 1 not in st.levels:
        return
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    log = st.settings.phase_log
    xs = np.geomspace(x0, x1, n) if log else np.linspace(x0, x1, n)
    ys = np.geomspace(y0, y1, n) if log else np.linspace(y0, y1, n)
    styles = {0: ("#1f77b4", "--", 2.0, r"Zeroth-order manifold $f^{1,(0)}=0$"),
              1: ("#9467bd", ":", 3.0, r"First-order manifold $f^{1,(1)}=0$")}
    for k in orders:
        F = np.full((n, n), np.nan)
        for a, xv in enumerate(xs):
            for b, yv in enumerate(ys):
                y = np.empty(2)
                y[idx[0]], y[idx[1]] = xv, yv
                try:
                    F[b, a] = csp_basis(st.model, y, k, 1).amplitudes(st.model.g(y))[0]
                except Exception:  # singular basis far from the region of interest
                    pass
        col, ls, lw, lab = styles[k]
        ax.contour(xs, ys, F, levels=[0.0], colors=[col], linestyles=[ls], linewidths=[lw])
        ax.plot([], [], color=col, ls=ls, lw=lw, label=lab)


def _exact_sim_curve(st, ax, idx):
    """Exact SIM (if the model provides one) drawn through the visible range of the first phase variable."""
    m = st.model
    if m.exact_sim is None or m.n != 2:
        return False
    x0, x1 = ax.get_xlim()
    xs = np.geomspace(x0, x1, 300) if st.settings.phase_log else np.linspace(x0, x1, 300)
    pts = []
    for xv in xs:
        y = np.array(m.y0, dtype=float)
        y[idx[0]] = xv
        pts.append(m.project_to_exact_sim(y))
    pts = np.array(pts).T
    ax.plot(pts[idx[0]], pts[idx[1]], "k-", lw=2.0, label=r"Exact SIM $\mathcal{M}_{\mathrm{SIM}}$")
    return True


def _limits(ax, arrays, idx, pad=0.08, log=False):
    """Set 2-D axis limits that enclose all given (N, T) arrays."""
    for axis, i in zip("xy", idx[:2]):
        v = np.concatenate([np.ravel(a[i]) for a in arrays])
        v = v[np.isfinite(v)]
        if log:
            v = v[v > 0]
            lo, hi = v.min() / 1.5, v.max() * 1.5
        else:
            span = (v.max() - v.min()) or 1.0
            lo, hi = v.min() - pad * span, v.max() + pad * span
        getattr(ax, f"set_{axis}lim")(lo, hi)


def _level1(st):
    return st.levels[0] if st.levels else 1


def _unrelaxed_A(st, M):
    """Formulation A integrated from the raw initial state at fixed sigma_c(y0) (no relaxation)."""
    key = ("unrelaxed", M)
    if key not in st._cache:
        sig = float(cutoff_scale(st.basis0.tau, M))
        rhs = two_stage_rhs(st.model, M, sig, st.filter, st.settings.order, stabilize=False)
        st._cache[key] = st.solve(rhs, st.model.y0)
    return st._cache[key]


# ================================================================== THEORY
def fig_time_evolution(st):
    """State variables against time, with the fast initial layer highlighted (paper Figs. 1a, 7a)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    m, t, Y = st.model, st.t, st.full
    tau1 = st.basis0.tau[0]
    if st.settings.time_scale != "log" and 3 * tau1 > 2e-3 * t[-1]:
        ax.axvspan(0.0, 3 * tau1, color="#ff7f0e", alpha=0.14,
                   label=r"Fast initial layer ($t\lesssim3\tau_1$)")
    for k in range(m.n):
        ax.plot(t, _clip(st, Y[k]), "-", color=PALETTE[k % 8], lw=2.3, label=f"{m.labels[k]}$(t)$")
    if m.exact_sim is not None:
        Ys = np.array([m.project_to_exact_sim(Y[:, j]) for j in range(t.size)]).T
        for k in range(m.n):
            if not np.allclose(Ys[k], Y[k]):
                ax.plot(t, _clip(st, Ys[k]), "--", color="#2ca02c", lw=1.8,
                        label=f"Exact SIM value of {m.labels[k]}")
    if st.settings.log_states:
        ax.set_yscale("log")
    _time_axis(st, ax)
    ax.set_ylabel("State variables")
    ax.legend(frameon=True, framealpha=0.9)
    _grid(ax)
    return fig


def _extra_trajectories(st):
    key = "extra"
    if key not in st._cache:
        s, m = st.settings, st.model
        ics = s.extra_initial_conditions
        if ics is None:  # automatic: perturb points of the trajectory off the manifold
            rng = np.random.default_rng(0)
            Y = st.full
            span = np.ptp(Y, axis=1)
            ics = [Y[:, j] + 0.35 * span * rng.uniform(-1, 1, m.n)
                   for j in np.linspace(0, 0.35 * Y.shape[1], 4, dtype=int)]
            if np.all(Y > 0):
                ics = [np.abs(y) for y in ics]
        t_end = s.extra_t_end or st.t[-1]
        tt = np.linspace(0.0, t_end, 1500)
        st._cache[key] = [st.solve(lambda t, y: m.g(y), np.asarray(y0, float), tt,
                                   jac=lambda t, y: m.J(y)) for y0 in ics]
    return st._cache[key]


def fig_phase_portrait(st):
    """Phase portrait: reference trajectory, off-manifold trajectories and the SIM (paper Figs. 1b, 7b)."""
    fig, ax, idx = _new_phase(st)
    Y, extra = st.full, _extra_trajectories(st)
    if len(idx) == 2:
        _limits(ax, [Y] + extra, idx, log=st.settings.phase_log)
    _phase_axes(st, ax, idx)
    _draw_sim(st, ax, idx, _level1(st))
    if len(idx) == 2 and not st.settings.phase_log:
        _exact_sim_curve(st, ax, idx)
    for j, Ye in enumerate(extra):
        ax.plot(*Ye[idx], "-.", color="#1f4e78", lw=1.5, label="Off-manifold trajectories" if j == 0 else None)
        ax.plot(*Ye[idx, 0], "o", color="#1f4e78", ms=4.5)
        if len(idx) == 2 and not st.settings.phase_log:
            j0, j1 = _by_arclength(Ye, [0.12, 0.2], _span(st))
            _arrow(ax, Ye[idx, j0], Ye[idx, max(j1, j0 + 1)], "#1f4e78", 1.3)
    ax.plot(*Y[idx], "-", color="#d62728", lw=2.3, label=r"Reference trajectory $\mathbf{y}(t)$")
    ax.plot(*Y[idx, 0], "o", color="#d62728", ms=6, label=r"Initial state $\mathbf{y}_0$")
    if len(idx) == 2 and not st.settings.phase_log:  # fast-fibre directions a_1 along the SIM
        Ys = _sim_part(st, _level1(st))
        L = 0.12 * np.array([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim())])
        for j in _by_arclength(Ys, [0.15, 0.35, 0.55, 0.75, 0.92], _span(st)):
            a = csp_basis(st.model, Ys[:, j]).A[idx, 0]
            a = a / np.linalg.norm(a / L)
            ax.plot([Ys[idx[0], j] - a[0], Ys[idx[0], j] + a[0]], [Ys[idx[1], j] - a[1], Ys[idx[1], j] + a[1]],
                    "--", color="#888888", lw=1.0)
        ax.plot([], [], "--", color="#888888", lw=1.0, label=r"Fast fibre directions $\mathbf{a}_1$")
    _legend_below(ax, 2, y=-0.16 if len(idx) == 2 else -0.02)
    return fig


def fig_filter_weights(st):
    """Modal survival weights against the observation scale at the initial state (paper Fig. 2)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    tau, sg, N = st.basis0.tau, st.sigma_grid, st.model.n
    _windows(st, ax, tau)
    ax.semilogx(sg, np.where(sg < tau[0], 1.0, 0.0), ":", color="#555555", lw=3.0,
                label=r"Sharp cutoff $\Theta(\tau_1-\sigma)$ (classical CSP switch)")
    ax.set_xlim(sg[0], sg[-1])
    for i in range(N):
        ax.semilogx(sg, st.filter.weights(sg, tau[i]), "-", color=PALETTE[(i + 1) % 8], lw=2.4,
                    label=rf"Mode {i + 1}: $w_{i + 1}(\sigma)$, $\tau_{i + 1}={tau[i]:.3g}$")
    for M in st.levels:
        ax.axvline(cutoff_scale(tau, M), color="#2ca02c", ls="-.", lw=1.5,
                   label=r"Cutoff scales $\sigma_{c,M}=\sqrt{\tau_M\tau_{M+1}}$" if M == st.levels[0] else None)
    _tau_lines(ax, tau, 0.6)
    ax.plot(tau[np.isfinite(tau)], np.full(np.isfinite(tau).sum(), st.filter.weights(1.0, 1.0)), "o",
            color="#333333", ms=4.5, zorder=5)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel(r"Observation scale $\sigma$" + _unit(st))
    ax.set_ylabel(rf"Modal survival weight $w_i={st.filter.label}$")
    _legend_below(ax, 2)
    _grid(ax)
    return fig


def _perturbed_cloud(st, M, k=7):
    """Off-manifold states (trajectory points + perturbation) and their fibre relaxations."""
    key = ("cloud", M)
    if key not in st._cache:
        rng = np.random.default_rng(1)
        Ys = _sim_part(st, M)
        span = _span(st)
        out = []
        for n, j in enumerate(_by_arclength(Ys, np.linspace(0.1, 0.9, k), span)):
            y = Ys[:, j]
            g = st.model.g(y) / span
            d = rng.normal(size=st.model.n)
            d -= g * (d @ g) / max(g @ g, 1e-300)       # perturb transversally to the flow
            d *= (-1) ** n / np.linalg.norm(d)
            yp = y + st.settings.perturbation * span * d
            if np.all(st.full > 0):
                yp = np.maximum(yp, 1e-3 * np.abs(y))
            try:
                out.append(fiber_relaxation(st.model, yp, "adaptive", n=60, filt=st.fiber_filter,
                                            order=st.settings.order, M=M))
            except Exception:
                continue
        st._cache[key] = out
    return st._cache[key]


def fig_isotropic_vs_anisotropic(st):
    """Naive isotropic smoothing vs. anisotropic CSP filtering (paper Fig. 3).

    Left: the trajectory convolved in time with a Gaussian kernel drifts off
    the SIM.  Right: Formulation B moves perturbed states along the fast
    fibres only, back onto the SIM.
    """
    from scipy.ndimage import gaussian_filter1d

    plt = _plt()
    M, m = _level1(st), st.model
    idx = _phase_idx(st, 2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.5))
    Y, t = st.full, st.t
    width = st.settings.smoothing_width or 0.5 * np.median(st.diagnostics["tau"][:, M])
    width = min(width, 0.25 * t[-1])
    tu = np.linspace(0.0, t[-1], 4000)
    Yu = np.array([np.interp(tu, t, Y[k]) for k in range(m.n)])
    Ysm = gaussian_filter1d(Yu, width / (tu[1] - tu[0]), axis=1, mode="nearest")
    keep = (tu > 2 * width) & (tu < tu[-1] - 2 * width)
    cloud = _perturbed_cloud(st, M)
    for ax in (a1, a2):
        _limits(ax, [Y] + [c.y for c in cloud], idx, log=st.settings.phase_log)
        _phase_axes(st, ax, idx)
        _draw_sim(st, ax, idx, M, label=ax is a1)
    a1.plot(*Y[idx], "-", color="#2b2b2b", lw=1.6, label=r"Exact trajectory $\mathbf{y}(t)$")
    a1.plot(*Ysm[np.ix_(idx, np.flatnonzero(keep))], "-.", color="#d95f02", lw=2.2,
            label=rf"Gaussian-smoothed $K(\cdot;\zeta)*\mathbf{{y}}$, width {width:.3g}")
    a1.set_title("Naive isotropic (Euclidean) scale-space")
    _legend_below(a1, 1)
    for n, c in enumerate(cloud):
        a2.plot(*c.y[idx], "--", color="#d62728", lw=1.3, label="Fibre relaxation (Formulation B)" if n == 0 else None)
        a2.plot(*c.y[idx, 0], "o", color="#2b2b2b", ms=5, label=r"Perturbed states $\mathbf{y}_0$" if n == 0 else None)
        a2.plot(*c.y[idx, -1], "o", color="#2ca02c", ms=5.5,
                label=r"Relaxed states $\mathbf{y}_{\sigma_c}$" if n == 0 else None)
        if not st.settings.phase_log:
            _arrow(a2, c.y[idx, 0], c.y[idx, -1], "#d62728", 1.2)
    a2.set_title(r"Anisotropic CSP scale-space ($\mathcal{D}_\sigma=\mathbf{A}\mathbf{D}\mathbf{B}$)")
    _legend_below(a2, 1)
    fig.tight_layout()
    return fig


def fig_geometric_duality(st):
    """Velocity filtering (Formulation A) vs. fibre relaxation (Formulation B) at the initial state (paper Fig. 4)."""
    from scipy.optimize import brentq

    M, m, order = _level1(st), st.model, st.settings.order
    fig, ax, idx = _new_phase(st, (7.2, 5.0), dims=2)
    y0, b0 = m.y0, st.basis0
    path, red, unrel = st.stage1(M), st.reduced(M), _unrelaxed_A(st, M)
    y_rad = radical_correction(m, y0, M, order, iterations=1)
    _limits(ax, [st.full, path.y, y_rad[:, None]], idx, pad=0.12, log=st.settings.phase_log)
    _phase_axes(st, ax, idx)
    _draw_sim(st, ax, idx, M)
    ax.plot(*path.y[idx], "-.", color="#d62728", lw=2.4,
            label=r"Formulation B: $\partial_\sigma\mathbf{y}_\sigma=-\sum\mathbf{a}_i\lambda_i^{-1}d_if^i$")
    for frac, lab in ((0.45, r"$\mathbf{y}_{\sigma_1}$"), (0.7, r"$\mathbf{y}_{\sigma_2}$")):
        j = _by_arclength(path.y, frac, _span(st))[0]
        ax.plot(*path.y[idx, j], "o", color="#d62728", ms=4.5)
        ax.annotate(lab, path.y[idx, j], textcoords="offset points", xytext=(-22, -3), color="#d62728")
    ax.plot(*np.c_[y0, y_rad][idx], ":", color="#7f7f7f", lw=1.8,
            label=r"Classical CSP jump $\Delta\mathbf{y}=-\sum_{i\leq M}\mathbf{a}_iu_i$")
    ax.plot(*y_rad[idx], "x", color="#7f7f7f", ms=7, mew=2)
    ax.plot(*unrel[idx], "--", color="#ff7f0e", lw=1.9, label=r"Unrelaxed Formulation A flow (frozen offset)")
    ax.plot(*red["y"][idx], "--", color="#2ca02c", lw=2.4,
            label=r"Stage 2 flow $\dot{\mathbf{y}}=\mathcal{S}_{\sigma_c}\mathbf{g}$ from $\mathbf{y}_{\sigma_c}$")
    # velocity vectors at y0: full, half-filtered, fully filtered (lengths compressed for visibility)
    sig_c = float(cutoff_scale(b0.tau, M))
    sig_h = brentq(lambda s: st.filter.weights(s, b0.tau[M - 1]) - 0.5, 1e-6 * b0.tau[M - 1], 1e3 * b0.tau[M - 1])
    vecs = [(m.g(y0), "#2b2b2b", r"$\mathbf{g}(\mathbf{y}_0)$ ($\sigma=0$)"),
            (filtered_field(m, y0, sig_h, st.filter, order, M), "#9467bd",
             rf"$\mathbf{{g}}_\sigma(\mathbf{{y}}_0)$ ($w_{M}=0.5$)"),
            (filtered_field(m, y0, sig_c, st.filter, order, M), "#d95f02",
             r"$\mathbf{g}_{\sigma_c}(\mathbf{y}_0)$")]
    if not st.settings.phase_log:
        scale = np.array([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim())])
        gmax = np.linalg.norm(vecs[0][0][idx] / scale)
        for v, col, lab in vecs:
            vn = v[idx] / scale
            L = 0.30 * (np.linalg.norm(vn) / gmax) ** 0.4
            tip = y0[idx] + L * vn / max(np.linalg.norm(vn), 1e-300) * scale
            _arrow(ax, y0[idx], tip, col, 2.1)
            ax.plot([], [], "-", color=col, lw=2.1, label=lab)
    ax.plot(*y0[idx], "o", color="#2b2b2b", ms=7, zorder=7, label=r"Initial state $\mathbf{y}_0$")
    ax.plot(*path.final[idx], "o", color="#2ca02c", ms=7, zorder=7,
            label=r"$\mathbf{y}_{\sigma_c}\in\mathcal{M}_{\sigma_c}$ (after Stage 1)")
    _legend_below(ax, 2, 8)
    return fig


def _defect(st, M):
    """Relative invariance defect ||P_fast^(k) g|| / ||g|| along the trajectory, k = 0, 1."""
    key = ("defect", M)
    if key not in st._cache:
        Y = st.full
        out = np.empty((2, Y.shape[1]))
        for j in range(Y.shape[1]):
            g = st.model.g(Y[:, j])
            for k in (0, 1):
                out[k, j] = np.linalg.norm(csp_basis(st.model, Y[:, j], k, M).fast_projector(M) @ g) \
                    / max(np.linalg.norm(g), 1e-300)
        st._cache[key] = out
    return st._cache[key]


def fig_refinement_geometry(st):
    """Effect of the first-order B- and A-refinements (paper Fig. 5).

    Left: CSP basis vectors before (dashed) and after (solid) refinement at a
    point of the SIM, with the zeroth- and first-order manifolds.  Right: the
    invariance defect :math:`\\|\\mathbf P_{fast}^{(k)}\\mathbf g\\|/\\|\\mathbf g\\|`
    along the trajectory - how much of the velocity on the SIM is wrongly
    attributed to the fast subspace - for the two bases.
    """
    plt = _plt()
    M, m = _level1(st), st.model
    idx = _phase_idx(st, 2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.5))
    Ys = _sim_part(st, M)
    p = Ys[:, _by_arclength(Ys, 0.45, _span(st))[0]]
    _limits(a1, [st.full], idx, log=st.settings.phase_log)
    _phase_axes(st, a1, idx)
    if not _exact_sim_curve(st, a1, idx):
        _draw_sim(st, a1, idx, M)
    if not st.settings.phase_log:
        _manifold_contours(st, a1, idx)
        scale = np.array([np.ptp(a1.get_xlim()), np.ptp(a1.get_ylim())])
        b0, b1 = csp_basis(m, p), csp_basis(m, p, 1, M)
        for basis, ls, lw, tag in ((b0, "--", 1.8, "(0)"), (b1, "-", 2.4, "(1)")):
            for i, col in ((M - 1, "#d62728"), (M, "#2ca02c")):
                a = basis.A[idx, i] / scale
                a = 0.22 * a / np.linalg.norm(a) * scale * (np.sign(a[0] * (-1 if i == M else 1)) or 1.0)
                _arrow(a1, p[idx], p[idx] + a, col, lw, ls)
                a1.plot([], [], color=col, ls=ls, lw=lw,
                        label=rf"$\mathbf{{a}}_{i + 1}^{{{tag}}}$ ({'slow' if i == M else 'fast'}, "
                              rf"{'refined' if tag == '(1)' else 'eigenvector'})")
    a1.plot(*p[idx], "o", color="#1f4e78", ms=6.5, zorder=6, label="Base point $p$ on the SIM")
    _legend_below(a1, 2, 8)
    a1.set_title("CSP basis before / after refinement")
    d, mask = _defect(st, M), st.on_manifold(M)
    for k, col, lab in ((0, "#1f77b4", "Zeroth-order basis (eigenvectors)"),
                        (1, "#d62728", "First-order refined basis")):
        a2.semilogy(st.t, np.maximum(d[k], 1e-17), "-", color=col, lw=1.0, alpha=0.35)
        a2.semilogy(st.t[mask], np.maximum(d[k][mask], 1e-17), "-", color=col, lw=2.2, label=lab)
    _time_axis(st, a2)
    a2.set_ylabel(r"Invariance defect $\|\mathbf{P}^{(k)}_{fast}\mathbf{g}\|/\|\mathbf{g}\|$")
    a2.set_title(f"Tangency error on the SIM (bold: first {M} mode(s) exhausted)")
    a2.legend(frameon=True, fontsize=8.5)
    _grid(a2, both=True)
    fig.tight_layout()
    return fig


def fig_two_stage(st):
    """Two-stage operator in phase space: fibre relaxation, then tangent flow (paper Figs. 6a, 13)."""
    fig, ax, idx = _new_phase(st, (6.6, 5.0))
    m, Y = st.model, st.full
    three = len(idx) == 3
    if not three:
        _limits(ax, [Y] + [st.stage1(M).y for M in st.levels], idx, log=st.settings.phase_log)
    _phase_axes(st, ax, idx)
    if not three and not st.settings.phase_log:
        _exact_sim_curve(st, ax, idx)
        _manifold_contours(st, ax, idx)
    ax.plot(*Y[idx], "-", color="#333333" if three else "#2ca02c", lw=2.4 if three else 1.8,
            label=r"Full-model trajectory $\mathbf{y}(t)$")
    if three:
        ax.plot(*st.fiber.y[idx], ":", color="#d62728", lw=1.5,
                label=r"Fibre relaxation $\mathbf{y}_\sigma$, all scales")
    for j, M in enumerate(st.levels):
        col = "#d62728" if (not three and j == 0) else _LEVEL_COLORS[j % 4]
        p, r = st.stage1(M), st.reduced(M)
        ax.plot(*p.y[idx], "-.", color=col, lw=1.8,
                label=rf"Stage 1: fibre relaxation to $\sigma_{{c,{M}}}$")
        ax.plot(*r["y"][idx], "--" if three else "-", color=col, lw=1.6, alpha=0.9,
                label=rf"Stage 2: scale-space flow, $M={M}$ ({m.n - M}D SIM)")
        ax.plot(*[[v] for v in p.final[idx]], marker="^s"[j % 2], color=col, ms=7, ls="none", zorder=8)
    if not three:
        ax.plot(*_unrelaxed_A(st, _level1(st))[idx], "--", color="#ff7f0e", lw=1.6,
                label=r"Unrelaxed Formulation A ($\dot{\mathbf{y}}=\mathcal{S}_{\sigma_c}\mathbf{g}$ from $\mathbf{y}_0$)")
    ax.plot(*[[v] for v in m.y0[idx]], "o", color="k", ms=6.5, ls="none", zorder=8,
            label=r"Initial state $\mathbf{y}_0$")
    if three:
        ax.view_init(elev=28, azim=-58)
    _legend_below(ax, 2, 8, y=-0.02 if three else -0.16)
    return fig


def convergence_data(st):
    """Errors of the zeroth- and first-order bases against the spectral-gap parameter.

    For each value of ``gap_parameter`` a reference point on the true SIM is
    built (exact SIM if the model has one, otherwise a long-time trajectory
    point).  Returned arrays, each of shape ``(2, n_values)`` for orders 0/1:

    ``velocity``
        :math:`\\|\\mathbf P_{fast}^{(k)}\\mathbf g\\|`: part of the true
        on-manifold velocity removed by the level-``M`` projector.
    ``manifold``
        distance, along the fast fibres, between the reference point and the
        ``k``-th order CSP manifold :math:`f^{1..M,(k)}=0`.
    """
    from scipy.optimize import root

    s, M = st.settings, _level1(st)
    gv = s.gap_values
    if isinstance(gv, dict):
        fn = np.logspace if gv.get("scale", "log") == "log" else np.linspace
        gv = fn(float(gv["start"]), float(gv["stop"]), int(gv.get("num", 30)))
    gv = np.asarray(gv, dtype=float)
    vel, man, eps = np.empty((2, gv.size)), np.empty((2, gv.size)), np.empty(gv.size)
    for n, val in enumerate(gv):
        mm = st.model.with_parameters(**{s.gap_parameter: float(val)})
        if mm.exact_sim is not None:
            ref = np.asarray(s.gap_reference_state if s.gap_reference_state is not None else mm.y0, float)
            y = mm.project_to_exact_sim(ref)
        else:
            tr = s.gap_reference_time or 0.25 * st.t[-1]
            y = st.solve(lambda t, yy: mm.g(yy), mm.y0, np.array([0.5 * tr, tr]),
                         jac=lambda t, yy: mm.J(yy))[:, -1]
        b0, g = csp_basis(mm, y), mm.g(y)
        eps[n] = b0.tau[M - 1] / b0.tau[M]
        Af = b0.A[:, :M]
        for k in (0, 1):
            vel[k, n] = np.linalg.norm(csp_basis(mm, y, k, M).fast_projector(M) @ g)
            sol = root(lambda c: csp_basis(mm, y + Af @ c, k, M).amplitudes(mm.g(y + Af @ c))[:M],
                       np.zeros(M), tol=1e-14)
            man[k, n] = np.linalg.norm(Af @ sol.x)
    return {"values": gv, "eps": eps, "velocity": vel, "manifold": man}


def fig_refinement_convergence(st):
    """Asymptotic accuracy of the zeroth- and first-order bases vs. the spectral gap (paper Fig. 6b)."""
    plt = _plt()
    d = convergence_data(st)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    x = d["eps"]
    order = np.argsort(x)
    spec = [("velocity", 0, "b-", "Zeroth-order velocity error"), ("velocity", 1, "r-", "First-order velocity error"),
            ("manifold", 0, "b--", "Zeroth-order manifold error"), ("manifold", 1, "r--", "First-order manifold error")]
    for key, k, fmt, lab in spec:
        yv = d[key][k]
        ok = (yv > 1e-13) & (x < 0.05)
        slope = np.polyfit(np.log(x[ok]), np.log(yv[ok]), 1)[0] if ok.sum() > 2 else np.nan
        ax.loglog(x[order], yv[order], fmt, lw=2.0, label=rf"{lab} $\sim\epsilon^{{{slope:.2f}}}$")
    ax.set_xlabel(r"Spectral gap $\epsilon=\tau_M/\tau_{M+1}$" + f"  (parameter '{st.settings.gap_parameter}')")
    ax.set_ylabel("Tangency and SIM errors")
    ax.legend(frameon=True, fontsize=9)
    _grid(ax, both=True)
    return fig


# ============================================================ APPLICATIONS
def fig_timescales(st):
    """Local timescales and adaptive cutoff scales along the trajectory (paper Figs. 8a, 12b)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    tau, N = st.diagnostics["tau"], st.model.n
    for i in range(N - 1, -1, -1):
        ax.semilogy(st.t, tau[:, i], "-", color=PALETTE[(i + 1) % 8], lw=2.3,
                    label=rf"Timescale $\tau_{i + 1}(\mathbf{{y}}(t))$")
    for j, M in enumerate(st.levels):
        ax.semilogy(st.t, st.sigma_c(M), "-.", color=_LEVEL_COLORS[j % 4], lw=2.1,
                    label=rf"Adaptive scale $\sigma_{{c,{M}}}(t)=\sqrt{{\tau_{M}\tau_{M + 1}}}$")
    _time_axis(st, ax)
    ax.set_ylabel("Timescale / scale" + _unit(st))
    ax.legend(frameon=True, fontsize=8.5)
    _grid(ax, both=st.settings.time_scale == "log")
    return fig


def fig_pointers_vs_scale(st, k: int = 0):
    """Fast (solid) and slow (dashed) pointers against the scale at one trajectory point (paper Figs. 9a-c, 11a)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    d, sg, m = st.diagnostics, st.sigma_grid, st.model
    tau, mp = d["tau"][k], d["mode_pointers"][k]
    S = st.filter.weights(sg[:, None], tau) @ mp
    _windows(st, ax, tau)
    for j in range(m.n):
        ax.semilogx(sg, 1 - S[:, j], "-", color=PALETTE[j % 8], lw=2.2, label=f"Fast pointer $D$ of {m.labels[j]}")
        ax.semilogx(sg, S[:, j], "--", color=PALETTE[j % 8], lw=1.6, alpha=0.85,
                    label=f"Slow pointer $S$ of {m.labels[j]}")
    for M in st.levels:
        ax.axvline(cutoff_scale(tau, M), color="#2ca02c", ls="-.", lw=1.5)
    _tau_lines(ax, tau)
    ax.set_xlim(sg[0], sg[-1])
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel(r"Observation scale $\sigma$" + _unit(st))
    ax.set_ylabel(rf"Scale-dependent pointers at $t={st.t[k]:.3g}$")
    _legend_below(ax, 2, 8)
    _grid(ax)
    return fig


def fig_pointers_trajectory(st):
    """Fast pointers at the adaptive cutoff scale along the trajectory (paper Figs. 8b, 9d)."""
    plt = _plt()
    L, m = st.levels, st.model
    fig, axes = plt.subplots(len(L), 1, figsize=(6.2, 1.2 + 3.0 * len(L)), squeeze=False, sharex=True)
    for ax, M in zip(axes[:, 0], L):
        D, _ = st.pointers_at(st.sigma_c(M))
        for j in range(m.n):
            ax.plot(st.t, D[:, j], "-", color=PALETTE[j % 8], lw=2.3,
                    label=rf"$D$ of {m.labels[j]} at $\sigma_{{c,{M}}}(t)$")
        ax.plot(st.t, D.sum(axis=1), "-.", color="k", lw=1.6, label=rf"$\sum_kD^k$ (expected: {M})")
        ax.set_ylabel(f"Fast pointers, $M={M}$")
        ax.set_ylim(-0.05, M + 0.15)
        ax.legend(frameon=True, fontsize=8.5)
        _grid(ax)
    _time_axis(st, axes[-1, 0])
    fig.tight_layout()
    return fig


def fig_realtime_pointers(st):
    """Sequential mode exhaustion: fast pointers with the scale tied to elapsed time, sigma = t (paper Fig. 12c)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    m, ok = st.model, st.t > 0
    D, _ = st.pointers_at(st.t)
    for j in range(m.n):
        ax.semilogx(st.t[ok], D[ok, j], "-", color=PALETTE[j % 8], lw=2.2,
                    label=rf"$D$ of {m.labels[j]} at $\sigma=t$")
    ax.semilogx(st.t[ok], D[ok].sum(axis=1), "-", color="k", lw=2.4,
                label=r"Number of exhausted modes $M(t)=\sum_kD^k$")
    ax.set_xlabel(r"Physical time $t$ with $\sigma=t$" + _unit(st))
    ax.set_ylabel("Fast pointers and $M(t)$")
    ax.set_ylim(-0.05, m.n + 0.3)
    ax.legend(frameon=True, fontsize=8.5, loc="upper left")
    _grid(ax, both=True)
    return fig


def fig_fiber_relaxation(st):
    """Formulation B: the initial state relaxed across all scales, revealing the SIM plateaus (paper Fig. 11b)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    m, f, tau = st.model, st.fiber, st.basis0.tau
    _windows(st, ax, tau)
    for j in range(m.n):
        ax.semilogx(f.sigma, f.y[j], "-", color=PALETTE[j % 8], lw=2.2,
                    label=rf"Relaxed state {m.labels[j]}$_\sigma$")
    _tau_lines(ax, tau)
    ax.set_xlim(f.sigma[0], f.sigma[-1])
    ax.set_xlabel(r"Observation scale $\sigma$" + _unit(st))
    ax.set_ylabel(r"Scale-relaxed state $\mathbf{y}_\sigma$")
    ax.legend(frameon=True, fontsize=8.5)
    _grid(ax)
    return fig


def fig_model_comparison(st, sigma="adaptive"):
    """Scale-space reduced models against the full model and reference reductions (paper Figs. 10, 12a)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    m, t, Y = st.model, st.t, st.full
    styles = ["--", ":", "-.", (0, (5, 1))]
    for k in range(m.n):
        ax.plot(t, _clip(st, Y[k]), "-", color=PALETTE[k % 8], lw=2.0)
        ax.annotate(m.labels[k], (t[-1], _clip(st, Y[k])[-1]), textcoords="offset points", xytext=(4, 0),
                    color=PALETTE[k % 8], fontsize=10, va="center")
    ax.plot([], [], "-", color="k", lw=2.0, label="Full model")
    for j, M in enumerate(st.levels):
        r = st.reduced(M, sigma)
        for k in range(m.n):
            ax.plot(t, _clip(st, r["y"][k]), ls=styles[j % 4], color=PALETTE[k % 8], lw=3.0)
        ax.plot([], [], ls=styles[j % 4], color="k", lw=3.0,
                label=rf"CSP scale-space, $M={M}$ (max. error {r['error']:.1e})")
    sub = np.linspace(0, t.size - 1, 36, dtype=int)
    for n, label in enumerate(st.settings.reference_models):
        R = st.reference(label)
        for k in range(m.n):
            ax.plot(t[sub], _clip(st, R[k, sub]), "+x1"[n % 3], color=PALETTE[k % 8], ms=7, mew=1.5)
        ax.plot([], [], "+x1"[n % 3], color="k", ms=7, mew=1.5, label=f"{label} model")
    if st.settings.log_states:
        ax.set_yscale("log")
    _time_axis(st, ax)
    ax.set_ylabel("State variables")
    title = r"adaptive $\sigma_c(\mathbf{y}(t))$" if isinstance(sigma, str) else rf"fixed $\sigma={sigma:g}$"
    ax.set_title("Reduced model at " + title)
    ax.legend(frameon=True, fontsize=8.5)
    _grid(ax)
    return fig


# ================================================================ catalogue
@dataclass
class FigureSpec:
    """Entry of the figure catalogue."""

    key: str
    group: str
    paper: str
    func: Callable
    caption: str
    expand: Callable | None = None      # study -> list of (suffix, kwargs)
    needs: Callable | None = None       # study -> bool


def _snap(st):
    return [(f"_t{st.t[k]:.3g}", {"k": k}) for k in st.snapshot_indices()]


def _sig(st):
    return [("_adaptive", {"sigma": "adaptive"})] + [(f"_sigma{float(s):g}", {"sigma": float(s)})
                                                     for s in st.settings.fixed_sigmas]


_has_level = lambda st: bool(st.levels)  # noqa: E731

#: Catalogue of the model-independent figures, in the order of the paper.
FIGURES = [
    FigureSpec("01_time_evolution", "theory", "Fig. 1a / 7a", fig_time_evolution,
               "State variables in time; the fast initial layer (t < 3 tau_1) is shaded."),
    FigureSpec("02_phase_portrait", "theory", "Fig. 1b / 7b", fig_phase_portrait,
               "Phase portrait with off-manifold trajectories collapsing onto the slow invariant manifold."),
    FigureSpec("03_filter_weights", "theory", "Fig. 2", fig_filter_weights,
               "Modal survival weights w_i(sigma) at the initial state, timescales and cutoff scales."),
    FigureSpec("04_isotropic_vs_anisotropic", "theory", "Fig. 3", fig_isotropic_vs_anisotropic,
               "Naive Gaussian smoothing of the trajectory (left) against anisotropic fibre relaxation of "
               "perturbed states (right).", needs=_has_level),
    FigureSpec("05_geometric_duality", "theory", "Fig. 4", fig_geometric_duality,
               "Formulation A (velocity filtering at fixed state) vs. Formulation B (state relaxation along "
               "fibres) at the initial state. Arrow lengths are compressed (power 0.4) for visibility.",
               needs=_has_level),
    FigureSpec("06_refinement_geometry", "theory", "Fig. 5", fig_refinement_geometry,
               "CSP basis vectors before/after the first-order B- and A-refinements and the resulting "
               "tangency error along the trajectory.", needs=_has_level),
    FigureSpec("07_two_stage", "theory", "Fig. 6a / 13", fig_two_stage,
               "Unified two-stage operator: Stage 1 fibre relaxation followed by the Stage 2 tangent flow.",
               needs=_has_level),
    FigureSpec("08_refinement_convergence", "theory", "Fig. 6b", fig_refinement_convergence,
               "Asymptotic errors of the zeroth- and first-order bases against the spectral gap.",
               needs=lambda st: bool(st.levels) and st.settings.gap_parameter is not None),
    FigureSpec("09_timescales", "applications", "Fig. 8a / 12b", fig_timescales,
               "Local timescales tau_i(y(t)) and adaptive cutoff scales sigma_c(t) along the trajectory."),
    FigureSpec("10_pointers_vs_scale", "applications", "Fig. 9a-c / 11a", fig_pointers_vs_scale,
               "Scale-dependent fast/slow CSP pointers at the snapshot times.", expand=_snap),
    FigureSpec("11_pointers_trajectory", "applications", "Fig. 8b / 9d", fig_pointers_trajectory,
               "Fast pointers evaluated at the adaptive cutoff scale along the trajectory.", needs=_has_level),
    FigureSpec("12_realtime_pointers", "applications", "Fig. 12c", fig_realtime_pointers,
               "Sequential mode exhaustion: pointers with sigma = t and the number of exhausted modes."),
    FigureSpec("13_fiber_relaxation", "applications", "Fig. 11b", fig_fiber_relaxation,
               "Formulation B across all scales: plateaus of the relaxed state on the nested manifolds."),
    FigureSpec("14_model_comparison", "applications", "Fig. 10 / 12a", fig_model_comparison,
               "Scale-space reduced models vs. the full model (and reference reduced models).",
               expand=_sig, needs=_has_level),
]


def make_all(study, outdir: Path, formats=("png",), which=None, dpi=200, report=True, verbose=False) -> dict:
    """Produce the catalogue for ``study``; see :meth:`ScaleSpaceStudy.make_figures`."""
    import time

    plt = _plt()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    which = None if which is None else set(which)
    written, lines = {}, []
    with paper_style():
        for spec in FIGURES:
            if which is not None and not ({spec.key, spec.key[3:], spec.group} & which):
                continue
            if spec.needs is not None and not spec.needs(study):
                continue
            for suffix, kwargs in (spec.expand(study) if spec.expand else [("", {})]):
                t0 = time.time()
                fig = spec.func(study, **kwargs)
                paths = []
                for fmt in formats:
                    path = outdir / f"{spec.key}{suffix}.{fmt}"
                    fig.savefig(path, dpi=dpi)
                    paths.append(path)
                plt.close(fig)
                written[spec.key + suffix] = paths
                lines.append(f"| `{paths[0].name}` | {spec.group} | {spec.paper} | {spec.caption} |")
                if verbose:
                    print(f"  {spec.key + suffix:<42s} {time.time() - t0:6.1f} s")
    if report:
        m = study.model
        text = [f"# CSP scale-space report: {m.name}", "", m.description, "",
                f"- variables: {', '.join(m.names)}", f"- parameters: {m.params}",
                f"- initial state: {m.y0.tolist()}", f"- reduction levels M: {study.levels}",
                f"- filter: {study.filter.kind} (p = {study.filter.p:g}), CSP basis order: {study.settings.order}",
                "", "## Figures", "", "| file | group | paper figure | content |", "|---|---|---|---|", *lines,
                "", "## Timescales, cutoff scales and pointers", "", study.summary(), ""]
        if study.levels and (which is None or "14_model_comparison" in written or "applications" in which):
            text += ["## Accuracy of the scale-space reduced models", "",
                     "Maximum absolute deviation from the full model once the first M modes are exhausted.", "",
                     "| M | sigma | Stage-1 state | max error |", "|---|---|---|---|"]
            for M in study.levels:
                for _, kw in _sig(study):
                    r = study.reduced(M, kw["sigma"])
                    text.append(f"| {M} | {kw['sigma']} | {np.round(r['y0'], 5).tolist()} | {r['error']:.3e} |")
        (outdir / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return written
