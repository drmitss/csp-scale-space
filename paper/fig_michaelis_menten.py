"""Figures 7-10: Michaelis-Menten regime-shift benchmark (Section 6.1).

Ported from the notebook that produced the manuscript figures; the annotations
are specific to the paper's models and parameters.  For the same analysis of
an arbitrary model use :class:`csp_scalespace.ScaleSpaceStudy` instead.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.integrate import solve_ivp

from _common import *  # noqa: F401,F403  (models, wrappers, style, save)

def make_fig_mm_intro_time():
    t_eval = np.linspace(0.0, 20.3, 2000)
    sol_full = solve_ivp(mm_rhs, (0.0, 20.3), y0_mm, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)
    sol_pea  = solve_ivp(pea_rhs, (0.0, 20.3), y0_mm, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)
    sol_ss   = solve_ivp(lambda t, y: mm_twostage_scalespace_rhs(t, y, 0.25),
                         (0.0, 20.3), y0_mm, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)

    c_full, s_full = sol_full.y
    c_pea,  s_pea  = sol_pea.y
    c_ss,   s_ss   = sol_ss.y

    # -------------------------------------------------------------------------
    # 1. Introductory MM Figure A: Solution in Time (fig_mm_intro_time.pdf)
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(6.0, 4.2))
    ax1.axvspan(0.0, 9.7, color="#f8bbd0", alpha=0.25, label=r"Regime I: sQSSA / PEA ($\nu \ll 1$)")
    ax1.axvspan(9.7, 10.3, color="#eeeeee", alpha=0.8, label=r"Transition Zone ($\nu \approx 1$, PEA only)")
    ax1.axvspan(10.3, 20.3, color="#c8e6c9", alpha=0.30, label=r"Regime II: rQSSA / PEA ($\nu \gg 1$)")
    ax1.axvline(10.0, color="#444444", ls="--", lw=1.2)

    ax1.plot(t_eval, s_full, "-", color="#1f4e78", lw=2, label=r"Substrate $s(t)$ (Full ODE)")
    ax1.plot(t_eval, c_full, "-", color="#d62728", lw=2, label=r"Complex $c(t)$ (Full ODE)")

    ax1.annotate(r"$c(t) \approx e_0 = 10$ (Fast Slaved)" + "\n" + r"$s(t)$ Decays Linearly (Slow Driver)",
                 xy=(4.5, 9.95), xytext=(0.9, 7.5),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3),
                 fontsize=9, color="#333333",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728", alpha=0.9))

    ax1.annotate(r"Dynamical Shift ($t \approx 10\mathrm{s}$)" + "\n" + r"$s(t) \approx 0$ (Fast Slaved)" + "\n" + r"$c(t)$ Decays Exponentially",
                 xy=(10.0, 9.9), xytext=(12.2, 8.5),
                 arrowprops=dict(arrowstyle="->", color="#1b7837", lw=1.3),
                 fontsize=9, color="#333333",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2ca02c", alpha=0.9))

    # ax1.set_title(r"Michaelis--Menten (Case 3): Temporal Evolution", pad=8)
    ax1.set_xlabel(r"Physical Time $t$ ($\mathrm{s}$)", fontsize=11)
    ax1.set_ylabel(r"Concentration $c(t),\, s(t)$ ($\mathrm{mol\,L^{-1}}$)", fontsize=11)
    ax1.set_xlim(0.0, 20.3)
    ax1.set_xticks([0, 5, 10, 15, 20])
    ax1.set_ylim(-0.3, 10.8)
    ax1.legend(bbox_to_anchor=(0.01, 0.22), loc="center left", frameon=True, framealpha=0.95, fontsize=9)
    ax1.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    save(fig1, "fig_mm_intro_time.pdf")
    plt.close(fig1)


def make_fig_mm_intro_phase():
    t_end = 40.0
    t_eval = np.linspace(0.0, t_end, 2000)
    sol_full = solve_ivp(mm_rhs, (0.0, t_end), y0_mm, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)
    sol_pea  = solve_ivp(pea_rhs, (0.0, t_end), y0_mm, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)
    sol_ss   = solve_ivp(lambda t, y: mm_twostage_scalespace_rhs(t, y, 0.25),
                         (0.0, t_end), y0_mm, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)

    c_full, s_full = sol_full.y
    c_pea,  s_pea  = sol_pea.y
    c_ss,   s_ss   = sol_ss.y

    # -------------------------------------------------------------------------
    # 2. Introductory MM Figure B: Phase Space Portrait (fig_mm_intro_phase.pdf)
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6.0, 4.2))

    # Highlight the L-shaped SIM: now vertical branch c ~ e0 and horizontal branch s ~ 0
    ax2.plot(c_full, s_full, "-", color="#2ca02c", lw=6.5, alpha=0.28,
             label=r"Slow Invariant Manifold $\mathcal{M}_{\mathrm{SIM}}$")
    ax2.plot(c_full, s_full, "-", color="#1b7837", lw=2.6, zorder=5,
             label=r"On-Manifold Trajectory $(c(t), s(t))$")

    # Off-manifold trajectories relaxing along fast stoichiometric fibers (c + s = const)
    off_ics = [
        np.array([18.5, 0.35]),
        np.array([15.0, 0.015]),
        np.array([12.0, 0.001]),
        # np.array([11.0, 0.000001]),
        np.array([3.0, 1.15]),
        np.array([6.5, 0.50]),
        np.array([1.0, 1.0]),
    ]
    for k_ic, y_ic in enumerate(off_ics):
        sol_ic = solve_ivp(mm_rhs, (0.0, 0.05), y_ic, t_eval=np.linspace(0.0, 0.05, 2000),
                           method="Radau", rtol=1e-10, atol=1e-12)
        lbl = r"Fast-Fiber Relaxations ($\mathbf{a}_1 \approx [-1, 1]^T$)" if k_ic == 0 else None
        # sol_ic.y[0] is c, sol_ic.y[1] is s
        ax2.plot(sol_ic.y[0], sol_ic.y[1], "-.", color="#d62728", lw=1.7, label=lbl, zorder=4)
        ax2.plot(y_ic[0], y_ic[1], "o", color="#d62728", ms=5.0, zorder=6)
        ax2.annotate("", xy=(sol_ic.y[0, 30], sol_ic.y[1, 30]),
                     xytext=(sol_ic.y[0, 20], sol_ic.y[1, 20]),
                     arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4))

    # Directional arrows on the SIM transposed: (x, y) = (c, s)
    ax2.annotate("", xy=(10.0, 1.), xytext=(10.0, 6.5),
                 arrowprops=dict(arrowstyle="->", color="#1b7837", lw=2.2))
    ax2.annotate("", xy=(5.0, 0.0), xytext=(7.2, 0.0),
                 arrowprops=dict(arrowstyle="->", color="#1b7837", lw=2.2))

    ax2.plot(c_full[0], s_full[0], "o", color="#1f4e78", ms=6.5, zorder=7,
             label=r"Initial State $(c_0, s_0) = (10, 10)$")
    ax2.plot(c_full[1000], s_full[1000], "D", color="#ff7f0e", ms=6.5, zorder=7,
             label=r"Corner Transition ($t \approx 10\mathrm{s}$, $\nu \approx 1$)")
    # ax2.plot(0.0, 0.0, "s", color="#000000", ms=6.5, zorder=7, label=r"Equilibrium $(0,0)$")

    # Transposed text placements: (x, y) = (c, s)
    ax2.text(4.2, 1.5, r"   Regime I" + "\n" + r"$($sQSSA: $c \approx e_0$$)$", color="#b2182b", fontsize=11, fontweight="bold")
    ax2.text(1.5, 1e-7, r"  Regime II" + "\n" + r"$($rQSSA: $s \approx 0$$)$", color="#1b7837", fontsize=11, fontweight="bold")

    # ax2.set_title(r"Michaelis--Menten (Case 3): Phase Space & L-Shaped SIM", pad=8)
    ax2.set_xlabel(r"Complex Concentration $c$ ($\mathrm{mol\,L^{-1}}$)", labelpad=3,fontsize=11)
    ax2.set_ylabel(r"Substrate Concentration $s$ ($\mathrm{mol\,L^{-1}}$)", labelpad=1,fontsize=11)

    ax2.set_xlim(9.5e-1, 20)
    ax2.set_ylim(1e-9, 20.0)
    plt.xscale('log')
    plt.yscale('log')

    # ax2.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=8.3)
    ax2.legend(loc="lower right", ncol=1, frameon=True, framealpha=0.95, fontsize=9.5)
    ax2.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    save(fig2, "fig_mm_intro_phase.pdf")
    plt.close(fig2)


# # Enforce built-in parser without needing system LaTeX
# plt.rcParams.update({
#     "text.usetex": True,
#     "mathtext.fontset": "cm",  # Computer Modern font matches standard TeX
#     "font.family": "serif"
# })


def make_fig_mm_adaptive_timescales():
    t_eval = np.linspace(0.0, 20.3, 2000)
    sol_full = solve_ivp(mm_rhs, (0.0, 20.3), y0_mm, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)
    c_full, s_full = sol_full.y
    nu_t = (e0 - c_full) / (KR + np.maximum(s_full, 0.0))

    tau1_t = np.zeros_like(t_eval)
    tau2_t = np.zeros_like(t_eval)
    sigma_c_t = np.zeros_like(t_eval)
    Dc_adapt = np.zeros_like(t_eval)
    Ds_adapt = np.zeros_like(t_eval)

    for k in range(len(t_eval)):
        lams, taus, A, B = csp_decomp_mm(sol_full.y[:, k])
        tau1_t[k], tau2_t[k] = taus[0], taus[1]
        sig_k = np.sqrt(taus[0] * taus[1])
        sigma_c_t[k] = sig_k
        w1 = np.exp(-((sig_k / taus[0]) ** 2))
        w2 = np.exp(-((sig_k / taus[1]) ** 2))
        d1_diag = A[:, 0] * B[0, :]
        d2_diag = A[:, 1] * B[1, :]
        Dc_adapt[k] = (1.0 - w1) * d1_diag[0] + (1.0 - w2) * d2_diag[0]
        Ds_adapt[k] = (1.0 - w1) * d1_diag[1] + (1.0 - w2) * d2_diag[1]

    # -------------------------------------------------------------------------
    # Figure 1: State-Dependent Timescales & Adaptive Scale
    # -------------------------------------------------------------------------
    fig_ts, ax_ts = plt.subplots(figsize=(6.2, 4))

    ax_ts.axvspan(0.0, 10.0, color="#f8bbd0", alpha=0.25)
    ax_ts.axvspan(10.0, 20.3, color="#c8e6c9", alpha=0.25)
    ax_ts.axvline(10.0, color="#555555", ls="--", lw=1.1)

    ax_ts.semilogy(t_eval, tau2_t, "-", color="#1f4e78", lw=2.5,
                   label=r"Slow Timescale $\tau_2(\mathbf{y}(t))$")
    ax_ts.semilogy(t_eval, sigma_c_t, "-.", color="#2ca02c", lw=2.5,
                   label=r"Adaptive Scale $\sigma_c(t) = \sqrt{\tau_1\tau_2}$")
    ax_ts.semilogy(t_eval, tau1_t, "-", color="#d62728", lw=2.5,
                   label=r"Fast Timescale $\tau_1(\mathbf{y}(t))$")

    ax_ts.text(2.5, 1e-5, r"sQSSA Valid", fontsize=12, fontweight="bold", color="#b2182b")
    ax_ts.text(13.0, 1e-5, r"rQSSA Valid", fontsize=12, fontweight="bold", color="#1b7837")

    # ax_ts.set_title(r"State-Dependent Timescales & Adaptive Scale $\sigma_c(\mathbf{y}(t))$", pad=8)
    ax_ts.set_xlabel(r"Physical Time $t$ ($\mathrm{s}$)", labelpad=4)
    ax_ts.set_ylabel(r"Timescale / Scale ($\mathrm{s}$, log scale)")
    ax_ts.set_xlim(0.0, 20.3)
    ax_ts.set_xticks([0, 5, 10, 15, 20])

    ax_ts.set_ylim(1e-6, 3e8)
    ax_ts.legend(bbox_to_anchor=(1.00, 0.98), loc="upper right", frameon=True, framealpha=0.95, fontsize=11.5)
    ax_ts.grid(True, ls=":", alpha=0.4)

    fig_ts.tight_layout()
    save(fig_ts, "fig_mm_adaptive_timescales.pdf")
    plt.close(fig_ts)


def make_fig_mm_pointers_trajectory():
    t_eval = np.linspace(0.0, 20.3, 2000)
    sol_full = solve_ivp(mm_rhs, (0.0, 23.0), y0_mm, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)
    c_full, s_full = sol_full.y
    nu_t = (e0 - c_full) / (KR + np.maximum(s_full, 0.0))

    tau1_t = np.zeros_like(t_eval)
    tau2_t = np.zeros_like(t_eval)
    sigma_c_t = np.zeros_like(t_eval)
    Dc_adapt = np.zeros_like(t_eval)
    Ds_adapt = np.zeros_like(t_eval)

    for k in range(len(t_eval)):
        lams, taus, A, B = csp_decomp_mm(sol_full.y[:, k])
        tau1_t[k], tau2_t[k] = taus[0], taus[1]
        sig_k = np.sqrt(taus[0] * taus[1])
        sigma_c_t[k] = sig_k
        w1 = np.exp(-((sig_k / taus[0]) ** 2))
        w2 = np.exp(-((sig_k / taus[1]) ** 2))
        d1_diag = A[:, 0] * B[0, :]
        d2_diag = A[:, 1] * B[1, :]
        Dc_adapt[k] = (1.0 - w1) * d1_diag[0] + (1.0 - w2) * d2_diag[0]
        Ds_adapt[k] = (1.0 - w1) * d1_diag[1] + (1.0 - w2) * d2_diag[1]


    # -------------------------------------------------------------------------
    # Figure 2: Adaptive Pointers Along Trajectory
    # -------------------------------------------------------------------------
    fig_pt, ax_pt = plt.subplots(figsize=(6.2, 4))

    ax_pt.axvspan(0.0, 10.0, color="#f8bbd0", alpha=0.25)
    ax_pt.axvspan(10.0, 20.3, color="#c8e6c9", alpha=0.25)
    ax_pt.axvline(10.0, color="#555555", ls="--", lw=1.1)

    ax_pt.plot(t_eval, Dc_adapt, "-", color="#d62728", lw=2.6,
               label=r"$D^c(\mathbf{y}(t); \sigma_c(t)) = S^s(\mathbf{y}(t); \sigma_c(t))$")
    ax_pt.plot(t_eval, Ds_adapt, "-", color="#1f4e78", lw=2.6,
               label=r"$D^s(\mathbf{y}(t); \sigma_c(t)) = S^c(\mathbf{y}(t); \sigma_c(t))$")
    ax_pt.plot(t_eval, 1.0 / (1.0 + nu_t), ":", color="#000000", lw=1.8,
               label=r"Analytical PEA Weights $\frac{1}{1+\nu},\, \frac{\nu}{1+\nu}$")
    ax_pt.plot(t_eval, nu_t / (1.0 + nu_t), ":", color="#000000", lw=1.8)
    ax_pt.plot(t_eval, Dc_adapt + Ds_adapt, "-.", color="#2ca02c", lw=1.8,
               label=r"Trace $\sum_k D^k(\sigma_c(t)) = 1.000$")

    ax_pt.text(2.5, 0.06, r"sQSSA Valid", fontsize=12, fontweight="bold", color="#b2182b")
    ax_pt.text(13.0, 0.06, r"rQSSA Valid", fontsize=12, fontweight="bold", color="#1b7837")

    # ax_pt.set_title(r"Dynamical Role Inversion at Adaptive $\sigma_c(\mathbf{y}(t))$", pad=8)
    ax_pt.set_xlabel(r"Physical Time $t$ ($\mathrm{s}$)", labelpad=4)
    ax_pt.set_ylabel(r"Pointer Value at Adaptive $\sigma_c(t)$")
    ax_pt.set_xlim(0.0, 20.3)
    ax_pt.set_xticks([0, 5, 10, 15, 20])

    ax_pt.set_ylim(-0.05, 1.08)
    ax_pt.legend(bbox_to_anchor=(0.00, 0.22), loc="lower left", frameon=True, framealpha=0.95, fontsize=12)
    ax_pt.grid(True, ls=":", alpha=0.4)

    fig_pt.tight_layout()
    save(fig_pt, "fig_mm_pointers_trajectory.pdf")
    plt.close(fig_pt)


def make_fig_mm_pointers_and_timescales():
    t_eval = np.linspace(0.0, 20.0, 2000)
    sol_full = solve_ivp(mm_rhs, (0.0, 20.0), y0_mm, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)
    c_full, s_full = sol_full.y
    nu_t = (e0 - c_full) / (KR + np.maximum(s_full, 0.0))

    tau1_t = np.zeros_like(t_eval)
    tau2_t = np.zeros_like(t_eval)
    sigma_c_t = np.zeros_like(t_eval)
    Dc_adapt = np.zeros_like(t_eval)
    Ds_adapt = np.zeros_like(t_eval)

    for k in range(len(t_eval)):
        lams, taus, A, B = csp_decomp_mm(sol_full.y[:, k])
        tau1_t[k], tau2_t[k] = taus[0], taus[1]
        sig_k = np.sqrt(taus[0] * taus[1])
        sigma_c_t[k] = sig_k
        w1 = np.exp(-((sig_k / taus[0]) ** 2))
        w2 = np.exp(-((sig_k / taus[1]) ** 2))
        d1_diag = A[:, 0] * B[0, :]
        d2_diag = A[:, 1] * B[1, :]
        Dc_adapt[k] = (1.0 - w1) * d1_diag[0] + (1.0 - w2) * d2_diag[0]
        Ds_adapt[k] = (1.0 - w1) * d1_diag[1] + (1.0 - w2) * d2_diag[1]


    # -------------------------------------------------------------------------
    # Figures 3, 4, 5: Scale-Dependent Pointers at t = 2s, 10s, 15s
    # -------------------------------------------------------------------------
    sigma_grid = np.logspace(-6.0, 7.5, 600)
    snapshots = [
        (2.0,  "fig_mm_pointers_t2.pdf"),
        (10.0, "fig_mm_pointers_t10.pdf"),
        (15.0, "fig_mm_pointers_t15.pdf"),
    ]

    for t_snap, fname in snapshots:
        fig_s, ax = plt.subplots(figsize=(6.2, 4))
        idx = np.argmin(np.abs(t_eval - t_snap))
        y_pt = sol_full.y[:, idx]
        lams, taus, A, B = csp_decomp_mm(y_pt)
        tau1, tau2 = taus[0], taus[1]
        sig_c_loc = np.sqrt(tau1 * tau2)

        w1_s = np.exp(-((sigma_grid / tau1) ** 2))
        w2_s = np.exp(-((sigma_grid / tau2) ** 2))

        d1_diag = A[:, 0] * B[0, :]
        d2_diag = A[:, 1] * B[1, :]

        Dc_s = (1.0 - w1_s) * d1_diag[0] + (1.0 - w2_s) * d2_diag[0]
        Ds_s = (1.0 - w1_s) * d1_diag[1] + (1.0 - w2_s) * d2_diag[1]
        Sc_s = w1_s * d1_diag[0] + w2_s * d2_diag[0]
        Ss_s = w1_s * d1_diag[1] + w2_s * d2_diag[1]

        ax.axvspan(tau1 * 3.0, tau2 / 3.0, color="#2ca02c", alpha=0.12,
                   label=r"1D SIM Window ($\tau_1 \ll \sigma \ll \tau_2$)")
        ax.axvline(tau1, color="#d62728", ls=":", lw=1.3)
        ax.axvline(tau2, color="#1f4e78", ls=":", lw=1.3)
        ax.axvline(sig_c_loc, color="#2ca02c", ls="-.", lw=1.6,
                   label=r"Adaptive $\sigma_c(t) = \sqrt{\tau_1(t)\tau_2(t)}$")

        ax.semilogx(sigma_grid, Dc_s, "-",  color="#d62728", lw=2.4, label=r"Fast Pointer $D^c(\sigma)$")
        ax.semilogx(sigma_grid, Ds_s, "-",  color="#1f4e78", lw=2.4, label=r"Fast Pointer $D^s(\sigma)$")
        ax.semilogx(sigma_grid, Sc_s, "--", color="#d62728", lw=1.8, label=r"Slow Pointer $S^c(\sigma)$")
        ax.semilogx(sigma_grid, Ss_s, "--", color="#1f4e78", lw=1.8, label=r"Slow Pointer $S^s(\sigma)$")

        ax.text(tau1 * 0.05, 0.52, r"$\tau_1(t)$", color="#d62728", fontsize=12, fontweight="bold"
               ,bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=1)
               )
        ax.text(sig_c_loc * 0.15, 0.82, rf"$\sigma_c={sig_c_loc:.2g}\,\mathrm{{s}}$",
                color="#1b7837", fontsize=11.5, fontweight="bold"
               ,bbox=dict(boxstyle="round,pad=0.1", facecolor="#e5f3e5", edgecolor="none", alpha=1)
               )
        ax.text(tau2 * 1.25, 0.52, r"$\tau_2(t)$", color="#1f4e78", fontsize=12, fontweight="bold"
               ,bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=1)
               )

        # ax.set_title(rf"Scale-Dependent Pointers at $t = {t_snap}\,\mathrm{{s}}$", pad=8)
        ax.set_xlabel(r"Observation Scale $\sigma$ ($\mathrm{s}$, log scale)")
        ax.set_ylabel(r"Scale-Dependent Pointer")
        ax.set_xlim(1e-6, 10**(7.5))
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, ls=":", alpha=0.4)
        if t_snap == 2.0:
            ax.legend(bbox_to_anchor=(0.50, 0.05), loc="lower center", frameon=True, framealpha=0.95, fontsize=11)

        fig_s.tight_layout()
        save(fig_s, fname)
        plt.close(fig_s)


def make_fig_mm_model_comparison(sigma_c="adaptive"):
    t_eval = np.linspace(0.0, 20.3, 2000)
    sol_full = solve_ivp(mm_rhs, (0.0, 20.3), y0_mm, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)
    sol_pea  = solve_ivp(pea_rhs, (0.0, 20.3), y0_mm, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)
    sol_ss   = solve_ivp(lambda t, y: mm_twostage_scalespace_rhs(t, y, sigma_c),
                         (0.0, 20.3), y0_mm, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)

    c_full, s_full = sol_full.y
    c_pea,  s_pea  = sol_pea.y
    c_ss,   s_ss   = sol_ss.y

    c_sqssa_alg = e0 * np.maximum(s_full, 0.0) / (KM + np.maximum(s_full, 0.0))
    s_rqssa_alg = KR * c_full / np.maximum(e0 - c_full, 1e-12)
    idx_sub = np.linspace(0, len(t_eval) - 1, 35, dtype=int)

    # -------------------------------------------------------------------------
    # Figure 1: Comparison with sQSSA and PEA
    # -------------------------------------------------------------------------
    fig_sqssa, ax_l = plt.subplots(figsize=(6.2, 3.5))

    ax_l.axvspan(0.0, 10.0, color="#f8bbd0", alpha=0.25)
    ax_l.axvspan(10.0, 20.3, color="#c8e6c9", alpha=0.25)
    ax_l.axvline(10.0, color="#555555", ls="--", lw=1.1)

    ax_l.semilogy(t_eval, c_full, "-", color="#333333", lw=2.6, label=r"Full Model $c(t),\, s(t)$")
    ax_l.semilogy(t_eval, np.maximum(s_full, 1e-8), "-", color="#333333", lw=2.6)
    ax_l.semilogy(t_eval, c_ss, "-", color="#2ca02c", lw=2.0, label=r"CSP Scale-Space")
    ax_l.semilogy(t_eval, np.maximum(s_ss, 1e-8), "-", color="#2ca02c", lw=2.0)
    ax_l.semilogy(t_eval[idx_sub], c_pea[idx_sub], "+", color="#ff7f0e", ms=7, mew=1.6, label=r"PEA Model")
    ax_l.semilogy(t_eval[idx_sub], np.maximum(s_full[idx_sub], 1e-8), "+", color="#ff7f0e", ms=7, mew=1.6)
    ax_l.semilogy(t_eval, c_sqssa_alg, "--", color="#d62728", lw=2.2,
                  label=r"sQSSA $c(t)$ " + "\n" + r"(Diverges for $t > 10\,\mathrm{s}$)")
    ax_l.semilogy(t_eval, np.maximum(s_full, 1e-8), "--", color="#1f4e78", lw=2.0, label=r"sQSSA $s(t)$")

    ax_l.text(3.0, 7e-7, r"sQSSA Valid", fontsize=12, fontweight="bold", color="#b2182b")
    ax_l.text(12.8, 7e-7, r"sQSSA Not Valid", fontsize=12, fontweight="bold", color="#1b7837")
    # ax_l.set_title(r"CSP Scale-Space vs. sQSSA, PEA, and Full Model", pad=8)
    ax_l.set_xlabel(r"Physical Time $t$ ($\mathrm{s}$)", fontsize=11)
    ax_l.set_ylabel(r"Concentration ($\mathrm{mol\,L^{-1}}$, log scale)", fontsize=11)
    ax_l.set_xlim(0.0, 20.3)
    ax_l.set_xticks([0, 5, 10, 15, 20])
    ax_l.set_ylim(5e-7, 25.0)
    ax_l.legend(bbox_to_anchor=(0.00, 0.05), loc="lower left", frameon=True, framealpha=0.95, fontsize=12)
    ax_l.grid(True, ls=":", alpha=0.4)

    fig_sqssa.tight_layout()
    save(fig_sqssa, f"fig_mm_comparison_sqssa_{sigma_c}.pdf")
    plt.close(fig_sqssa)

    # -------------------------------------------------------------------------
    # Figure 2: Comparison with rQSSA and PEA
    # -------------------------------------------------------------------------
    fig_rqssa, ax_r = plt.subplots(figsize=(6.2, 3.5))

    ax_r.axvspan(0.0, 10.0, color="#f8bbd0", alpha=0.25)
    ax_r.axvspan(10.0, 20.3, color="#c8e6c9", alpha=0.25)
    ax_r.axvline(10.0, color="#555555", ls="--", lw=1.1)

    ax_r.semilogy(t_eval, c_full, "-", color="#333333", lw=2.6, label=r"Full Model $c(t),\, s(t)$")
    ax_r.semilogy(t_eval, np.maximum(s_full, 1e-8), "-", color="#333333", lw=2.6)
    ax_r.semilogy(t_eval, c_ss, "-", color="#2ca02c", lw=2.0, label=r"CSP Scale-Space")
    ax_r.semilogy(t_eval, np.maximum(s_ss, 1e-8), "-", color="#2ca02c", lw=2.0)
    ax_r.semilogy(t_eval[idx_sub], c_pea[idx_sub], "+", color="#ff7f0e", ms=7, mew=1.6, label=r"PEA Model")
    ax_r.semilogy(t_eval[idx_sub], np.maximum(s_full[idx_sub], 1e-8), "+", color="#ff7f0e", ms=7, mew=1.6)
    ax_r.semilogy(t_eval, c_full, "--", color="#d62728", lw=2.0, label=r"rQSSA $c(t)$")
    ax_r.semilogy(t_eval, np.minimum(s_rqssa_alg, 12.0), "--", color="#1f4e78", lw=2.2,
                  label=r"rQSSA $s(t)$ " + "\n" + r"(Invalid for $t < 10\,\mathrm{s}$)")

    ax_r.text(2.5, 7e-7, r"rQSSA Not Valid", fontsize=12, fontweight="bold", color="#b2182b")
    ax_r.text(13.2, 7e-7, r"rQSSA Valid", fontsize=12, fontweight="bold", color="#1b7837")
    # ax_r.set_title(r"CSP Scale-Space vs. rQSSA, PEA, and Full Model", pad=8)
    ax_r.set_xlabel(r"Physical Time $t$ ($\mathrm{s}$)", fontsize=11)
    ax_r.set_ylabel(r"Concentration ($\mathrm{mol\,L^{-1}}$, log scale)", fontsize=11)
    ax_r.set_xlim(0.0, 20.3)
    ax_r.set_xticks([0, 5, 10, 15, 20])
    ax_r.set_ylim(5e-7, 25.0)
    ax_r.legend(bbox_to_anchor=(0.00, 0.05), loc="lower left", frameon=True, framealpha=0.95, fontsize=12)
    ax_r.grid(True, ls=":", alpha=0.4)

    fig_rqssa.tight_layout()
    save(fig_rqssa, f"fig_mm_comparison_rqssa_{sigma_c}.pdf")
    plt.close(fig_rqssa)
