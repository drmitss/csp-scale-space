"""Figures 11-13: three-scale reaction network (Section 6.2).

Ported from the notebook that produced the manuscript figures; the annotations
are specific to the paper's models and parameters.  For the same analysis of
an arbitrary model use :class:`csp_scalespace.ScaleSpaceStudy` instead.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.integrate import solve_ivp

from _common import *  # noqa: F401,F403  (models, wrappers, style, save)

# =============================================================================
# DATA INTEGRATION PIPELINE
# =============================================================================
def compute_3d_network_simulations():
    """Runs all fiber relaxation and time integrations, returning a data dictionary."""
    # 1. Integrate Formulation B Fiber Relaxation across sigma in [1e-5, 20]
    u_span = (np.log(1e-5), np.log(20.0))
    u_eval = np.linspace(u_span[0], u_span[1], 600)
    sol_fib = solve_ivp(net3d_fiber_log_rhs, u_span, y0_3d, t_eval=u_eval,
                        method="Radau", rtol=1e-10, atol=1e-12)
    sigma_fib = np.exp(u_eval)

    # Extract Stage 1 relaxed initial states on the 2D SIM (M=1) and 1D SIM (M=2)
    _, taus0, _, _ = csp_decomp_3d(y0_3d)
    sig_c1_0 = np.sqrt(taus0[0] * taus0[1])
    sig_c2_0 = np.sqrt(taus0[1] * taus0[2])

    idx_c1 = np.argmin(np.abs(sigma_fib - sig_c1_0))
    idx_c2 = np.argmin(np.abs(sigma_fib - sig_c2_0))
    y_init_M1 = sol_fib.y[:, idx_c1]
    y_init_M2 = sol_fib.y[:, idx_c2]

    # 2. Integrate Full 3D ODE, 2D SIM Scale-Space (M=1), and 1D SIM Scale-Space (M=2)
    t_eval = np.logspace(-5.0, np.log10(20.0), 700)
    t_eval[-1] = 20.0  # Guard against precision overshoot beyond t_span[1]
    t_span = (0.0, 20.0)

    sol_full = solve_ivp(g_rev, t_span, y0_3d, t_eval=t_eval,
                         method="Radau", rtol=1e-11, atol=1e-13)
    sol_ss_M1 = solve_ivp(lambda t, y: net3d_twostage_scalespace_rhs(t, y, level=1),
                          t_span, y_init_M1, t_eval=t_eval,
                          method="Radau", rtol=1e-10, atol=1e-12)
    sol_ss_M2 = solve_ivp(lambda t, y: net3d_twostage_scalespace_rhs(t, y, level=2),
                          t_span, y_init_M2, t_eval=t_eval,
                          method="Radau", rtol=1e-10, atol=1e-12)

    return {
        "t_eval": t_eval,
        "sol_full": sol_full,
        "sol_ss_M1": sol_ss_M1,
        "sol_ss_M2": sol_ss_M2,
        "sol_fib": sol_fib,
        "y_init_M1": y_init_M1,
        "y_init_M2": y_init_M2,
    }


def generate_figure_2a():
    # y_state = np.array([1.0, 0.6, 0.5])
    y_state = y0_3d
    J = jac_rev(y_state)
    eigvals, eigvecs = np.linalg.eig(J)
    idx = np.argsort(np.abs(eigvals))[::-1]
    lam = eigvals[idx].real
    A0 = eigvecs[:, idx].real
    B0 = np.linalg.inv(A0)
    tau = 1.0 / np.abs(lam)

    sig_grid = np.logspace(-5.5, 2.0, 600)
    D_pointers = np.zeros((3, len(sig_grid)))
    S_pointers = np.zeros((3, len(sig_grid)))
    mode_pointers = np.array([A0[:, i] * B0[i, :] for i in range(3)])

    for idx_s, sig in enumerate(sig_grid):
        w = np.exp(-(sig / tau)**2)
        for k in range(3):
            S_pointers[k, idx_s] = np.sum(w * mode_pointers[:, k])
            D_pointers[k, idx_s] = np.sum((1.0 - w) * mode_pointers[:, k])

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    colors_var = ['#2ca02c', '#1f4e78', '#d62728']

    labels_fast = [r'$D^1(\mathbf{y}_0;\sigma)$ ($y^1$)', r'$D^2(\mathbf{y}_0;\sigma)$ ($y^2$)', r'$D^3(\mathbf{y}_0;\sigma)$ ($y^3$)']
    labels_slow = [r'$S^1(\mathbf{y}_0;\sigma)$ ($y^1$)', r'$S^2(\mathbf{y}_0;\sigma)$ ($y^2$)', r'$S^3(\mathbf{y}_0;\sigma)$ ($y^3$)']

    for k in range(3):
        ax.semilogx(sig_grid, D_pointers[k, :], color=colors_var[k], ls='-', lw=2.1, label=labels_fast[k])
        ax.semilogx(sig_grid, S_pointers[k, :], color=colors_var[k], ls='--', lw=1.6, alpha=0.85, label=labels_slow[k])

    for m_idx, t_val in enumerate(tau):
        ax.axvline(t_val, color='k', ls=':', lw=1.2, alpha=0.7)
        ax.text(t_val * 1.15, 0.93, rf'$\tau_{m_idx+1}$', fontsize=15
                          ,bbox=dict(boxstyle="round,pad=0.0", facecolor="white", edgecolor="none", alpha=0.9))

    ax.axvspan(2.5e-3, 1.2e-2, color='#d9e8f5', alpha=0.5, label=r'2D SIM window' + '\n' + r'($\tau_1 \ll \sigma \ll \tau_2$)')
    ax.axvspan(0.18, 0.9, color='#e2f0d9', alpha=0.5, label=r'1D SIM window' + '\n' + r'($\tau_2 \ll \sigma \ll \tau_3$)')

    ax.set_xlabel(r'Observation scale parameter $\sigma$')
    ax.set_ylabel(r'Scale-Dependent Pointers $D^k(\sigma), S^k(\sigma)$')
    # ax.set_title(r'Continuous Fast (Solid) & Slow (Dashed) CSP Pointers')
    ax.set_xlim(10**-5.5, 10**2.0)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, which='both', ls=':', alpha=0.5)
    # ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0), ncol=1, frameon=True, fontsize=12)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0), ncol=1, frameon=True, fontsize=12, handlelength=1.0)

    plt.tight_layout()
    save(plt.gcf(), 'fig2a_3d_continuous_pointers.pdf', dpi=400, bbox_inches='tight')
    plt.close(fig)


def generate_figure_2b():
    # y_state = np.array([1.0, 0.6, 0.5])
    y_state = y0_3d
    J = jac_rev(y_state)
    eigvals, _ = np.linalg.eig(J)
    idx = np.argsort(np.abs(eigvals))[::-1]
    lam = eigvals[idx].real
    tau = 1.0 / np.abs(lam)

    def rev_fiber_log_rhs(u, y):
        sigma = np.exp(u)
        J_loc = jac_rev(y)
        ev, evec = np.linalg.eig(J_loc)
        ix = np.argsort(np.abs(ev))[::-1]
        lam_loc = ev[ix].real
        A_loc = evec[:, ix].real
        B_loc = np.linalg.inv(A_loc)
        tau_loc = 1.0 / np.abs(lam_loc)
        f_loc = B_loc @ g_rev(None,y)
        d_i = (2.0 * sigma) / (tau_loc**2 + sigma**2)
        coeff = sigma * (d_i / (-lam_loc))
        return A_loc @ (coeff * f_loc)

    u_eval = np.linspace(np.log(1e-5), np.log(20.0), 500)
    sol_fiber_3d = solve_ivp(
        rev_fiber_log_rhs, [u_eval[0], u_eval[-1]], y_state,
        method='Radau', t_eval=u_eval, rtol=1e-9, atol=1e-11
    )
    sig_fiber_3d = np.exp(sol_fiber_3d.t)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    colors_var = ['#2ca02c', '#1f4e78', '#d62728']


    for k in range(3):
        ax.semilogx(sig_fiber_3d, sol_fiber_3d.y[k, :], color=colors_var[k], lw=2.2, label=rf'Relaxed state $y_\sigma^{k+1}$')

    for m_idx, t_val in enumerate(tau):
        lbl = r'Intrinsic timescales $\tau_1, \tau_2, \tau_3$' if m_idx == 0 else None
        ax.axvline(t_val, color='k', ls=':', lw=1.2, alpha=0.7, label=lbl)
        ax.text(t_val * 1.15, 0.95, rf'$\tau_{m_idx+1}$', fontsize=15)

    ax.axvspan(2.5e-3, 1.2e-2, color='#d9e8f5', alpha=0.5, label=r'2D SIM window' + '\n' + r'($\tau_1 \ll \sigma \ll \tau_2$)')
    ax.axvspan(0.18, 0.9, color='#e2f0d9', alpha=0.5, label=r'1D SIM window' + '\n' + r'($\tau_2 \ll \sigma \ll \tau_3$)')

    ax.set_xlabel(r'Observation scale parameter $\sigma$')
    ax.set_ylabel(r'Scale-Relaxed State Variables $y_\sigma^k$')
    # ax.set_title(r'Formulation B Hierarchical State Fiber Relaxation $\mathbf{y}_\sigma$')
    ax.set_xlim(1e-5, 20.0)
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(loc='lower left', frameon=True, fontsize=12, ncol=2, handlelength=1.0)

    plt.tight_layout()
    save(plt.gcf(), 'fig2b_3d_fiber_relaxation.pdf', dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_figure_1_time_comparison(save_path="fig_3d_time_comparison.pdf"):
    """Figure 13(a): Multi-Resolution Temporal Comparison (Logarithmic Time t)."""
    u_span = (np.log(1e-5), np.log(20.0))
    u_eval = np.linspace(u_span[0], u_span[1], 600)
    sol_fib = solve_ivp(net3d_fiber_log_rhs, u_span, y0_3d, t_eval=u_eval,
                        method="Radau", rtol=1e-10, atol=1e-12)
    sigma_fib = np.exp(u_eval)

    _, taus0, _, _ = csp_decomp_3d(y0_3d)
    idx_c1 = np.argmin(np.abs(sigma_fib - np.sqrt(taus0[0] * taus0[1])))
    idx_c2 = np.argmin(np.abs(sigma_fib - np.sqrt(taus0[1] * taus0[2])))
    y_init_M1 = sol_fib.y[:, idx_c1]
    y_init_M2 = sol_fib.y[:, idx_c2]

    t_eval = np.logspace(-5.0, np.log10(20.0), 700)
    t_eval[-1] = 20.0
    t_span = (0.0, 20.0)

    sol_full = solve_ivp(g_rev, t_span, y0_3d, t_eval=t_eval,
                         method="Radau", rtol=1e-11, atol=1e-13)
    sol_M1 = solve_ivp(lambda t, y: net3d_twostage_scalespace_rhs(t, y, level=1),
                       t_span, y_init_M1, t_eval=t_eval,
                       method="Radau", rtol=1e-10, atol=1e-12)
    sol_M2 = solve_ivp(lambda t, y: net3d_twostage_scalespace_rhs(t, y, level=2),
                       t_span, y_init_M2, t_eval=t_eval,
                       method="Radau", rtol=1e-10, atol=1e-12)

    fig_t, ax_t = plt.subplots(figsize=(6.2, 4.2))

    ax_t.axvspan(20, 21, color="mistyrose", alpha=0.45, label=r"Ultra-Fast Layer ($t \lesssim \tau_1$)")
    ax_t.axvspan(20, 21, color="#d9e8f5", alpha=0.45, label=r"2D SIM Regime ($\tau_1 \ll t \ll \tau_2$)")
    ax_t.axvspan(20, 21, color="#e2f0d9", alpha=0.45, label=r"1D SIM Regime ($\tau_2 \ll t \ll \tau_3$)")

    shade_regimes(ax_t)


    # Plot species y^1, y^2, y^3 for Full (solid), Scale-1 (dashed), Scale-2 (dotted)
    colors = ["#2ca02c", "#1f4e78", "#d62728"]
    names = [r"$y^1(t)$", r"$y^2(t)$", r"$y^3(t)$"]
    for k in range(3):
        lbl_f = rf"Full 3D {names[k]}" if k == 1 else None
        lbl_1 = rf"Scale-1 $\mathbf{{g}}_{{\sigma_{{c,1}}}}$ ($M=1$)" if k == 1 else None
        lbl_2 = rf"Scale-2 $\mathbf{{g}}_{{\sigma_{{c,2}}}}$ ($M=2$)" if k == 1 else None
        ax_t.semilogx(t_eval, sol_full.y[k], "-", color=colors[k], lw=2, label=lbl_f)
        ax_t.semilogx(t_eval, sol_M1.y[k], "--", color=colors[k], lw=3, label=lbl_1)
        ax_t.semilogx(t_eval[t_eval >= 0.08], sol_M2.y[k, t_eval >= 0.08], ":", color=colors[k], lw=4, label=lbl_2)

    ax_t.text(4e-5, 1.08, r"$M(t)=0$", fontsize=11, fontweight="bold", color="#d62728"
              ,bbox=dict(boxstyle="round,pad=0.1", facecolor="#fff3f1", edgecolor="none", alpha=1)
              )
    ax_t.text(1.2e-3, 1.08, r"$M(t)=1$ $($2D SIM$)$", fontsize=11, fontweight="bold", color="#1f4e78"
              ,bbox=dict(boxstyle="round,pad=0.1", facecolor="#edf4fa", edgecolor="none", alpha=1)
              )
    ax_t.text(1.e-1, 1.08, r"$M(t)=2$ $($1D SIM$)$", fontsize=11, fontweight="bold", color="#1b7837"
              ,bbox=dict(boxstyle="round,pad=0.1", facecolor="#f1f8ed", edgecolor="none", alpha=1)
              )

    ax_t.text(0.9, 0.90, r"$y^1(t)$", color="#2ca02c", fontsize=11.5, fontweight="bold")
    ax_t.text(0.6, 0.53, r"$y^2(t)$", color="#1f4e78", fontsize=11.5, fontweight="bold")
    ax_t.text(0.12, 0.34, r"$y^3(t)$", color="#d62728", fontsize=11.5, fontweight="bold")

    ax_t.set_xlabel(r"Physical Time $t$ ($\mathrm{s}$, log scale)")
    ax_t.set_ylabel(r"Concentrations $y^1(t),\, y^2(t),\, y^3(t)$")
    ax_t.set_xlim(1e-5, 20.0)
    ax_t.set_ylim(-0.03, 1.18)
    ax_t.legend(bbox_to_anchor=(0.00, 0.0), loc="lower left", ncol=2, frameon=True, framealpha=0.85, fontsize=9.8)
    # ax_t.grid(True, ls=":", alpha=0.4)
    ax_t.grid(True, which='both', ls=':', alpha=0.2)

    fig_t.tight_layout()
    save(fig_t, save_path)
    plt.close(fig_t)


def plot_figure_2_dynamic_timescales(save_path="fig_3d_dynamic_timescales.pdf"):
    """Figure 13(b): Evolving Local Timescales & Adaptive Gap Scales Along y(t)."""
    t_eval = np.logspace(-5.0, np.log10(20.0), 700)
    t_eval[-1] = 20.0
    sol_full = solve_ivp(g_rev, (0.0, 20.0), y0_3d, t_eval=t_eval,
                         method="Radau", rtol=1e-11, atol=1e-13)

    tau_traj = np.zeros((3, len(t_eval)))
    sig_c1_t = np.zeros(len(t_eval))
    sig_c2_t = np.zeros(len(t_eval))

    for k in range(len(t_eval)):
        _, taus_k, _, _ = csp_decomp_3d(sol_full.y[:, k])
        tau_traj[:, k] = taus_k
        sig_c1_t[k] = np.sqrt(taus_k[0] * taus_k[1])
        sig_c2_t[k] = np.sqrt(taus_k[1] * taus_k[2])

    fig_ts, ax_ts = plt.subplots(figsize=(6.2, 4.2))

    # Unified regime background palette
    # ax_ts.axvspan(1e-5, 2e-3, color="#d62728", alpha=0.05)
    # ax_ts.axvspan(2e-3, 1.5e-1, color="#d9e8f5", alpha=0.45)
    # ax_ts.axvspan(1.5e-1, 10.0, color="#e2f0d9", alpha=0.45)

    shade_regimes(ax_ts)

    ax_ts.loglog(t_eval, tau_traj[2], "-", color="#2ca02c", lw=2.4, label=r"Slow Timescale $\tau_3(\mathbf{y}(t))$")
    ax_ts.loglog(t_eval, sig_c2_t, "-.", color="#ff7f0e", lw=2.2,
                 label=r"1D SIM Adaptive Scale $\sigma_{c,2}(t) = \sqrt{\tau_2\tau_3}$")
    ax_ts.loglog(t_eval, tau_traj[1], "-", color="#1f4e78", lw=2.4, label=r"Intermediate Timescale $\tau_2(\mathbf{y}(t))$")
    ax_ts.loglog(t_eval, sig_c1_t, "-.", color="mediumpurple", lw=2.2,
                 label=r"2D SIM Adaptive Scale $\sigma_{c,1}(t) = \sqrt{\tau_1\tau_2}$")
    ax_ts.loglog(t_eval, tau_traj[0], "-", color="#d62728", lw=2.4, label=r"Ultra-Fast Timescale $\tau_1(\mathbf{y}(t))$")

    ax_ts.text(4e-5, 6, r"$M(t)=0$", fontsize=11, fontweight="bold", color="#d62728"
              ,bbox=dict(boxstyle="round,pad=0.1", facecolor="#fff3f1", edgecolor="none", alpha=1)
              )
    ax_ts.text(1.2e-3, 6, r"$M(t)=1$ $($2D SIM$)$", fontsize=11, fontweight="bold", color="#1f4e78"
              ,bbox=dict(boxstyle="round,pad=0.1", facecolor="#edf4fa", edgecolor="none", alpha=1)
              )
    ax_ts.text(1.e-1, 6, r"$M(t)=2$ $($1D SIM$)$", fontsize=11, fontweight="bold", color="#1b7837"
              ,bbox=dict(boxstyle="round,pad=0.1", facecolor="#f1f8ed", edgecolor="none", alpha=1)
              )


    ax_ts.set_xlabel(r"Physical Time $t$ ($\mathrm{s}$, log scale)")
    ax_ts.set_ylabel(r"Timescale / Adaptive Scale ($\mathrm{s}$, log scale)")
    ax_ts.set_xlim(1e-5, 20.0)
    ax_ts.set_ylim(2e-4, 15.0)
    ax_ts.legend(bbox_to_anchor=(0.01, 0.0), loc="lower left", frameon=True, framealpha=0.95, fontsize=10)
    # ax_ts.grid(True, ls=":", alpha=0.4)
    ax_ts.grid(True, which='both', ls=':', alpha=0.2)

    fig_ts.tight_layout()
    save(fig_ts, save_path)
    plt.close(fig_ts)


def plot_figure_3_dynamic_pointers(save_path="fig_3d_dynamic_pointers.pdf"):
    """Figure 13(c): Real-Time Mode Exhaustion Along Trajectory (sigma = t)."""
    t_eval = np.logspace(-5.0, np.log10(20.0), 700)
    t_eval[-1] = 20.0
    sol_full = solve_ivp(g_rev, (0.0, 20.0), y0_3d, t_eval=t_eval,
                         method="Radau", rtol=1e-11, atol=1e-13)

    D_realtime = np.zeros((3, len(t_eval)))

    for m in range(len(t_eval)):
        t_now = t_eval[m]
        _, taus, A, B = csp_decomp_3d(sol_full.y[:, m])
        w_t = np.exp(-((t_now / taus) ** 2))
        for i in range(3):
            D_realtime[:, m] += (1.0 - w_t[i]) * (A[:, i] * B[i, :])

    fig_rt, ax_rt = plt.subplots(figsize=(6.2, 4.2))

    # ax_rt.axvspan(1e-5, 2e-3, color="#d62728", alpha=0.05)
    # ax_rt.axvspan(2e-3, 1.5e-1, color="#d9e8f5", alpha=0.45)
    # ax_rt.axvspan(1.5e-1, 10.0, color="#e2f0d9", alpha=0.45)

    shade_regimes(ax_rt)

    ax_rt.semilogx(t_eval, D_realtime[0], "-", color="#2ca02c", lw=2.2, label=r"$D^1(\mathbf{y}(t); \sigma=t)$")
    ax_rt.semilogx(t_eval, D_realtime[1], "-", color="#1f4e78", lw=2.2, label=r"$D^2(\mathbf{y}(t); \sigma=t)$")
    ax_rt.semilogx(t_eval, D_realtime[2], "-", color="#d62728", lw=2.2, label=r"$D^3(\mathbf{y}(t); \sigma=t)$")

    # ax_rt.semilogx(t_eval, np.sum(D_realtime, axis=0) / 3.0, "-.", color="#000000", lw=2.4,
                   # label=r"Normalized Trace $\frac{1}{3}\sum_{k=1}^3 D^k(\mathbf{y}(t); \sigma=t)$")
    ax_rt.semilogx(t_eval, np.sum(D_realtime, axis=0), "-", color="#000000", lw=2.4,
                   label=r"Active Fast Dimension $\sum_{k=1}^3 D^k(\mathbf{y};\, t)$")

    ax_rt.text(4e-5, 3.09, r"$M(t)=0$", fontsize=11, fontweight="bold", color="#d62728"
              ,bbox=dict(boxstyle="round,pad=0.1", facecolor="#fff3f1", edgecolor="none", alpha=1)
              )
    ax_rt.text(1.2e-3, 3.09, r"$M(t)=1$ $($2D SIM$)$", fontsize=11, fontweight="bold", color="#1f4e78"
              ,bbox=dict(boxstyle="round,pad=0.1", facecolor="#edf4fa", edgecolor="none", alpha=1)
              )
    ax_rt.text(1.e-1, 3.09, r"$M(t)=2$ $($1D SIM$)$", fontsize=11, fontweight="bold", color="#1b7837"
              ,bbox=dict(boxstyle="round,pad=0.1", facecolor="#f1f8ed", edgecolor="none", alpha=1)
              )


    ax_rt.set_xlabel(r"Physical Time $t$ with $\sigma = t$ ($\mathrm{s}$, log scale)")
    # ax_rt.set_ylabel(r"Real-Time Fast Pointer $D^k(\mathbf{y}(t); \sigma=t)$")
    ax_rt.set_ylabel(r"Fast Pointers $D^k$ & Active $M(t)$")


    ax_rt.set_xlim(1e-5, 20.0)
    ax_rt.set_ylim(-0.05, 3.3)
    ax_rt.legend(bbox_to_anchor=(0.00, 0.52), loc="lower left", frameon=True, framealpha=0.95, fontsize=10.0)
    # ax_rt.grid(True, ls=":", alpha=0.4)
    ax_rt.grid(True, which='both', ls=':', alpha=0.2)

    fig_rt.tight_layout()
    save(fig_rt, save_path)
    plt.close(fig_rt)


def plot_figure_4_sim_trajectories_3d(sim_data, save_path="fig_3d_sim_trajectories.pdf"):
    """Figure 4: 3D Phase-Space SIM Surface, 1D SIM Curve, Full & Scale-Space Trajectories."""
    sol_full = sim_data["sol_full"]
    sol_fib = sim_data["sol_fib"]
    sol_ss_M1 = sim_data["sol_ss_M1"]
    sol_ss_M2 = sim_data["sol_ss_M2"]
    y_init_M1 = sim_data["y_init_M1"]
    y_init_M2 = sim_data["y_init_M2"]

    fig_3d = plt.figure(figsize=(5, 7.5))
    ax3 = fig_3d.add_subplot(111, projection="3d")

    # 2D SIM surface M_1: y^3 = y^1 * y^2 (expanded to cover full trajectory domain)
    y1_surf = np.linspace(0.0, 1.15, 42)
    y2_surf = np.linspace(0.0, 0.82, 42)
    Y1, Y2 = np.meshgrid(y1_surf, y2_surf)
    Y3_M1 = Y1 * Y2
    ax3.plot_surface(Y1, Y2, Y3_M1, cmap="Blues", alpha=0.30, edgecolor="none",label=r"2D SIM Curve $\mathcal{M}_1$ ($M=1: f^1=0$)")
    ax3.plot_wireframe(Y1, Y2, Y3_M1, color="#1f4e78", alpha=0.18, rstride=6, cstride=6, lw=0.4)

    # 1D SIM curve M_2: y^2 = y^1 / (1 + y^1), y^3 = y^1 * y^2
    y1_curve = np.linspace(0.0, 1.15, 200)
    y2_curve = y1_curve / (1.0 + y1_curve)
    y3_curve = y1_curve * y2_curve
    ax3.plot(y1_curve, y2_curve, y3_curve, "-", color="#2ca02c", lw=2.0,
             label=r"1D SIM Curve $\mathcal{M}_2$ ($M=2$: $f^1, f^2 = 0$)")

    # Full 3D ODE trajectory y(t)
    ax3.plot(sol_full.y[0], sol_full.y[1], sol_full.y[2], "-", color="#333333", lw=3,
             label=r"Full 3D ODE Trajectory $\mathbf{y}(t)$")

    # Formulation B Fiber Relaxation homotopy y_sigma
    ax3.plot(sol_fib.y[0], sol_fib.y[1], sol_fib.y[2], ":", color="#d62728", lw=1.5,
             label=r"Stage 1 Fiber Relaxation $\mathbf{y}_\sigma$ ($\sigma \in [0,\infty)$)")

    # Stage 2 CSP Scale-Space trajectory at sigma_{c,1}(t) (2D SIM reduced model, M=1)
    ax3.plot(sol_ss_M1.y[0], sol_ss_M1.y[1], sol_ss_M1.y[2], "--", color="mediumpurple", lw=1.5,
             label=r"Stage 2 Scale-Space Flow at $\sigma_{c,1}(t)$ (2D SIM, $M=1$)")

    # Stage 2 CSP Scale-Space trajectory at sigma_{c,2}(t) (1D SIM reduced model, M=2)
    ax3.plot(sol_ss_M2.y[0], sol_ss_M2.y[1], sol_ss_M2.y[2], "-.", color="#ff7f0e", lw=1.5,
             label=r"Stage 2 Scale-Space Flow at $\sigma_{c,2}(t)$ (1D SIM, $M=2$)")

    # Key points
    ax3.scatter(*y0_3d, color="#d62728", s=55, zorder=10,
                label=rf"Off-Manifold $\mathbf{{y}}_0 = ({y0_3d[0]:.2f}, {y0_3d[1]:.2f}, {y0_3d[2]:.2f})^T$")
    ax3.scatter(*y_init_M1, color="mediumpurple", s=55, marker="^", zorder=10,
                label=r"2D SIM State $\mathbf{y}_{\sigma_{c,1}} \in \mathcal{M}_1$")
    ax3.scatter(*y_init_M2, color="#ff7f0e", s=55, marker="D", zorder=10,
                label=r"1D SIM State $\mathbf{y}_{\sigma_{c,2}} \in \mathcal{M}_2$")
    y_eq = [0.0, 0.0, 0.0]
    ax3.scatter(*y_eq, color="#000000", s=65, marker="s", zorder=10,
                label=r"Equilibrium $\mathbf{y}^* = (0,0,0)^T$")


    # 7. Key manifold plateau points (scatter styling matching Fig 3B)
    # ax3.scatter(*y0_3d, color="#2b2b2b", s=50, zorder=10)
    ax3.text(
        y0_3d[0] - 0.05, y0_3d[1], y0_3d[2] - 0.1,
        r"$\mathbf{y}_0 ~(\sigma=0)$",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f2f2f2", edgecolor="none", alpha=1.0),
    )

    # ax3.scatter(*y_init_M1, color="#d62728", s=55, marker="^", zorder=10)
    ax3.text(
        y_init_M1[0] - 0.35, y_init_M1[1] + 0.25, y_init_M1[2] - 0.2,
        r"2D SIM Plateau $\mathcal{M}_1$" + "\n" + r"$(\tau_1 \ll \sigma \ll \tau_2, M=1)$",
        color="#1f4e78",
        fontsize=8.2,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f2f2f2", edgecolor="none", alpha=1.0),
    )

    # ax3.scatter(*y_init_M2, color="#2ca02c", s=60, marker="D", zorder=10)
    ax3.text(
        y_init_M2[0] - 0.48, y_init_M2[1] + 0.0, y_init_M2[2] - 0.5,
        r"1D SIM Plateau $\mathcal{M}_2$" + "\n" + r"$(\tau_2 \ll \sigma \ll \tau_3, M=2)$",
        color="#2ba22a",
        fontsize=8.2,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="none", alpha=1.0),
    )

    # ax3.scatter(*y_eq, color="#000000", s=50, marker="s", zorder=10)
    ax3.text(
        0.0, 0.02, 0.06,
        r"$\mathbf{y}^* ~(\sigma \gg \tau_3)$",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="none", alpha=1.0),
    )


    ax3.set_xlabel(r"$y^1$ (Slow Driver)", labelpad=4, fontsize=10)
    ax3.set_ylabel(r"$y^2$ (Intermediate)", labelpad=4, fontsize=10)
    ax3.set_zlabel(r"$y^3$ (Ultra-Fast)", labelpad=2, fontsize=10)
    ax3.set_xlim(0.0, 1.18)
    ax3.set_ylim(0.0, 0.82)
    ax3.set_zlim(0.0, 0.7)
    ax3.set_box_aspect((1.0, 1.0, 0.6))
    ax3.view_init(elev=30, azim=-60)

    ax3.legend(loc="upper left", bbox_to_anchor=(0.01, 0.9), ncol=1, frameon=True, framealpha=0.92, fontsize=8.5)
    fig_3d.subplots_adjust(left=0.0, right=1.15, bottom=0.05, top=0.95)

    save(fig_3d, save_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig_3d)
