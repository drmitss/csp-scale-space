"""Figures 1-6: step-by-step theory on the Davis-Skodje system (Sections 3-5).

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
# PART 1: DAVIS-SKODJE INTRODUCTORY FIGURES (Section 3.4)
# =============================================================================
def make_fig_ds_intro_time():
    eps = 0.08
    y0 = np.array([1.5, 0.2])

    t_span = (0.0, 4.0)
    t_eval = np.linspace(0.0, 4.0, 800)
    sol = solve_ivp(ds_rhs, t_span, y0, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)
    y1_t, y2_t = sol.y
    y2_sim_t = y1_t / (1.0 + y1_t)

    # --- Figure 1a: Solution in Time ---
    fig1, ax1 = plt.subplots(figsize=(6.0, 4.5))
    t_bl = 3.0 * eps
    ax1.axvspan(0.0, t_bl, color="#ff7f0e", alpha=0.14, label=r"Fast Boundary Layer ($t \lesssim 3\tau_1$)")
    ax1.axvline(t_bl, color="#d95f02", ls=":", lw=1.3)

    ax1.plot(t_eval, y1_t, "-", color="#1f4e78", lw=2.5, label=r"Slow Variable $y^1(t)$")
    ax1.plot(t_eval, y2_t, "-", color="#d62728", lw=2.5, label=r"Fast Variable $y^2(t)$")
    ax1.plot(t_eval, y2_sim_t, "--", color="#2ca02c", lw=2.0,
             label=r"Exact SIM $\gamma_{\mathrm{SIM}}(y^1(t)) = \frac{y^1}{1+y^1}$")

    ax1.plot(0.0, y0[0], "o", color="#1f4e78", ms=6, zorder=5)
    ax1.plot(0.0, y0[1], "o", color="#d62728", ms=6, zorder=5)

    ax1.annotate(r"Rapid Fast-Fiber Relaxation" + "\n" + r"$($$\tau_1 = \epsilon = 0.08$$)$",
                 xy=(0.10, 0.44), xytext=(0.3, 0.04),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3),
                 fontsize=10, color="#b2182b", fontweight="bold")
    ax1.annotate(r"Slow Evolution on $\mathcal{M}_{\mathrm{SIM}}$" + "\n" + r"$($$\tau_2 = 1$$)$",
                 xy=(1.4, y1_t[280] / (1.0 + y1_t[280])), xytext=(1.75, 0.72),
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.3),
                 fontsize=10, color="#1b7837", fontweight="bold")

    # ax1.set_title(r"2D Davis--Skodje System: Temporal Evolution ($\epsilon = 0.08$)", pad=8)
    ax1.set_xlabel(r"Physical Time $t$", fontsize=10)
    ax1.set_ylabel(r"State Variables $y^1(t),\, y^2(t)$", fontsize=10)
    ax1.set_xlim(0.0, 4.0)
    ax1.set_ylim(0.0, 1.6)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.tick_params(labelsize=9)

    plt.tight_layout()
    save(fig1, "fig_ds_intro_time.pdf")
    plt.close(fig1)


def make_ds_phase_space():
    eps = 0.08
    y0 = np.array([1.5, 0.2])

    t_span = (0.0, 4.0)
    t_eval = np.linspace(0.0, 4.0, 800)
    sol = solve_ivp(ds_rhs, t_span, y0, t_eval=t_eval, method="Radau", rtol=1e-11, atol=1e-13)
    y1_t, y2_t = sol.y
    y2_sim_t = y1_t / (1.0 + y1_t)

    # --- Figure 1b: Phase Space Portrait with Highlighted SIM ---
    fig2, ax2 = plt.subplots(figsize=(6.0, 4.5))
    y1_grid = np.linspace(0.0, 2.25, 400)
    y2_sim_grid = y1_grid / (1.0 + y1_grid)

    ax2.fill_between(y1_grid, y2_sim_grid - 0.038, y2_sim_grid + 0.038,
                     color="#2ca02c", alpha=0.18, label=r"Attracting SIM Neighborhood ($\mathcal{O}(\epsilon)$)")
    ax2.plot(y1_grid, y2_sim_grid, "-", color="#1b7837", lw=3.2, zorder=4,
             label=r"Exact Slow Invariant Manifold $\mathcal{M}_{\mathrm{SIM}}$")

    for y1_fib in [0.4, 0.8, 1.2, 1.6, 2.0]:
        ax2.axvline(y1_fib, color="#888888", ls="--", lw=0.9, alpha=0.45)
    ax2.plot([], [], "--", color="#888888", lw=1.0, label=r"Fast Fibers $\mathbf{a}_1^{(0)} = [0, 1]^T$")

    ic_list = [
        (np.array([1.5, 0.20]), "#d62728", "-", 2.5, r"Reference Trajectory $\mathbf{y}_0 = (1.5, 0.2)^T$"),
        (np.array([2.05, 0.25]), "#1f4e78", "-.", 1.8, r"Off-Manifold Trajectories"),
        (np.array([1.85, 0.88]), "#1f4e78", "-.", 1.8, None),
        (np.array([1.05, 0.82]), "#1f4e78", "-.", 1.8, None),
        (np.array([0.75, 0.12]), "#1f4e78", "-.", 1.8, None),
    ]
    for y_init, col, ls_style, lw_val, lbl in ic_list:
        s_tr = solve_ivp(ds_rhs, (0.0, 5.0), y_init, t_eval=np.linspace(0.0, 5.0, 600),
                         method="Radau", rtol=1e-11, atol=1e-13)
        ax2.plot(s_tr.y[0], s_tr.y[1], ls=ls_style, color=col, lw=lw_val, label=lbl,
                 zorder=5 if lbl and "Reference" in lbl else 3)
        ax2.plot(y_init[0], y_init[1], "o", color=col, ms=5.5, zorder=6)
        ax2.annotate("", xy=(s_tr.y[0, 8], s_tr.y[1, 8]),
                     xytext=(s_tr.y[0, 2], s_tr.y[1, 2]),
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

    ax2.annotate("", xy=(0.35, 0.35 / 1.35), xytext=(0.60, 0.60 / 1.60),
                 arrowprops=dict(arrowstyle="->", color="#1b7837", lw=2.2))
    ax2.plot(0.0, 0.0, "s", color="#000000", ms=6.5, zorder=7, label=r"Equilibrium $\mathbf{y}^* = (0,0)^T$")

    ax2.text(1.48, 0.13, r"$\mathbf{y}_0 = (1.5, 0.2)^T$", color="#d62728", fontsize=11, fontweight="bold")
    ax2.text(0.05, 0.47, r"$\mathcal{M}_{\mathrm{SIM}}:\, y^2 = \frac{y^1}{1+y^1}$",
             color="#1b7837", fontsize=14, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#2ca02c", alpha=0.85))

    # ax2.set_title(r"2D Davis--Skodje System: Phase Space & Highlighted SIM", pad=8)
    ax2.set_xlabel(r"Slow Coordinate $y^1$", fontsize=10)
    ax2.set_ylabel(r"Fast Coordinate $y^2$", fontsize=10)
    ax2.set_xlim(0.0, 2.25)
    ax2.set_ylim(0.0, 1.2)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.5, fontsize=10)
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.tick_params(labelsize=9)

    plt.tight_layout()
    save(fig2, "fig_ds_intro_phase.pdf")
    plt.close(fig2)
    # print("Saved fig_ds_intro_time.pdf and fig_ds_intro_phase.pdf")


# =============================================================================
def make_fig2b_filter_weights():
    fig, ax2 = plt.subplots(figsize=(6.6, 5.2))

    sigma_log = np.logspace(-3.2, 1.2, 600)
    tau1, tau2 = 0.01, 1.0
    sigma_c = np.sqrt(tau1 * tau2)  # 0.1

    # Smooth Gaussian scale-space weights (p = 2)
    w1_smooth = np.exp(-((sigma_log / tau1) ** 2))
    w2_smooth = np.exp(-((sigma_log / tau2) ** 2))

    # Sharp binary Heaviside cutoff (p -> inf): Theta(tau_1 - sigma)
    w1_disc = np.where(sigma_log < tau1, 1.0, 0.0)

    # Tightened asymptotic SIM scaling window tau_1 << sigma << tau_2
    # Across [0.03, 0.25], w1_smooth <= 1.2e-4 (~0) and w2_smooth >= 0.94 (~1)
    win_left, win_right = 0.03, 0.25
    ax2.axvspan(
        win_left,
        win_right,
        color="#2ca02c",
        alpha=0.13,
        label=r"Asymptotic SIM Window ($\tau_1 \ll \sigma \ll \tau_2$, $M=1$)",
    )

    # Plot curves
    ax2.semilogx(
        sigma_log,
        w1_disc,
        ":",
        color="#555555",
        lw=3.8,
        label=r"Sharp Binary Cutoff $w_1^{\mathrm{disc}}(\sigma) = \Theta(\tau_1 - \sigma)$ ($p \to \infty$)",
    )
    ax2.semilogx(
        sigma_log,
        w1_smooth,
        "-",
        color="#d62728",
        lw=2.6,
        label=r"Mode 1 (Fast) Weight $w_1(\sigma) = e^{-(\sigma/\tau_1)^2}$",
    )
    ax2.semilogx(
        sigma_log,
        w2_smooth,
        "-",
        color="#1f4e78",
        lw=2.6,
        label=r"Mode 2 (Slow) Weight $w_2(\sigma) = e^{-(\sigma/\tau_2)^2}$",
    )

    # Vertical reference lines
    ax2.axvline(tau1, color="#d62728", ls="--", lw=1.2, alpha=0.75)
    ax2.axvline(sigma_c, color="#2ca02c", ls="-.", lw=1.5, alpha=0.9)
    ax2.axvline(tau2, color="#1f4e78", ls="--", lw=1.2, alpha=0.75)

    # Mark e^(-1) intersection at sigma = tau_1 and sigma = tau_2
    e_inv = np.exp(-1.0)
    ax2.plot([tau1, tau2], [e_inv, e_inv], "o", color="#333333", ms=4.5, zorder=5)

    # Timescale & SIM window labels
    ax2.text(
        tau1 * 1.12,
        0.62,
        r"$\tau_1 = \epsilon$",
        color="#d62728",
        fontsize=12,
        fontweight="bold",
    )
    ax2.text(
        sigma_c * 1.08,
        0.51,
        r"$\sigma_c = \sqrt{\tau_1\tau_2}$" + "\n" + r"$w_1 \approx 0$" + "\n" + r"$w_2 \approx 1$",
        color="#2ca02c",
        fontsize=12,
        fontweight="bold",
    )
    ax2.text(
        tau2 * 1.15,
        0.62,
        r"$\tau_2 = 1$",
        color="#1f4e78",
        fontsize=10,
        fontweight="bold",
    )

    # Top regime annotations (M=0, M=1, M=2)
    ax2.text(
        0.0025,
        1.07,
        r"Fine Scale ($M=0$)" + "\n" + r"$w_1, w_2 \approx 1$",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )
    ax2.text(
        sigma_c,
        1.07,
        r"SIM Regime $($$M=1$$)$" + "\n" + r"$w_1 \approx 0, w_2 \approx 1$",
        ha="center",
        va="center",
        fontsize=10,
        color="#1b7837",
        fontweight="bold",
        # ,bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.9))
    )
    ax2.text(
        4.2,
        1.07,
        r"Equilibrium ($M=2$)" + "\n" + r"$w_1, w_2 \to 0$",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )

    # Discontinuous binary switch callout
    ax2.annotate(
        r"Discontinuous Binary Switch" + "\n" + r"at $\sigma = \tau_1$ (Non-differentiable)",
        xy=(tau1, 0.88),
        xytext=(tau1 * 2.8, 0.82),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.3),
        fontsize=11,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.9))
    # )

    # ax2.set_title(r"Continuous Scale-Space Homotopy vs. Sharp Binary CSP Cutoff", pad=10)
    ax2.set_xlabel(r"Observation Scale $\sigma$ (log scale)")
    ax2.set_ylabel(r"Modal Survival Weight $w_i(\mathbf{y}; \sigma)$")
    ax2.set_xlim(10 ** (-3.2), 10 ** (1.2))
    ax2.set_ylim(-0.05, 1.16)

    ax2.legend(loc="lower left", bbox_to_anchor=(0.28, 0.04), frameon=True, fontsize=10.2)
    ax2.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    save(fig, "fig_geom2_filter_weights.pdf")
    plt.close(fig)


# =============================================================================
# FIGURE 1a: Naive Isotropic Euclidean Scale-Space
# =============================================================================
def make_fig1a_isotropic_euclidean():
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    # Underlying curved Slow Invariant Manifold: y2 = y1 / (1 + y1)
    y1_grid = np.linspace(0.1, 2.3, 300)
    y2_sim = y1_grid / (1.0 + y1_grid)
    ax.plot(y1_grid, y2_sim, color="#1f4e78", lw=2.8, label=r"Curved Slow Manifold $\mathcal{M}_{\mathrm{SIM}}$")

    # Fast fiber slope (oblique fast fibers a_1 = [-0.25, 1.0]^T)
    fiber_ratio = -0.25
    for y1_base in [0.4, 0.9, 1.4, 1.9]:
        y2_base = y1_base / (1.0 + y1_base)
        dy2 = np.linspace(-0.28, 0.28, 50)
        ax.plot(y1_base + fiber_ratio * dy2, y2_base + dy2, "--", color="#b0b0b0", lw=0.9)

    # Exact multiscale trajectory relaxing onto M_SIM
    t_vals = np.linspace(0, 1.6, 400)
    eps = 0.06
    y1_0, y2_0 = 2.05, 0.25
    y1_traj = y1_0 * np.exp(-t_vals)
    y2_traj = (y1_traj / (1.0 + y1_traj)) + (y2_0 - y1_0 / (1.0 + y1_0)) * np.exp(-t_vals / eps)
    ax.plot(y1_traj, y2_traj, color="#2b2b2b", lw=1.8, label=r"Exact Trajectory $\mathbf{y}(t)$")

    # Naive Isotropic Gaussian Convolved Trajectory (pulls inside concave curve)
    y1_smooth = np.linspace(0.25, 1.85, 200)
    y2_iso_drift = (y1_smooth / (1.0 + y1_smooth)) - 0.11 / ((1.0 + y1_smooth) ** 1.2)
    ax.plot(y1_smooth, y2_iso_drift, color="#d95f02", lw=2.2, ls="-.",
            label=r"Isotropically Smoothed $K(\cdot;\zeta)*\mathbf{y}$ (Off-Manifold Drift)")

    # Draw Isotropic Gaussian Diffusion Circles at a manifold point p
    p_y1 = 0.85
    p_y2 = p_y1 / (1.0 + p_y1)
    ax.scatter([p_y1], [p_y2], color="#1f4e78", s=45, zorder=6)
    for r, alpha_val in [(0.09, 0.28), (0.18, 0.16), (0.26, 0.08)]:
        circ = Ellipse((p_y1, p_y2), width=2*r, height=2*r*0.55, angle=0,
                       edgecolor="#d95f02", facecolor="#d95f02", alpha=alpha_val, lw=1.2)
        ax.add_patch(circ)

    # Isotropic arrows in all directions
    for ang in [45, 135, 225, 315]:
        rad = np.deg2rad(ang)
        dx, dy = 0.16 * np.cos(rad), 0.09 * np.sin(rad)
        ax.annotate("", xy=(p_y1 + dx, p_y2 + dy), xytext=(p_y1, p_y2),
                    arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.4))

    # Highlight Jensen's inequality drift arrow pulling state off M_SIM
    p_drift_y2 = p_y2 - 0.11 / ((1.0 + p_y1) ** 1.2)
    ax.scatter([p_y1], [p_drift_y2], color="#d95f02", s=45, zorder=6)
    ax.annotate("", xy=(p_y1, p_drift_y2 + 0.01), xytext=(p_y1, p_y2 - 0.01),
                arrowprops=dict(arrowstyle="->", color="#c00000", lw=2.2))
    ax.text(p_y1 + 0.1, (p_y2 + p_drift_y2)/2 - 0.13,
            r"Transverse Distortion" + "\n" + r"$(\mathbf{y}_{\zeta} \notin \mathcal{M}_{\mathrm{SIM}}, ~f^1 \neq 0)$",
            color="#c00000", fontsize=10, fontweight="bold")

    ax.text(p_y1 - 0.35, p_y2 + 0.15, r"Isotropic $\dfrac{1}{2}\nabla_{\mathbf{y}}^2$" + "\n$($Smooths Normal & Tangent$)$",
            color="#d95f02", fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d95f02", alpha=0.9))

    # ax.set_title(r"Naive Isotropic Euclidean Scale-Space ($\frac{1}{2}\nabla_{\mathbf{y}}^2$)", pad=8)
    ax.set_xlabel(r"Slow Coordinate $y^1$",fontsize=12)
    ax.set_ylabel(r"Fast Coordinate $y^2$",fontsize=12)
    ax.set_xlim(0.05, 2.25)
    ax.set_ylim(0.05, 0.92)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.tick_params(labelsize=9)

    ax.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    save(fig, "fig_geom1_isotropic.pdf")
    plt.close(fig)


# =============================================================================
# FIGURE 2: Anisotropic CSP Spectral Scale-Space
# =============================================================================
def make_fig1b_anisotropic_csp():
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    # Underlying curved Slow Invariant Manifold: y2 = y1 / (1 + y1)
    y1_grid = np.linspace(0.1, 2.3, 300)
    y2_sim = y1_grid / (1.0 + y1_grid)
    ax.plot(y1_grid, y2_sim, color="#1f4e78", lw=2.8, label=r"Curved Slow Manifold $\mathcal{M}_{\mathrm{SIM}}$ ($f^1 = 0$)")

    # Fast fiber slope (oblique fast fibers a_1 = [-0.25, 1.0]^T)
    fiber_ratio = -0.25
    for y1_base in [0.45, 0.95, 1.45, 1.90]:
        y2_base = y1_base / (1.0 + y1_base)
        dy2 = np.linspace(-0.28, 0.28, 50)
        ax.plot(y1_base + fiber_ratio * dy2, y2_base + dy2, "--", color="#d62728", lw=1.0, alpha=0.55)
        ax.annotate("", xy=(y1_base + fiber_ratio * 0.04, y2_base + 0.04),
                    xytext=(y1_base + fiber_ratio * 0.24, y2_base + 0.24),
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6))
        ax.annotate("", xy=(y1_base - fiber_ratio * 0.04, y2_base - 0.04),
                    xytext=(y1_base - fiber_ratio * 0.22, y2_base - 0.22),
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6))

    # Draw Anisotropic Spectral Diffusion Ellipses aligned with fast fiber a_1 at point q
    q_y1 = 0.95
    q_y2 = q_y1 / (1.0 + q_y1)
    ax.scatter([q_y1], [q_y2], color="#1f4e78", s=50, zorder=6)

    fiber_angle_deg = np.rad2deg(np.arctan2(1.0, fiber_ratio))
    for length, alpha_val in [(0.34, 0.25), (0.22, 0.40)]:
        ell = Ellipse((q_y1, q_y2), width=length, height=0.028, angle=fiber_angle_deg,
                      edgecolor="#d62728", facecolor="#d62728", alpha=alpha_val, lw=1.2)
        ax.add_patch(ell)

    # Show local CSP basis vectors a_1 (fast fiber) and a_2 (slow tangent) at q
    slope_sim = 1.0 / ((1.0 + q_y1) ** 2)
    a2_vec = np.array([-0.42, -0.42 * slope_sim])
    a1_vec = np.array([fiber_ratio * 0.22, 0.22])

    ax.annotate("", xy=(q_y1 + a1_vec[0], q_y2 + a1_vec[1]), xytext=(q_y1, q_y2),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.4))
    ax.text(q_y1 + a1_vec[0] - 0.46, q_y2 + a1_vec[1] + 0.05,
            r"Fast Basis $\mathbf{a}_1(\mathbf{y})$" + "\n" + r"$($Attenuated: $d_1(\sigma) \gg 1)$",
            color="#d62728", fontsize=10, fontweight="bold"
            ,bbox=dict(boxstyle="round,pad=0.0", facecolor="white", edgecolor="none", alpha=0.9))

    ax.annotate("", xy=(q_y1 + a2_vec[0], q_y2 + a2_vec[1]), xytext=(q_y1, q_y2),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.6))
    ax.text(q_y1 + a2_vec[0] + 0.0, q_y2 + a2_vec[1] - 0.11,
            r"Slow Basis $\mathbf{a}_2(\mathbf{y}) \in T_{\mathbf{y}}\mathcal{M}_{\mathrm{SIM}}$" + "\n" + r"$($Preserved: $w_2(\sigma) \approx 1)$",
            color="#2ca02c", fontsize=10, fontweight="bold"
           ,bbox=dict(boxstyle="round,pad=0.0", facecolor="white", edgecolor="none", alpha=0.9))

    # Off-manifold point y_0 relaxing along fiber onto M_SIM
    y0_pt = np.array([1.45 + fiber_ratio * 0.25, (1.45 / 2.45) + 0.25])
    y_sim_pt = np.array([1.45, 1.45 / 2.45])
    ax.scatter([y0_pt[0]], [y0_pt[1]], color="#2b2b2b", s=45, zorder=6, label=r"Off-Manifold State $\mathbf{y}_0$")
    ax.scatter([y_sim_pt[0]], [y_sim_pt[1]], color="#2ca02c", s=50, zorder=6,
               label=r"Relaxed State $\mathbf{y}_{\sigma} \in \mathcal{M}_{\mathrm{SIM}}$")
    ax.text(y0_pt[0] + 0.04, y0_pt[1] - 0.01, r"$\mathbf{y}_0$ $($$\sigma=0$$)$", fontsize=12, fontweight="bold")
    ax.text(y_sim_pt[0] + 0.04, y_sim_pt[1] - 0.04, r"$\mathbf{y}_\sigma$ $($$\tau_1 \ll \sigma \ll \tau_2$$)$",
            color="#1f4e78", fontsize=12, fontweight="bold"
           ,bbox=dict(boxstyle="round,pad=0.0", facecolor="white", edgecolor="none", alpha=0.9))

    # ax.set_title(r"Proposed Anisotropic CSP Scale-Space ($\mathcal{D}_\sigma = \mathbf{A}\mathbf{D}(\sigma)\mathbf{B}$)", pad=8)
    ax.set_xlabel(r"Slow Coordinate $y^1$", fontsize=12)
    ax.set_ylabel(r"Fast Coordinate $y^2$", fontsize=12)
    ax.set_xlim(0.05, 2.25)
    ax.set_ylim(0.05, 0.92)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, ls=":", alpha=0.4)


    plt.tight_layout()
    save(fig, "fig_geom1_anisotropic.pdf")
    plt.close(fig)


# =============================================================================
# FIGURE 2: Geometric Duality (Velocity Filtration vs. Fiber Relaxation)
# =============================================================================
def make_fig2a_geometric_duality():
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    y1_grid = np.linspace(0.2, 2.1, 300)
    y2_sim = y1_grid / (1.0 + y1_grid)
    ax.plot(y1_grid, y2_sim, color="#1f4e78", lw=3.0, label=r"Slow Invariant Manifold $\mathcal{M}_{\sigma_c}$ ($f^1 = 0$)")

    # Off-manifold state y_0 = (1.55, 0.26)
    y0 = np.array([1.55, 0.26])
    y2_on_sim = y0[0] / (1.0 + y0[0])
    y_sigma_c = np.array([1.43, 1.43 / 2.43])

    # 1. Formulation B continuous state-fiber trajectory y_sigma
    s_param = np.linspace(0, 1, 100)
    y1_formB = y0[0] + (y_sigma_c[0] - y0[0]) * (s_param ** 1.5)
    y2_formB = y0[1] + (y_sigma_c[1] - y0[1]) * (1.0 - np.exp(-4.5 * s_param)) / (1.0 - np.exp(-4.5))
    ax.plot(y1_formB, y2_formB, color="#d62728", lw=2.6, ls="-.",
            label=r"Formulation B ($\partial_\sigma \mathbf{y}_\sigma = -\sum \mathbf{a}_i \lambda_i^{-1} d_i f^i$)")
    ax.annotate("", xy=(y1_formB[55], y2_formB[55]), xytext=(y1_formB[40], y2_formB[40]),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.2))

    # Intermediate homotopy states y_sigma along the fiber
    for idx, lbl in [(22, r"$\mathbf{y}_{\sigma_1}$"), (50, r"$\mathbf{y}_{\sigma_2}$")]:
        ax.scatter([y1_formB[idx]], [y2_formB[idx]], color="#d62728", s=30, zorder=6)
        ax.text(y1_formB[idx] - 0.08, y2_formB[idx] - 0.01, lbl, color="#d62728", fontsize=12)

    # 2. Classical Discrete CSP Radical Correction
    ax.plot([y0[0], y0[0]], [y0[1], y2_on_sim], color="#7f7f7f", lw=1.6, ls=":",
            label=r"Classical Discrete CSP Jump ($\Delta\mathbf{y} = -\mathbf{a}_1 u_1$)")
    ax.annotate("", xy=(y0[0], y2_on_sim - 0.01), xytext=(y0[0], y0[1] + 0.02),
                arrowprops=dict(arrowstyle="->", color="#7f7f7f", lw=1.5, ls=":"))
    ax.text(y0[0] + 0.025, 0.48, r"Modal Displacement" + "\n" + r"$u_1 = \lambda_1^{-1}f^1$",
            color="#555555", fontsize=11)

    # 3. Tangent Space Velocity Vectors at y_0 (Formulation A)
    v_slow = np.array([-0.38, -0.38 / ((1.0 + y0[0]) ** 2)])
    v_fast = np.array([0.0, 0.26])
    v_full = v_slow + v_fast
    v_mid  = v_slow + 0.45 * v_fast

    # Parallelogram guides
    ax.plot([y0[0] + v_slow[0], y0[0] + v_full[0]], [y0[1] + v_slow[1], y0[1] + v_full[1]],
            ":", color="#888888", lw=1.0)
    ax.plot([y0[0] + v_fast[0], y0[0] + v_full[0]], [y0[1] + v_fast[1], y0[1] + v_full[1]],
            ":", color="#888888", lw=1.0)

    # Velocity vectors
    ax.annotate("", xy=y0 + v_full, xytext=y0,
                arrowprops=dict(arrowstyle="->", color="#2b2b2b", lw=2.2))
    ax.text(y0[0] + v_full[0] - 0.16, y0[1] + v_full[1] + 0.02,
            r"$\mathbf{g}(\mathbf{y}_0)~(\sigma=0)$", color="#2b2b2b", fontsize=12, fontweight="bold")

    ax.annotate("", xy=y0 + v_mid, xytext=y0,
                arrowprops=dict(arrowstyle="->", color="#9467bd", lw=2.0))
    ax.text(y0[0] + v_mid[0] - 0.34, y0[1] + v_mid[1] + 0.01,
            r"$\mathbf{g}_\sigma(\mathbf{y}_0)$ ($w_1 \approx 0.45$)", color="#9467bd", fontsize=12)

    ax.annotate("", xy=y0 + v_slow, xytext=y0,
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=2.3))
    ax.text(y0[0] + v_slow[0] - 0.32, y0[1] + v_slow[1] - 0.00,
            r"$\mathbf{g}_{\sigma_c}(\mathbf{y}_0) = \mathbf{a}_2 f^2$", color="#d95f02", fontsize=12, fontweight="bold")

    # Curved arc showing rotation g(y_0) -> g_{sigma_c}(y_0)
    ax.annotate("", xy=y0 + v_slow + np.array([0.04, 0.0]),
                xytext=y0 + v_full + np.array([0.2, -0.38]),
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.6,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(y0[0] - 0.16, y0[1] - 0.2,
            r"Formulation A ($\mathcal{S}_\sigma\mathbf{g}$)" + "\n" + r"Damps $f^1$ at fixed $\mathbf{y}_0$",
            color="#d95f02", fontsize=11, bbox=dict(boxstyle="round,pad=0.2", fc="#fff8f0", ec="#ff7f0e"))

    # 4. Unrelaxed Formulation A flow
    y1_unrel = np.linspace(0.45, y0[0], 150)
    offset = y0[1] - (y0[0] / (1.0 + y0[0]))
    y2_unrel = (y1_unrel / (1.0 + y1_unrel)) + offset
    ax.plot(y1_unrel, y2_unrel, "--", color="#ff7f0e", lw=2.0,
            label=r"Unrelaxed Formulation A Flow (Frozen Offset $u_1 \neq 0$)")
    ax.annotate("", xy=(0.75, (0.75/1.75) + offset), xytext=(0.95, (0.95/1.95) + offset),
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.8))

    # 5. Unified Stage 2 flow on M_SIM
    y1_stage2 = np.linspace(0.35, y_sigma_c[0], 150)
    y2_stage2 = y1_stage2 / (1.0 + y1_stage2)
    ax.plot(y1_stage2, y2_stage2, color="#2ca02c", lw=2.8, ls="--",
            label=r"Unified Stage 2 Flow ($\dot{\mathbf{y}}_\sigma = \mathcal{S}_{\sigma_c}\mathbf{g}(\mathbf{y}_{\sigma_c})$)")
    ax.annotate("", xy=(0.75, 0.75/1.75), xytext=(1.05, 1.05/2.05),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.4))

    # Scatter key points
    ax.scatter([y0[0]], [y0[1]], color="#2b2b2b", s=55, zorder=7)
    ax.text(y0[0] + 0.03, y0[1] - 0.03, r"$\mathbf{y}_0$ $($Off-Manifold$)$", fontsize=11, fontweight="bold")
    ax.scatter([y_sigma_c[0]], [y_sigma_c[1]], color="#2ca02c", s=65, zorder=7)
    ax.text(y_sigma_c[0] - 0.13, y_sigma_c[1] + 0.047,
            r"$~\mathbf{y}_{\sigma_c} \in \mathcal{M}_{\sigma_c}$" + "\n$($After Stage 1$)$",
            color="#2ca02c", fontsize=11, fontweight="bold"
            # ,bbox=dict(boxstyle="round,pad=0.0", facecolor="white", edgecolor="none", alpha=0.9)
           )

    # ax.set_title(r"Geometric Duality: Velocity Filtration (A) vs. Fiber Relaxation (B)", pad=8)
    ax.set_xlabel(r"Slow State Variable $y^1$", fontsize=10)
    ax.set_ylabel(r"Fast State Variable $y^2$", fontsize=10)
    ax.set_xlim(0.2, 2.08)
    ax.set_ylim(0.02, 0.8)
    ax.legend(loc="upper left", frameon=True, fontsize=8.7)
    ax.grid(True, ls=":", alpha=0.4)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    save(fig, "fig_geom2_duality_phase_space.pdf")
    plt.close(fig)


# =============================================================================
# FIGURE 3A: Curvature Corrections via B- and A-Refinements
# =============================================================================
def make_fig3a_curvature_refinements():
    fig, ax1 = plt.subplots(figsize=(6.8, 5.2))

    # Curved Fenichel Slow Invariant Manifold M_eps
    x_sim = np.linspace(0.15, 2.0, 300)
    y_sim = 0.18 + 0.55 * (1.0 - np.exp(-1.2 * x_sim))
    # Zeroth-order unrefined manifold f^{1,(0)} = 0 (O(eps) offset)
    y_sim_0 = y_sim - 0.085 * np.exp(-0.6 * x_sim)

    ax1.plot(x_sim, y_sim_0, "--", color="#d95f0e", lw=1.8,
             label=r"Zeroth-Order Manifold " + "\n" + r"$f^{1,(0)}=0$ ($\mathcal{O}(\epsilon)$)")
    ax1.plot(x_sim, y_sim, color="#1f4e78", lw=3.0,
             label=r"True Fenichel SIM $\mathcal{M}_\epsilon$"+ "\n" +  r"$f^{1,(1)}=0$ ($\mathcal{O}(\epsilon^2)$)")

    # Base point p on M_eps
    px = 1.05
    py = 0.18 + 0.55 * (1.0 - np.exp(-1.2 * px))
    ax1.scatter([px], [py], color="#1f4e78", s=60, zorder=8)
    # ax1.text(px + 0.04, py - 0.02, r"Base Point $p \in \mathcal{M}_\epsilon$", fontsize=10, fontweight="bold")
    ax1.text(px + 0.04, py - 0.02, r"Base Point $p \in \mathcal{M}_\epsilon$", fontsize=10, fontweight="bold"
             ,bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.9))

    # Curved Fenichel Fast Stable Fiber F_eps^p passing through p
    s_fib = np.linspace(-0.28, 0.34, 200)
    fx = px - 0.35 * s_fib + 0.45 * (s_fib ** 2)
    fy = py + s_fib
    ax1.plot(fx, fy, color="#8c2d04", lw=2.2, ls="-.",
             label=r"Curved Fenichel Fast Fiber $\mathcal{F}_\epsilon^p$")

    # Tangent to M_eps at p (First-order B-refined slow basis vector a_2^(1))
    slope_sim_p = 0.55 * 1.2 * np.exp(-1.2 * px)
    a2_1 = np.array([-0.55, -0.55 * slope_sim_p])
    # Zeroth-order slow eigenvector a_2^(0) (misaligned by O(eps) angle)
    a2_0 = np.array([-0.55, -0.55 * (slope_sim_p + 0.28)])

    ax1.annotate("", xy=(px + a2_0[0], py + a2_0[1]), xytext=(px, py),
                 arrowprops=dict(arrowstyle="->", color="#d95f0e", lw=2.2, ls="--"))
    ax1.text(px + a2_0[0] - 0.15, py + a2_0[1] - 0.055,
             r"$\mathbf{a}_2^{(0)}$ $($Unrefined, $\mathcal{O}(\epsilon)$ Drift$)$", color="#d95f0e", fontsize=10, fontweight="bold")

    ax1.annotate("", xy=(px + a2_1[0], py + a2_1[1]), xytext=(px, py),
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.6))
    ax1.text(px + a2_1[0] - 0.25, py + a2_1[1] + 0.06,
             r"$\mathbf{a}_2^{(1)} \in T_p\mathcal{M}_\epsilon$ $($$\mathbf{B}$-Refined$)$",
             color="#2ca02c", fontsize=10, fontweight="bold")

    # Curved arc showing B-refinement rotating a_2^(0) -> a_2^(1)
    ax1.annotate("", xy=(px + 0.72*a2_1[0], py + 0.72*a2_1[1]),
                 xytext=(px + 0.72*a2_0[0], py + 0.72*a2_0[1]),
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.6,
                                 connectionstyle="arc3,rad=-0.25"))
    ax1.text(px - 0.37, py - 0.20, r"Step 1: $\mathbf{B}$-Refinement" + "\n" + r"($\mathbf{\Lambda}^f_s \to \mathcal{O}(\epsilon^2)$)",
             color="#2ca02c", fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="#f2fff2", ec="#2ca02c"))

    # Tangent to Fast Fiber F_eps^p at p (First-order A-refined fast basis vector a_1^(1))
    a1_1 = np.array([-0.35 * 0.28, 0.28])
    # Zeroth-order fast eigenvector a_1^(0) (vertical, unrefined)
    a1_0 = np.array([0.05, 0.28])

    ax1.annotate("", xy=(px + a1_0[0], py + a1_0[1]), xytext=(px, py),
                 arrowprops=dict(arrowstyle="->", color="#d95f0e", lw=2.2, ls="--"))
    ax1.text(px + a1_0[0] + 0.03, py + a1_0[1] - 0.02,
             r"$\mathbf{a}_1^{(0)}$ $($Unrefined$)$", color="#d95f0e", fontsize=10, fontweight="bold")

    ax1.annotate("", xy=(px + a1_1[0], py + a1_1[1]), xytext=(px, py),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.6))
    ax1.text(px + a1_1[0] - 0.46, py + a1_1[1] + 0.0,
             r"$\mathbf{a}_1^{(1)} \in T_p\mathcal{F}_\epsilon^p$ $($$\mathbf{A}$-Refined$)$",
             color="#d62728", fontsize=10, fontweight="bold")

    # Curved arc showing A-refinement rotating a_1^(0) -> a_1^(1)
    ax1.annotate("", xy=(px + 0.78*a1_1[0], py + 0.78*a1_1[1]),
                 xytext=(px + 0.78*a1_0[0], py + 0.78*a1_0[1]),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6,
                                 connectionstyle="arc3,rad=0.25"))
    ax1.text(px - 0.6, py + 0.18, r"Step 2: $\mathbf{A}$-Refinement" + "\n" + r"($\mathbf{\Lambda}^s_f \to \mathcal{O}(\epsilon^2)$)",
             color="#d62728", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="#fff2f2", ec="#d62728"))

    # ax1.set_title(r"Curvature Correction via First-Order $\mathbf{B}$- and $\mathbf{A}$-Refinements", pad=8)
    ax1.set_xlabel(r"State Variable $y^1$", fontsize=11)
    ax1.set_ylabel(r"State Variable $y^2$", fontsize=11)
    ax1.set_xlim(0.15, 1.95)
    ax1.set_ylim(0.18, 0.90)
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    ax1.tick_params(labelsize=9)

    ax1.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    save(fig, "fig_geom3_refinements_2d.pdf")
    plt.close(fig)


eps_ds = 0.08

def generate_figure_1a():
    y_init_ds = [1.5, 0.2]
    sigma_c_ds = np.sqrt(eps_ds)
    sig_span_ds = np.linspace(0.0, sigma_c_ds, 400)

    sol_fiber_ds = solve_ivp(
        ds_fiber_rhs, [0.0, sigma_c_ds], y_init_ds,
        t_eval=sig_span_ds, args=(eps_ds, True), rtol=1e-10, atol=1e-12
    )
    sol_phys_ds = solve_ivp(
        lambda t, y: g_ds(y, eps_ds), [0.0, 4.5], y_init_ds,
        t_eval=np.linspace(0, 4.5, 600), rtol=1e-10, atol=1e-12
    )

    def g_sigma_formA_ds(t, y, sigma=sigma_c_ds, eps=eps_ds):
        y1, y2 = y
        g1, g2 = g_ds(y, eps)
        alp = alpha1_ds(y1, eps)
        f1 = -alp * g1 + g2
        f2 = g1
        w1 = np.exp(-(sigma / eps)**2)
        w2 = np.exp(-(sigma / 1.0)**2)
        return [w2 * f2, w1 * f1 + w2 * alp * f2]

    sol_formA_unrelaxed = solve_ivp(
        g_sigma_formA_ds, [0.0, 4.5], y_init_ds,
        t_eval=np.linspace(0, 4.5, 500), rtol=1e-10, atol=1e-12
    )
    y_relaxed_init = sol_fiber_ds.y[:, -1]
    sol_two_stage = solve_ivp(
        g_sigma_formA_ds, [0.0, 4.5], y_relaxed_init,
        t_eval=np.linspace(0, 4.5, 500), rtol=1e-10, atol=1e-12
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))

    y1_grid = np.linspace(0.0, 1.7, 300)
    y2_exact_grid = y1_grid / (1.0 + y1_grid)
    y2_0_grid = y2_exact_grid + eps_ds * y1_grid * (alpha_ds(y1_grid, eps_ds) - 1.0 / ((1.0 + y1_grid)**2))
    y2_1_grid = y2_exact_grid + eps_ds * y1_grid * (alpha1_ds(y1_grid, eps_ds) - 1.0 / ((1.0 + y1_grid)**2))

    ax.plot(y1_grid, y2_exact_grid, 'k-', lw=2.2, label=r'Exact SIM $\mathcal{M}_{\mathrm{SIM}}: y^2 = y^1/(1+y^1)$')
    ax.plot(y1_grid, y2_0_grid, color='#1f77b4', ls='--', lw=2.5, label=r'Zeroth-order manifold $f^{1,(0)}=0$')
    ax.plot(y1_grid, y2_1_grid, color='#9467bd', ls=':', lw=3.5, label=r'First-order refined manifold $f^{1,(1)}=0$')
    ax.plot(sol_phys_ds.y[0], sol_phys_ds.y[1], color='#2ca02c', lw=1.8, label=r'Exact trajectory $\mathbf{y}(t)$')
    ax.plot(sol_fiber_ds.y[0], sol_fiber_ds.y[1], color='#d62728', lw=2.0, ls='-.', label=r'Stage 1: Fiber relaxation $\mathbf{y}_\sigma$ ($\sigma \in [0,\sigma_c]$)')
    ax.plot(sol_two_stage.y[0], sol_two_stage.y[1], color='#d62728', lw=1.6, ls='-', alpha=0.85, label=r'Stage 2: Tangent flow $\dot{\mathbf{y}}_\sigma = \mathcal{S}_{\sigma_c}^{(1)}\mathbf{g}(\mathbf{y}_\sigma)$')
    ax.plot(sol_formA_unrelaxed.y[0], sol_formA_unrelaxed.y[1], color='#ff7f0e', lw=1.6, ls='--', label=r'Unrelaxed Formulation A ($\dot{\mathbf{y}}=\mathcal{S}_{\sigma_c}^{(1)}\mathbf{g}(\mathbf{y})$)')
    ax.scatter([y_init_ds[0]], [y_init_ds[1]], color='k', s=45, zorder=6, label=r'Initial state $\mathbf{y}_0=(1.5, 0.2)$')
    ax.scatter([y_relaxed_init[0]], [y_relaxed_init[1]], color='#d62728', marker='s', s=40, zorder=6)

    ax.set_xlabel(r'Slow variable $y^1$')
    ax.set_ylabel(r'Fast variable $y^2$')
    # ax.set_title(r'Davis-Skodje Phase Space & Two-Stage Filtration ($\epsilon=0.08$)')
    ax.set_xlim(0.0, 1.68)
    ax.set_ylim(-0.0, 0.68)
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, fontsize=12)

    plt.tight_layout()
    save(plt.gcf(), 'fig1a_phase_space.pdf', dpi=400, bbox_inches='tight')
    plt.close(fig)


def generate_figure_1b():
    eps_array = np.logspace(-3.5, -0.8, 50)
    vel_err0_arr, vel_err1_arr = [], []
    state_err0_arr, state_err1_arr = [], []
    y1_ref = 1.0
    g_ref = np.array([-y1_ref, -y1_ref / ((1.0 + y1_ref)**2)])

    for ep in eps_array:
        a0_val = alpha_ds(y1_ref, ep)
        a1_val = alpha1_ds(y1_ref, ep)
        vel_err0_arr.append(np.linalg.norm(np.array([-y1_ref, -y1_ref * a0_val]) - g_ref))
        vel_err1_arr.append(np.linalg.norm(np.array([-y1_ref, -y1_ref * a1_val]) - g_ref))
        state_err0_arr.append(abs(ep * y1_ref * (a0_val - 1.0 / ((1.0 + y1_ref)**2))))
        state_err1_arr.append(abs(ep * y1_ref * (a1_val - 1.0 / ((1.0 + y1_ref)**2))))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.loglog(eps_array, vel_err0_arr, 'b-', lw=2.0, label=r'Zeroth-order velocity error $\|\mathbf{g}_{\sigma_c}^{(0)} - \mathbf{g}\|_2 \sim \mathcal{O}(\epsilon)$')
    ax.loglog(eps_array, vel_err1_arr, 'r-', lw=2.0, label=r'First-order velocity error $\|\mathbf{g}_{\sigma_c}^{(1)} - \mathbf{g}\|_2 \sim \mathcal{O}(\epsilon^2)$')
    ax.loglog(eps_array, state_err0_arr, 'b--', lw=1.6, label=r'Zeroth-order manifold error $|y^{2,(0)} - y^2_{\mathrm{SIM}}| \sim \mathcal{O}(\epsilon^2)$')
    ax.loglog(eps_array, state_err1_arr, 'r--', lw=1.6, label=r'First-order manifold error $|y^{2,(1)} - y^2_{\mathrm{SIM}}| \sim \mathcal{O}(\epsilon^3)$')

    ax.set_xlabel(r'Spectral gap parameter $\epsilon = \tau_1 / \tau_2$')
    ax.set_ylabel(r'Asymptotic Tangency and SIM Errors at $y^1=1$')
    # ax.set_title(r'Asymptotic Convergence of Zeroth- vs. First-Order Basis')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, fontsize=11)

    plt.tight_layout()
    save(plt.gcf(), 'fig1b_asymptotic_convergence.pdf', dpi=400, bbox_inches='tight')
    plt.close(fig)
