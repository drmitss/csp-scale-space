"""Example 3 - the building blocks, one formula of the paper at a time.

Run:  python examples/03_low_level_operators.py
"""
import numpy as np

import csp_scalespace as css
from csp_scalespace.models import three_scale

np.set_printoptions(precision=4, suppress=True)
model = three_scale()                 # any ODEModel works here
y = model.y0

# CSP basis (Eq. 26): eigenvalues, timescales, basis vectors a_i (columns of A), duals b^i (rows of B)
basis = css.csp_basis(model, y)
print("lambda_i :", basis.lam)
print("tau_i    :", basis.tau)
print("f^i = b^i.g (Eq. 12):", basis.amplitudes(model.g(y)))

# Cutoff scales in the centre of the two spectral gaps (Thm 4.2)
s1, s2 = css.cutoff_scale(basis.tau, 1), css.cutoff_scale(basis.tau, 2)
print("sigma_c1, sigma_c2:", s1, s2)

# Scale-space operator S_sigma = A W B (Eq. 43) and its classical limit (Eq. 45)
S = css.scale_space_operator(basis, s1, "gaussian")
print("S_sigma - P_slow(M=1):\n", S - basis.slow_projector(1))

# Scale-dependent pointers (Eqs. 56-57)
D, Sp = css.pointers(basis, [1e-5, s1, s2, 20.0])
print("fast pointers D^k at four scales:\n", D, "\nsum_k D^k:", D.sum(axis=1))

# Formulation A: filtered velocity at fixed state (Eq. 42)
print("g(y)        :", model.g(y))
print("g_sigma_c1  :", css.filtered_field(model, y, s1))

# Formulation B: relax the state along the fast fibres onto the 2D and 1D manifolds (Eq. 49)
for M in (1, 2):
    path = css.fiber_relaxation(model, y, "adaptive", M=M)
    print(f"M = {M}: y_sigma_c = {path.final}  (reached at sigma = {path.sigma[-1]:.3g})")

# First-order refined basis (Eqs. 66-73) and the two-stage reduced model (Eq. 50)
refined = css.csp_basis(model, y, order=1, M=1)
print("B^(1) A^(1) = I :", np.allclose(refined.B @ refined.A, np.eye(3)))
rhs = css.two_stage_rhs(model, M=1, sigma="adaptive", order=1)
y_red = css.integrate(rhs, css.fiber_relaxation(model, y, "adaptive", M=1, order=1).final, np.linspace(0, 5, 6))
print("reduced-model states at t = 0..5:\n", y_red.T)
