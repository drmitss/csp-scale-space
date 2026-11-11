# Assumptions and limitations

| topic | behaviour of the package |
|---|---|
| Complex eigenvalues (paper Section 8.5) | outside the theory; `ComplexSpectrumWarning`, real parts are used, results unreliable at such states. `eigenbasis(J, strict=True)` raises instead. |
| Mode crossing (Assumption 3.3) | modes are re-sorted by $|\lambda_i|$ at every state; near a crossing the refinement divides by $\lambda_m-\lambda_n$ and is ill-conditioned. |
| Non-dissipative modes | Formulation B relaxes only modes with $\lambda_i<0$. Conserved quantities ($\lambda=0$, $\tau=\infty$) are never filtered: $w=1$. |
| Level selection | by default every $M$ with $\lambda_M<0$ and finite $\tau_{M+1}$ at the initial state; set `levels` explicitly otherwise. |
| Stage 2 drift | with the zeroth-order basis the pure tangent flow drifts off a curved manifold at $O(\epsilon)$; `stabilize` (default) or `order: 1` control it. |
| Refined timescales | the first-order basis keeps $\tau_i=1/|\lambda_i|$. |
| Gaussian filter in Formulation B | rates per $\ln\sigma$ grow like $(\sigma/\tau_i)^2$; for sweeps over many decades use `fiber_filter: rational`. |
| Finite-difference derivatives | sufficient for `order: 0`; for `order: 1` prefer symbolic models (exact $(\mathbf g\cdot\nabla)\mathbf J$). |
| Size | dense eigen-decomposition at every evaluation; intended for small and medium $N$. |
| SIM in phase portraits | drawn from the attracting part of the computed trajectory unless `exact_sim` is given; contours $f^{1,(k)}=0$ only for two-variable models. |
