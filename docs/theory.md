# Theory and its implementation

This page follows the paper section by section and names the function that implements each
formula. Notation: $N$ variables, modes ordered $\tau_1<\dots<\tau_N$, $M$ exhausted modes.

## CSP decomposition (Section 3)

At every state the tangent space is spanned by column vectors $\mathbf a_i$ (matrix $\mathbf A$)
with dual row vectors $\mathbf b^i$ (matrix $\mathbf B$), $\mathbf{BA}=\mathbf I$. The vector field
is expanded in modes,

$$\mathbf g(\mathbf y)=\sum_{i=1}^{N}\mathbf a_i f^i,\qquad f^i=\mathbf b^i\cdot\mathbf g .$$

Zeroth-order basis: right and left eigenvectors of the Jacobian, $\tau_i=1/|\lambda_i|$.

→ {func}`csp_scalespace.csp.eigenbasis`, {class}`csp_scalespace.csp.CSPBasis`

## Modal scale-space (Section 4.2)

Each amplitude is attenuated along the scale axis, $\partial_\sigma f^i=-d_i f^i$, giving
$f^i(\sigma)=w_i f^i$ with the survival weight

$$w_i(\mathbf y;\sigma)=\exp\!\big[-(\sigma/\tau_i)^p\big]\quad\text{or}\quad
w_i=\frac{1}{1+(\sigma/\tau_i)^2}.$$

→ {class}`csp_scalespace.filters.ScaleFilter`

## Formulation A: velocity filtration (Section 4.3)

$$\mathbf g_\sigma=\mathcal S_\sigma\mathbf g,\qquad \mathcal S_\sigma=\mathbf{A\,W\,B},\qquad
\partial_\sigma\mathbf g_\sigma=-\mathcal D_\sigma\mathbf g_\sigma,\quad \mathcal D_\sigma=\mathbf{A\,D\,B}.$$

For $p\to\infty$, $\mathcal S_\sigma$ is the classical slow projector with $M(\sigma)$ fast modes.

→ {func}`~csp_scalespace.scalespace.scale_space_operator`,
{func}`~csp_scalespace.scalespace.diffusion_tensor`,
{func}`~csp_scalespace.scalespace.filtered_field`

## Formulation B: state fibre filtration (Section 4.4)

Filtering the modal displacements $u_i=f^i/\lambda_i$ moves the state itself:

$$\frac{\partial\mathbf y_\sigma}{\partial\sigma}=-\sum_i\mathbf a_i(\mathbf y_\sigma)
\frac{d_i(\mathbf y_\sigma;\sigma)}{\lambda_i(\mathbf y_\sigma)}f^i(\mathbf y_\sigma),\qquad \mathbf y_{0}=\mathbf y .$$

This is the continuous version of the CSP radical correction
$\Delta\mathbf y=-\sum_{i\le M}\mathbf a_i\lambda_i^{-1}f^i$. The implementation integrates in
$u=\ln\sigma$ with a stiff solver.

→ {func}`~csp_scalespace.scalespace.fiber_relaxation`,
{func}`~csp_scalespace.scalespace.radical_correction`

## Cutoff scale and recovery of classical CSP (Section 4.5)

For a gap $\epsilon=\tau_M/\tau_{M+1}\ll1$ and $\sigma_c=\sqrt{\tau_M\tau_{M+1}}$:
$w_{i\le M}\le e^{-1/\epsilon}$, $w_{j>M}=1-O(\epsilon)$ and
$\mathcal S_{\sigma_c}=\mathbf P_{slow}(M)+O(\epsilon)$ (Theorem 4.2).

→ {func}`~csp_scalespace.scalespace.cutoff_scale`,
{func}`~csp_scalespace.scalespace.active_dimension`

## Scale-dependent pointers (Section 4.6)

$$S^k(\mathbf y;\sigma)=\sum_i w_i\,a_i^k b^i_k,\qquad D^k=1-S^k,\qquad \sum_kD^k\approx M(\sigma).$$

$D^k\approx1$: variable $k$ is slaved (quasi-steady) at this scale; $S^k\approx1$: it is an active
slow variable.

→ {func}`~csp_scalespace.scalespace.pointers`, {meth}`csp_scalespace.csp.CSPBasis.mode_pointers`

## First-order refinements (Section 5)

With $\dot{\mathbf J}=(\mathbf g\cdot\nabla)\mathbf J$ and $C^m_n=\mathbf b^m\dot{\mathbf J}\mathbf a_n$
(Remark 5.2),

$$P^m_{0,n}=\frac{C^m_n}{\lambda_m(\lambda_m-\lambda_n)},\qquad
Q^n_{0,m}=\frac{C^n_m}{\lambda_m(\lambda_n-\lambda_m)},\qquad m\le M<n,$$

$$\mathbf A_s^{(1)}=\mathbf A_s-\mathbf A_f\mathbf P_0,\quad
\mathbf A_f^{(1)}=\mathbf A_f-\mathbf A_s^{(1)}\mathbf Q_0,\quad
\mathbf B_f^{(1)}=\mathbf B_f+\mathbf P_0\mathbf B_s,\quad
\mathbf B_s^{(1)}=\mathbf B_s+\mathbf Q_0\mathbf B_f^{(1)} .$$

The refined basis is exactly biorthogonal and makes the filtered field tangent to the SIM up to
$O(\epsilon^2)$ (Theorem 5.3). $\dot{\mathbf J}$ is exact for models given as equations.

→ {func}`csp_scalespace.csp.refine_basis`, {func}`csp_scalespace.csp.csp_basis`

## Unified two-stage operator (Eq. 50)

Stage 1: $\mathbf y\mapsto\mathbf y_\sigma$ (Formulation B); Stage 2:
$\dot{\mathbf y}_\sigma=\mathcal S_\sigma(\mathbf y_\sigma)\mathbf g(\mathbf y_\sigma)$ (Formulation A).

→ {func}`~csp_scalespace.scalespace.two_stage_rhs`,
{meth}`csp_scalespace.study.ScaleSpaceStudy.reduced`
