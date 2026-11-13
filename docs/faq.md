# Questions and troubleshooting

## The reduced model is slightly slower than the full model. Why?

With the Gaussian filter the slow modes are not left entirely untouched at the cutoff scale: for a
gap $\epsilon=\tau_M/\tau_{M+1}$ the slowest surviving mode keeps the weight
$w_{M+1}(\sigma_c)=e^{-\epsilon}=1-O(\epsilon)$ (Theorem 4.2 of the paper). The same attenuation
acts during Stage 1, which therefore also moves the state a little *along* the manifold. Both
effects vanish as $\epsilon\to0$ but are visible for moderate gaps. For the Davis-Skodje system:

| $\epsilon$ | filter | $w_2(\sigma_c)$ | Stage-1 state | max. error of the reduced model |
|---|---|---|---|---|
| 0.08 | `gaussian` | 0.9231 | (1.3847, 0.5807) | 6.2e-2 |
| 0.08 | `exponential`, `p: 4` | 0.9936 | (1.4904, 0.5985) | 8.8e-3 |
| 0.01 | `gaussian` | 0.9900 | (1.4851, 0.5976) | 1.4e-2 |
| 0.01 | `exponential`, `p: 4` | 0.9999 | (1.4999, 0.6000) | 1.2e-2 |

A sharper filter (`filter: exponential`, `p: 4` or larger) confines the attenuation to the modes
below the cutoff. The remaining error in the last row is the initial layer itself: the reduced
model starts on the manifold, the full model needs a time of order $\tau_1$ to reach it.

## `ComplexSpectrumWarning`

The Jacobian has complex eigenvalues at some state that was visited. The formulation assumes a
real spectrum (Assumption 3.1; complex pairs are discussed as an open problem in Section 8.5 of the
paper). The computation continues with the real parts, but pointers and reduced models are not
reliable where the warning applies. If it appears only at an isolated state - for example an
initial condition at which two eigenvalues coincide - the rest of the analysis is unaffected.

## `undefined symbol(s) [...] in '...'`

A name in an equation is neither a variable, nor a parameter, nor a definition. Add it to
`parameters` or `definitions`. In YAML, also check that numbers are written as `1.0e-3` (with the
dot): `1e-3` is read as text by some YAML parsers.

## No reduction level is found (`levels: []`)

By default a level $M$ is analysed when mode $M$ is dissipative ($\lambda_M<0$) and
$\tau_{M+1}$ is finite at the initial state. If the initial state is special (an equilibrium of a
sub-system, zero concentrations), set the levels explicitly: `levels: [1]`.

## The fibre relaxation fails or takes very long

Formulation B is integrated in $\ln\sigma$. With the Gaussian filter its rates grow like
$(\sigma/\tau_i)^2$, which becomes extremely stiff when the scale axis covers many decades. Use
`fiber_filter: rational` (bounded rates), and/or narrow `sigma_range`.

## The sum of the fast pointers is not an integer

Inside a wide spectral gap $\sum_kD^k$ equals the number of exhausted modes $M$ to high accuracy.
Deviations mean that the gap is narrow at that state: the weight of the first slow mode is already
below one, or the last fast mode has not yet vanished. This is information, not an error - see the
transition zone of the {doc}`Michaelis-Menten tutorial <tutorial_michaelis_menten>`.

## Individual pointers are negative or larger than one

Pointers are diagonal entries of oblique projectors; they sum to the dimension of the subspace but
are not individually confined to $[0,1]$ when the basis vectors are far from orthogonal.

## How large can the system be?

Each evaluation performs a dense eigen-decomposition of the $N\times N$ Jacobian, and the
symbolic set-up differentiates every equation with respect to every variable. Systems with tens of
variables are comfortable; for hundreds, supply numerical `rhs`/`jac` functions and restrict the
figures with `--only`.

## Can I use the figures in a publication?

Yes. Request vector output with `--formats pdf` (or `formats=("pdf",)` in Python). The plotting
style can be changed by wrapping the call in a Matplotlib `rc_context`, and every figure function
in {mod}`csp_scalespace.figures` returns a normal Matplotlib `Figure` that can be edited further.
