"""Computational Singular Perturbation (CSP) basis vectors.

This module provides the two bases used throughout the paper:

* the **zeroth-order basis** :math:`(\\mathbf{A}^{(0)},\\mathbf{B}^{(0)})` made of
  the right/left eigenvectors of the local Jacobian (Section 4.1, Eqs. 26-28);
* the **first-order refined basis** :math:`(\\mathbf{A}^{(1)},\\mathbf{B}^{(1)})`
  obtained by one :math:`\\mathbf{B}`-refinement followed by one
  :math:`\\mathbf{A}`-refinement (Section 5.2, Eqs. 66-73), which accounts for
  the curvature of the slow invariant manifold and of the fast fibres.

Conventions
-----------
* Modes are ordered from fastest to slowest:
  :math:`\\tau_1 < \\tau_2 < \\dots < \\tau_N` with
  :math:`\\tau_i = 1/|\\lambda_i|` (Eqs. 19-20).
* ``A[:, i]`` is the column vector :math:`\\mathbf{a}_{i+1}` and ``B[i, :]`` the
  dual row vector :math:`\\mathbf{b}^{i+1}`; ``B @ A`` is the identity (Eq. 10).
* ``M`` always denotes the number of modes treated as fast/exhausted.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["CSPBasis", "ComplexSpectrumWarning", "eigenbasis", "refine_basis", "csp_basis"]


class ComplexSpectrumWarning(UserWarning):
    """The Jacobian has complex eigenvalues (outside Assumption 3.1 of the paper)."""


@dataclass
class CSPBasis:
    """CSP basis at one state :math:`\\mathbf{y}`.

    Attributes
    ----------
    lam : (N,) array
        Eigenvalues :math:`\\lambda_i`, fastest first.
    tau : (N,) array
        Timescales :math:`\\tau_i = 1/|\\lambda_i|` (``inf`` for a zero eigenvalue).
    A : (N, N) array
        Columns are the basis vectors :math:`\\mathbf{a}_i`.
    B : (N, N) array
        Rows are the dual vectors :math:`\\mathbf{b}^i`.
    order : int
        0 for the eigenbasis, 1 after one B- and one A-refinement.
    M : int
        Number of fast modes used for the refinement (0 when ``order == 0``).
    """

    lam: np.ndarray
    tau: np.ndarray
    A: np.ndarray
    B: np.ndarray
    order: int = 0
    M: int = 0

    @property
    def n(self) -> int:
        return self.lam.size

    def amplitudes(self, g) -> np.ndarray:
        """Modal amplitudes :math:`f^i = \\mathbf{b}^i\\cdot\\mathbf{g}` (Eq. 12)."""
        return self.B @ np.asarray(g, dtype=float)

    def mode_pointers(self) -> np.ndarray:
        """Array ``P[i, k] = a_i^k b^i_k``: weight of variable ``k`` in mode ``i``.

        Each row sums to one.  These are the building blocks of the CSP
        pointers (Eqs. 54-57).
        """
        return self.A.T * self.B

    def projector(self, modes) -> np.ndarray:
        """Oblique projector :math:`\\sum_{i\\in modes}\\mathbf{a}_i\\mathbf{b}^i` (Eq. 13)."""
        modes = np.atleast_1d(modes)
        return self.A[:, modes] @ self.B[modes, :]

    def fast_projector(self, M: int) -> np.ndarray:
        """:math:`\\mathbf{P}_{fast}(\\mathbf{y};M)` of Eq. (13)."""
        return self.projector(np.arange(M))

    def slow_projector(self, M: int) -> np.ndarray:
        """:math:`\\mathbf{P}_{slow}(\\mathbf{y};M)` of Eq. (13)."""
        return np.eye(self.n) - self.fast_projector(M)


def eigenbasis(J, strict: bool = False) -> CSPBasis:
    """Zeroth-order CSP basis: eigen-decomposition of the Jacobian (Eq. 26).

    Parameters
    ----------
    J : (N, N) array
        Local Jacobian.
    strict : bool
        If ``True`` raise instead of warn when the spectrum is complex.

    Returns
    -------
    CSPBasis
        Eigenvalues sorted by decreasing magnitude (fastest mode first),
        right eigenvectors of unit Euclidean norm whose largest component is
        positive, and the dual basis ``B = inv(A)``.
    """
    J = np.asarray(J, dtype=float)
    vals, vecs = np.linalg.eig(J)
    scale = max(np.max(np.abs(vals)), np.finfo(float).tiny)
    if np.max(np.abs(vals.imag)) > 1e-9 * scale:
        msg = ("Jacobian has complex eigenvalues; the CSP scale-space formulation assumes a real "
               "spectrum (Assumption 3.1). Real parts are used - results are not reliable here.")
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, ComplexSpectrumWarning, stacklevel=2)
    idx = np.argsort(-np.abs(vals.real), kind="stable")
    lam = vals.real[idx]
    # a complex pair is represented by the real and imaginary parts of its eigenvector
    A = np.where(vals.imag[idx] < 0, vecs.imag[:, idx], vecs.real[:, idx]).copy()
    for i in range(A.shape[1]):
        A[:, i] /= np.linalg.norm(A[:, i])
        if A[np.argmax(np.abs(A[:, i])), i] < 0:
            A[:, i] = -A[:, i]
    B = np.linalg.inv(A)
    with np.errstate(divide="ignore"):
        tau = 1.0 / np.abs(lam)
    return CSPBasis(lam, tau, A, B, 0, 0)


def refine_basis(basis: CSPBasis, Jdot, M: int) -> CSPBasis:
    """One B-refinement followed by one A-refinement (Eqs. 66-73).

    The couplings between the fast block (modes ``1..M``) and the slow block
    are computed from the convective derivative of the Jacobian through the
    exact identity of Remark 5.2 (Eq. 74),

    .. math::
       \\big[(\\mathbf{g}\\cdot\\nabla)\\mathbf{b}^m\\big]\\mathbf{a}_n
       = \\frac{\\mathbf{b}^m\\,\\dot{\\mathbf{J}}\\,\\mathbf{a}_n}{\\lambda_m-\\lambda_n},

    so no eigenvector is ever differentiated numerically.

    Parameters
    ----------
    basis : CSPBasis
        Zeroth-order basis from :func:`eigenbasis`.
    Jdot : (N, N) array
        :math:`(\\mathbf{g}\\cdot\\nabla)\\mathbf{J}` at the same state.
    M : int
        Number of fast modes, ``0 < M < N``.  For ``M`` outside this range the
        basis is returned unchanged.

    Returns
    -------
    CSPBasis
        ``order = 1`` basis; exactly biorthogonal (Proposition 5.1).  The
        timescales of the zeroth-order basis are kept (they differ from the
        refined ones only at higher order).
    """
    N = basis.n
    if M <= 0 or M >= N:
        return CSPBasis(basis.lam, basis.tau, basis.A, basis.B, 1, M)
    lam, A, B = basis.lam, basis.A, basis.B
    C = B @ np.asarray(Jdot, dtype=float) @ A  # C[m, n] = b^m Jdot a_n
    lf, ls = lam[:M], lam[M:]
    # P0 = Omega_f^{-1} (dB_f/dt) A_s   (Eqs. 66-67 with Eq. 74)
    P0 = C[:M, M:] / ((lf[:, None] - ls[None, :]) * lf[:, None])
    # Q0 = (dB_s/dt) A_f Omega_f^{-1}   (Eqs. 70-71 with Eq. 74)
    Q0 = C[M:, :M] / ((ls[:, None] - lf[None, :]) * lf[None, :])
    Af, As, Bf, Bs = A[:, :M], A[:, M:], B[:M, :], B[M:, :]
    As1 = As - Af @ P0            # Eq. (69)/(72)
    Af1 = Af - As1 @ Q0           # Eq. (72)
    Bf1 = Bf + P0 @ Bs            # Eq. (68)/(73)
    Bs1 = Bs + Q0 @ Bf1           # Eq. (73)
    return CSPBasis(lam, basis.tau, np.hstack([Af1, As1]), np.vstack([Bf1, Bs1]), 1, M)


def csp_basis(model, y, order: int = 0, M: int | None = None) -> CSPBasis:
    """CSP basis of ``model`` at state ``y``.

    Parameters
    ----------
    model : ODEModel
    y : array_like
        State.
    order : {0, 1}
        0: Jacobian eigenbasis.  1: first-order refined basis.
    M : int, optional
        Number of fast modes for the refinement (required when ``order=1``).
    """
    basis = eigenbasis(model.J(y))
    if order == 0:
        return basis
    if order != 1:
        raise ValueError("only order 0 and 1 are implemented")
    if M is None:
        raise ValueError("order=1 needs the number of fast modes M")
    return refine_basis(basis, model.Jdot(y), int(M))
