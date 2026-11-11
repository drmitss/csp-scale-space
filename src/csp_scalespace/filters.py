"""Scale filters: modal survival weights and attenuation rates.

A scale filter answers the question *"what fraction of mode* ``i`` *survives
when the system is observed at scale* :math:`\\sigma` *?"*  The answer is the
weight :math:`w_i(\\mathbf{y};\\sigma)\\in[0,1]`, which depends only on the ratio
:math:`\\sigma/\\tau_i(\\mathbf{y})` (Section 4.2).

``"exponential"``
    :math:`w_i=\\exp[-(\\sigma/\\tau_i)^p]` (Eq. 36), rate
    :math:`d_i=p\\,\\sigma^{p-1}/\\tau_i^p`.
``"gaussian"``
    The exponential filter with :math:`p=2` (default of the paper), rate
    :math:`d_i=2\\sigma/\\tau_i^2`.
``"rational"``
    :math:`w_i=1/[1+(\\sigma/\\tau_i)^2]` (Eq. 37), rate
    :math:`d_i=2\\sigma/(\\tau_i^2+\\sigma^2)`.
``"sharp"``
    :math:`w_i=\\Theta(\\tau_i-\\sigma)`, the limit :math:`p\\to\\infty`; no rate.

The rate is :math:`d_i=-\\partial_\\sigma\\ln w_i` (Eq. 33).

The sharp filter reproduces classical CSP with ``M(σ)`` exhausted modes
(Eq. 45).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ScaleFilter", "as_filter"]


@dataclass(frozen=True)
class ScaleFilter:
    """A scale filter.

    Parameters
    ----------
    kind : {"gaussian", "exponential", "rational", "sharp"}
    p : float
        Sharpness exponent of the exponential family (ignored otherwise).
    """

    kind: str = "gaussian"
    p: float = 2.0

    def __post_init__(self):
        if self.kind not in ("gaussian", "exponential", "rational", "sharp"):
            raise ValueError(f"unknown filter kind '{self.kind}'")
        if self.kind == "gaussian":
            object.__setattr__(self, "p", 2.0)

    def weights(self, sigma, tau) -> np.ndarray:
        """Survival weights :math:`w_i(\\sigma)` for timescales ``tau`` (broadcasts)."""
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = np.asarray(sigma, dtype=float) / np.asarray(tau, dtype=float)
            if self.kind == "sharp":
                return np.where(r < 1.0, 1.0, 0.0)
            if self.kind == "rational":
                return 1.0 / (1.0 + r ** 2)
            return np.exp(-(r ** self.p))

    def log_rates(self, sigma, tau) -> np.ndarray:
        """Dimensionless rates :math:`\\sigma\\,d_i(\\sigma)` (rate per unit :math:`\\ln\\sigma`)."""
        if self.kind == "sharp":
            raise ValueError("the sharp filter has no attenuation rate; use a smooth filter")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = np.asarray(sigma, dtype=float) / np.asarray(tau, dtype=float)
            if self.kind == "rational":
                return np.where(r < 1e8, 2.0 * r ** 2 / (1.0 + r ** 2), 2.0)
            return self.p * r ** self.p

    def rates(self, sigma, tau) -> np.ndarray:
        """Attenuation rates :math:`d_i(\\mathbf{y};\\sigma)` of Eq. (33)."""
        sigma = np.asarray(sigma, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.kind in ("gaussian", "exponential"):
                return self.p * sigma ** (self.p - 1.0) / np.asarray(tau, dtype=float) ** self.p
            return np.where(sigma > 0, self.log_rates(sigma, tau) / sigma, 0.0)

    @property
    def label(self) -> str:
        """LaTeX label of the weight, for figure legends."""
        if self.kind == "rational":
            return r"1/[1+(\sigma/\tau_i)^2]"
        if self.kind == "sharp":
            return r"\Theta(\tau_i-\sigma)"
        p = int(self.p) if float(self.p).is_integer() else self.p
        return rf"e^{{-(\sigma/\tau_i)^{{{p}}}}}"


def as_filter(spec) -> ScaleFilter:
    """Convert ``"gaussian"``, ``("exponential", 4)``, a dict or a filter into a :class:`ScaleFilter`."""
    if isinstance(spec, ScaleFilter):
        return spec
    if spec is None:
        return ScaleFilter()
    if isinstance(spec, str):
        return ScaleFilter(spec)
    if isinstance(spec, dict):
        return ScaleFilter(**spec)
    return ScaleFilter(*spec)
