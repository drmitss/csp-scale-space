"""Unit tests of the CSP basis, filters and scale-space operators."""
import numpy as np
import pytest

import csp_scalespace as css
from csp_scalespace.models import davis_skodje, three_scale


@pytest.fixture(scope="module")
def net():
    return three_scale()


def test_basis_is_biorthogonal_and_ordered(net):
    for order in (0, 1):
        b = css.csp_basis(net, net.y0, order, 1)
        assert np.allclose(b.B @ b.A, np.eye(3), atol=1e-10)
        assert np.all(np.diff(b.tau) > 0)
    b = css.csp_basis(net, net.y0)
    assert np.allclose(b.B @ net.J(net.y0) @ b.A, np.diag(b.lam), atol=1e-8 * abs(b.lam[0]))


def test_filter_limits():
    tau = np.array([1e-3, 1.0])
    for kind in ("gaussian", "rational", ("exponential", 4)):
        f = css.as_filter(kind)
        assert np.allclose(f.weights(1e-9, tau), 1.0)
        assert np.allclose(f.weights(1e9, tau), 0.0, atol=1e-12)
        assert f.weights(np.sqrt(1e-3), tau)[0] < 1e-2 < 0.9 < f.weights(np.sqrt(1e-3), tau)[1]
    assert np.isclose(css.ScaleFilter("gaussian").weights(1.0, 1.0), np.exp(-1))
    # d_i = -d ln(w_i)/d sigma
    f, s, h = css.ScaleFilter("rational"), 0.3, 1e-6
    num = -(np.log(f.weights(s + h, tau)) - np.log(f.weights(s - h, tau))) / (2 * h)
    assert np.allclose(f.rates(s, tau), num, rtol=1e-5)


def test_operator_reduces_to_identity_and_slow_projector(net):
    b = css.csp_basis(net, net.y0)
    assert np.allclose(css.scale_space_operator(b, 1e-9), np.eye(3))
    for M in (1, 2):
        sharp = css.scale_space_operator(b, css.cutoff_scale(b.tau, M), "sharp")
        assert np.allclose(sharp, b.slow_projector(M))           # Eq. (45)
        smooth = css.scale_space_operator(b, css.cutoff_scale(b.tau, M))
        assert np.abs(smooth - b.slow_projector(M)).max() < 0.05  # Theorem 4.2: O(eps)


def test_pointers_sum_to_number_of_exhausted_modes(net):
    b = css.csp_basis(net, net.y0)
    D, S = css.pointers(b, [1e-7, css.cutoff_scale(b.tau, 1), css.cutoff_scale(b.tau, 2), 1e4])
    assert np.allclose(D + S, 1.0)
    assert np.allclose(D.sum(axis=1), [0, 1, 2, 3], atol=0.03)


def test_first_order_refinement_matches_closed_form():
    """Eq. (77): the refined Davis-Skodje basis has alpha replaced by alpha^(1)."""
    eps, y1 = 0.08, 1.3
    m = davis_skodje(eps)
    alpha = (1 + y1 + eps * (y1 - 1)) / ((1 - eps) * (1 + y1) ** 3)
    dalpha = ((1 + eps) * (1 + y1) ** 3 - 3 * (1 + y1 + eps * (y1 - 1)) * (1 + y1) ** 2) / ((1 - eps) * (1 + y1) ** 6)
    b0, b1 = css.csp_basis(m, [y1, 0.4]), css.csp_basis(m, [y1, 0.4], 1, 1)
    assert np.isclose(-b0.B[0, 0] / b0.B[0, 1], alpha)
    assert np.isclose(-b1.B[0, 0] / b1.B[0, 1], alpha + eps * y1 * dalpha)
    assert np.allclose(b1.B @ b1.A, np.eye(2))


def test_fiber_relaxation_lands_on_manifold_and_is_consistent(net):
    path = css.fiber_relaxation(net, net.y0, "adaptive", M=1)
    y = path.final
    assert abs(y[0] * y[1] - y[2]) < 2e-3                 # R1 ~ 0: on the 2D SIM
    f = css.csp_basis(net, y).amplitudes(net.g(y))
    f0 = css.csp_basis(net, net.y0).amplitudes(net.g(net.y0))
    assert abs(f[0]) < 1e-6 * abs(f0[0])
    assert np.allclose(path.at(path.sigma[50]), path.y[:, 50], atol=1e-8)


def test_numeric_model_agrees_with_symbolic():
    sym = davis_skodje(0.05)

    def rhs(y, p):
        return [-y[0], -(y[1] - y[0] / (1 + y[0])) / p["eps"] - y[0] / (1 + y[0]) ** 2]

    num = css.ODEModel(rhs, [1.5, 0.2], params={"eps": 0.05})
    y = np.array([0.9, 0.3])
    assert np.allclose(num.J(y), sym.J(y), rtol=1e-6, atol=1e-6)
    assert np.allclose(num.Jdot(y), sym.Jdot(y), rtol=1e-4, atol=1e-4)
    b_num, b_sym = css.csp_basis(num, y, 1, 1), css.csp_basis(sym, y, 1, 1)
    assert np.allclose(b_num.B, b_sym.B, atol=1e-4)


def test_model_validation():
    with pytest.raises(ValueError, match="undefined symbol"):
        css.ODEModel.from_equations({"x": "-k*x"}, {"x": 1.0})
    with pytest.raises(KeyError):
        davis_skodje().with_parameters(nope=1.0)
    with pytest.warns(css.ComplexSpectrumWarning):
        css.eigenbasis(np.array([[0.0, 1.0], [-1.0, -0.1]]))
