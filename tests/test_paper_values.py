"""Regression tests against numbers quoted in the manuscript."""
import numpy as np
import pytest

import csp_scalespace as css
from csp_scalespace.models import builtin_model_path, davis_skodje, three_scale


def test_davis_skodje_stage1_state():
    """Section 5.3: y_sigma_c ~ (1.3847, 0.5807) for eps = 0.08, y0 = (1.5, 0.2)."""
    m = davis_skodje(0.08)
    for param in ("linear", "log"):
        p = css.fiber_relaxation(m, m.y0, np.sqrt(0.08), order=1, M=1, param=param)
        assert np.allclose(p.final, [1.3847, 0.5807], atol=1e-4)


def test_davis_skodje_asymptotic_orders():
    """Theorem 5.3 / Fig. 6b: velocity errors O(eps), O(eps^2); manifold errors O(eps^2), O(eps^3)."""
    from csp_scalespace.figures import convergence_data

    st = css.ScaleSpaceStudy.from_file(builtin_model_path("davis_skodje"),
                                       gap_values={"start": -3.5, "stop": -2.0, "num": 6})
    d = convergence_data(st)
    slope = lambda v: np.polyfit(np.log(d["eps"]), np.log(v), 1)[0]  # noqa: E731
    assert slope(d["velocity"][0]) == pytest.approx(1, abs=0.05)
    assert slope(d["velocity"][1]) == pytest.approx(2, abs=0.05)
    assert slope(d["manifold"][0]) == pytest.approx(2, abs=0.05)
    assert slope(d["manifold"][1]) == pytest.approx(3, abs=0.05)


@pytest.fixture(scope="module")
def mm():
    return css.ScaleSpaceStudy.from_file(builtin_model_path("michaelis_menten"), t_end=20.0)


@pytest.mark.parametrize("t, tau1, tau2, sig, Dc", [
    (0.5, 1.05e-4, 8.21e5, 9.29, 1.0), (2.0, 1.25e-4, 5.82e5, 8.53, 1.0),
    (15.0, 2.54e-4, 10.0, 5.04e-2, 0.0), (20.0, 1.58e-4, 10.0, 3.98e-2, 0.0)])
def test_michaelis_menten_table2(mm, t, tau1, tau2, sig, Dc):
    k = int(np.argmin(np.abs(mm.t - t)))
    tau = mm.diagnostics["tau"][k]
    assert tau == pytest.approx([tau1, tau2], rel=0.02)
    assert mm.sigma_c(1)[k] == pytest.approx(sig, rel=0.02)
    D, S = mm.pointers_at(mm.sigma_c(1))
    assert D[k] == pytest.approx([Dc, 1 - Dc], abs=1e-3)
    assert D[k].sum() == pytest.approx(1.0, abs=1e-3)


def test_michaelis_menten_reduced_model_error(mm):
    """Section 6.1: maximum state error 5.9e-3 for the adaptive and sigma_c = 0.1 models."""
    assert mm.reduced(1)["error"] == pytest.approx(5.9e-3, rel=0.05)
    assert mm.reduced(1, 0.1)["error"] == pytest.approx(5.9e-3, rel=0.05)


def test_three_scale_spectrum_and_plateaus():
    """Eq. (104) and Table 3."""
    m = three_scale()
    b = css.csp_basis(m, m.y0)
    assert b.lam == pytest.approx([-1913.4, -24.22, -0.298], rel=2e-3)
    assert css.cutoff_scale(b.tau, 1) == pytest.approx(4.65e-3, rel=5e-3)
    assert css.cutoff_scale(b.tau, 2) == pytest.approx(0.372, rel=5e-3)
    path = css.fiber_relaxation(m, m.y0, 20.0, sigma_min=1e-5, filt="rational")
    assert path.at(6e-3) == pytest.approx([0.874, 0.582, 0.504], abs=2e-3)
    assert path.at(0.40) == pytest.approx([0.959, 0.497, 0.477], abs=2e-3)
