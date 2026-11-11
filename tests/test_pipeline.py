"""End-to-end tests: model file -> study -> figures, and the command line."""
import matplotlib

matplotlib.use("Agg")

import csp_scalespace as css
from csp_scalespace.cli import main
from csp_scalespace.figures import FIGURES
from csp_scalespace.models import BUILTIN_MODELS, builtin_model_path


def test_all_figures_for_davis_skodje(tmp_path):
    st = css.ScaleSpaceStudy.from_file(builtin_model_path("davis_skodje"), n_points=200,
                                       gap_values={"start": -3, "stop": -2, "num": 4})
    written = st.make_figures(tmp_path, dpi=50)
    assert len(written) == len(FIGURES)
    assert all(p.stat().st_size > 2000 for paths in written.values() for p in paths)
    assert "sigma_c (M=1)" in (tmp_path / "report.md").read_text()


def test_three_scale_reduced_models_track_full_model():
    st = css.ScaleSpaceStudy.from_file(builtin_model_path("three_scale"), n_points=150)
    assert st.levels == [1, 2]
    assert st.reduced(1)["error"] < 0.02 and st.reduced(2)["error"] < 0.03


def test_cli_template_roundtrip(tmp_path, capsys):
    path = tmp_path / "mine.yaml"
    assert main(["template", str(path)]) == 0
    assert main(["run", str(path), "-o", str(tmp_path / "out"), "--only", "applications", "--dpi", "50"]) == 0
    assert (tmp_path / "out" / "09_timescales.png").exists()
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in BUILTIN_MODELS)
