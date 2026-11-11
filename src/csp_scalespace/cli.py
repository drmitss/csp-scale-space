"""Command-line interface: ``csp-scalespace``.

::

    csp-scalespace run MODEL [-o OUTDIR] [--formats png pdf] [--only GROUP_OR_KEY ...]
    csp-scalespace models                 # list the built-in models
    csp-scalespace figures                # list the figure catalogue
    csp-scalespace template my_model.yaml # write a commented model-file template

``MODEL`` is the path of a YAML/JSON model file or the name of a built-in
model (``davis_skodje``, ``michaelis_menten``, ``three_scale``).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


def _resolve(model: str) -> Path:
    from .models import BUILTIN_MODELS, builtin_model_path

    if Path(model).exists():
        return Path(model)
    if model in BUILTIN_MODELS:
        return builtin_model_path(model)
    raise SystemExit(f"error: '{model}' is neither a file nor a built-in model {BUILTIN_MODELS}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="csp-scalespace", description="CSP scale-space analysis of ODE models.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="analyse a model and write all figures")
    run.add_argument("model", help="model file (YAML/JSON) or built-in model name")
    run.add_argument("-o", "--outdir", default=None, help="output directory (default: figures_<name>)")
    run.add_argument("--formats", nargs="+", default=["png"], help="figure formats, e.g. png pdf")
    run.add_argument("--only", nargs="+", default=None,
                     help="figure keys or groups (theory, applications) to produce")
    run.add_argument("--dpi", type=int, default=200)
    run.add_argument("--set", nargs="+", default=[], metavar="NAME=VALUE",
                     help="override model parameters, e.g. --set eps=0.01")
    sub.add_parser("models", help="list built-in models")
    sub.add_parser("figures", help="list the figure catalogue")
    tpl = sub.add_parser("template", help="write a commented model-file template")
    tpl.add_argument("path", nargs="?", default="my_model.yaml")
    args = parser.parse_args(argv)

    if args.command == "models":
        from .models import BUILTIN_MODELS, builtin_model_path, load_builtin

        for name in BUILTIN_MODELS:
            print(f"{name:<18s} {load_builtin(name)[0].description}\n{'':<18s} {builtin_model_path(name)}")
        return 0
    if args.command == "figures":
        from .figures import FIGURES

        for f in FIGURES:
            print(f"{f.key:<30s} {f.group:<13s} {f.paper:<17s} {f.caption}")
        return 0
    if args.command == "template":
        from .models import builtin_model_path

        if Path(args.path).exists():
            raise SystemExit(f"error: {args.path} already exists")
        shutil.copy(builtin_model_path("template"), args.path)
        print(f"template written to {args.path}")
        return 0

    import matplotlib

    matplotlib.use("Agg")
    from .model import load_model_file
    from .study import ScaleSpaceStudy

    model, settings = load_model_file(_resolve(args.model))
    if args.set:
        model = model.with_parameters(**{k: float(v) for k, v in (kv.split("=", 1) for kv in args.set)})
    study = ScaleSpaceStudy(model, settings)
    outdir = Path(args.outdir or f"figures_{model.name}")
    print(f"model '{model.name}': N = {model.n}, levels M = {study.levels}, output -> {outdir}/")
    t0 = time.time()
    written = study.make_figures(outdir, formats=args.formats, which=args.only, dpi=args.dpi, verbose=True)
    print(f"{len(written)} figures and report.md written in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
