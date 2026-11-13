# Command line

Installing the package provides the command `csp-scalespace` with four sub-commands.

## `run` - analyse a model

```text
csp-scalespace run MODEL [-o OUTDIR] [--formats FMT ...] [--only KEY ...] [--dpi DPI] [--set NAME=VALUE ...]
```

| argument | meaning |
|---|---|
| `MODEL` | path of a YAML/JSON {doc}`model file <model_file>`, or the name of a built-in model |
| `-o`, `--outdir` | output directory (default `figures_<model name>`) |
| `--formats` | one or more figure formats, e.g. `png pdf svg` (default `png`) |
| `--only` | restrict the output to groups (`theory`, `applications`) and/or figure keys such as `03_filter_weights` or `filter_weights` |
| `--dpi` | resolution of raster formats (default 200) |
| `--set` | override parameter values of the model, e.g. `--set eps=0.01 k2=0.5` |

Examples:

```bash
csp-scalespace run my_model.yaml -o figs
csp-scalespace run three_scale --formats png pdf
csp-scalespace run davis_skodje --set eps=0.01 --only theory
csp-scalespace run my_model.yaml --only timescales pointers_vs_scale
```

The command prints one line per figure with the time it took, and writes `report.md` next to the
figures.

## `template` - start a model file

```bash
csp-scalespace template my_model.yaml
```

writes a commented model file (shown in {doc}`model_file`) that runs as it is. The command refuses
to overwrite an existing file.

## `models` - list the built-in models

```text
davis_skodje       2D Davis-Skodje system: one fast and one slow mode, exactly known curved SIM.
michaelis_menten   Michaelis-Menten kinetics, Case 3: sQSSA -> PEA -> rQSSA regime shift at t ~ 10 s.
three_scale        3D three-timescale network: sequential exhaustion of two fast modes (2D SIM, then 1D SIM).
```

The path of each YAML file is printed as well, so that it can be copied and modified.

## `figures` - list the figure catalogue

Prints key, group, corresponding figure of the paper and a one-line caption for every entry of the
{doc}`catalogue <figures>`.

## Reproducing the paper

The manuscript figures have their own script, see {doc}`paper`:

```bash
python paper/reproduce_paper_figures.py
```
