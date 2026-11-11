"""csp_scalespace - anisotropic spectral scale-space for multiscale ODEs via CSP.

Reference implementation of

    D. M. Manias and L. Najman, "Anisotropic Spectral Scale-Space Formulation
    for Multiscale Nonlinear Ordinary Differential Equations via Computational
    Singular Perturbation".

Quick start::

    import csp_scalespace as css

    model = css.ODEModel.from_equations(
        equations={"y1": "-y1",
                   "y2": "-(y2 - y1/(1+y1))/eps - y1/(1+y1)**2"},
        parameters={"eps": 0.08},
        initial_conditions={"y1": 1.5, "y2": 0.2})
    study = css.ScaleSpaceStudy(model, t_end=4.0)
    study.make_figures("figures/")
"""
from .csp import CSPBasis, ComplexSpectrumWarning, csp_basis, eigenbasis, refine_basis
from .filters import ScaleFilter, as_filter
from .model import ODEModel, load_model_file
from .scalespace import (FiberPath, active_dimension, cutoff_scale, diffusion_tensor,
                         fiber_relaxation, filtered_field, integrate, pointers,
                         radical_correction, scale_space_operator, two_stage_rhs)
from .study import ScaleSpaceStudy, Settings  # noqa

__version__ = "0.1.0"

__all__ = [
    "ODEModel", "load_model_file", "CSPBasis", "ComplexSpectrumWarning", "csp_basis",
    "eigenbasis", "refine_basis", "ScaleFilter", "as_filter", "FiberPath", "active_dimension",
    "cutoff_scale", "diffusion_tensor", "fiber_relaxation", "filtered_field", "integrate",
    "pointers", "radical_correction", "scale_space_operator", "two_stage_rhs",
    "ScaleSpaceStudy", "Settings", "__version__",
]
