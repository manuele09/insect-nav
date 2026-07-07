"""
Evolutionary tuning of network hyperparameters via differential evolution,
using a PerfectMemory baseline as teacher.

Note: Tuner/PmTeacher still expect caller-provided, project-specific output
directory conventions (see e.g. a downstream tune.py entry point) — this
subpackage only holds the reusable tuning logic itself.
"""

from insect_nav.tuning.pm_teacher import PmTeacher
from insect_nav.tuning.tuner import Tuner
from insect_nav.tuning.pca_plotter import PCAPlotter
from insect_nav.tuning.utilities import (
    get_best_individual_per_generation,
    name_to_vars,
    params_dict_to_vars,
    vars_to_dict,
    vars_to_name,
    vars_to_params_dict,
)

__all__ = [
    "PmTeacher",
    "Tuner",
    "PCAPlotter",
    "get_best_individual_per_generation",
    "name_to_vars",
    "params_dict_to_vars",
    "vars_to_dict",
    "vars_to_name",
    "vars_to_params_dict",
]
