"""
insect_nav — biologically-inspired visual navigation with spiking neural networks.

Core models:
    NeuralNetwork   — GeNN spiking mushroom body (requires insect_nav[genn])
    Infomax         — information-theoretic unsupervised learner
    PerfectMemory   — snapshot memory baseline

Usage:
    from insect_nav import NetworkConfig, Infomax, PerfectMemory
    from insect_nav import NeuralNetwork  # requires pygenn
"""

from insect_nav.config import NetworkConfig
from insect_nav.base import NeuralModelBase
from insect_nav.infomax import Infomax
from insect_nav.memory import PerfectMemory
from insect_nav.logger import NetworkLogger
from insect_nav.parameters import load_parameters_from_file, save_parameters_to_file
from insect_nav.vision import (
    computeNovelty,
    countFrames,
    cropFrame,
    extractFeatures,
    loadFrame,
    preprocessFrame,
    resizeFrame,
    saveFrameAsPNG,
    saveFeaturesAsPNG,
    shiftFrame,
)

try:
    from insect_nav.spiking import NeuralNetwork
    from insect_nav.genn_models import (
        anti_hebbian,
        cs_model,
        if_model,
        pwl_model,
        reset_model_if,
        reset_model_lif,
        reset_model_syn,
    )
    _GENN_AVAILABLE = True
except ImportError:
    _GENN_AVAILABLE = False

__all__ = [
    # Config
    "NetworkConfig",
    # Models
    "NeuralModelBase",
    "NeuralNetwork",
    "Infomax",
    "PerfectMemory",
    # Logging
    "NetworkLogger",
    # Vision utils
    "loadFrame",
    "countFrames",
    "preprocessFrame",
    "extractFeatures",
    "cropFrame",
    "resizeFrame",
    "shiftFrame",
    "computeNovelty",
    "saveFrameAsPNG",
    "saveFeaturesAsPNG",
    # Parameters I/O
    "load_parameters_from_file",
    "save_parameters_to_file",
]
