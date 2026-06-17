from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional


@dataclass
class NetworkConfig:
    """
    Typed configuration for insect-nav models.

    Behaves like a dict (supports cfg["KEY"] and cfg["KEY"] = val) so that
    existing code written against plain parameter dicts works without changes.
    """

    # ── Vision / preprocessing ───────────────────────────────────────────────
    WIDTH: int = 40
    HEIGHT: int = 8
    CROP_TOP: int = 0
    CROP_BOTTOM: int = 0
    NUM_SHIFTS: int = 10
    DEGREES_PER_SHIFT: float = 10.0
    USE_VERTICAL_DIST: bool = True
    USE_HORIZONTAL_DIST: bool = False
    VERTICAL_WEIGHT: float = 0.0
    HORIZONTAL_WEIGHT: float = 0.0

    # ── Network architecture ─────────────────────────────────────────────────
    name: str = "network"
    NUM_KC: int = 2000
    PN_KC_FAN_IN: int = 10
    PN_KC_WEIGHT: float = 0.1
    PN_KC_TAU: float = 3.0
    KC_APLN_WEIGHT: float = 0.01
    APLN_KC_WEIGHT: float = -0.1
    APLN_KC_TAU: float = 3.0
    KC_MBON_WEIGHT: float = 0.1
    KC_MBON_TAU: float = 3.0

    # ── GeNN neuron model params (populated from JSON) ───────────────────────
    LIF_PARAMS: Dict[str, Any] = field(default_factory=dict)
    LIF_INIT: Dict[str, Any] = field(default_factory=dict)
    IF_PARAMS: Dict[str, Any] = field(default_factory=dict)
    IF_INIT: Dict[str, Any] = field(default_factory=dict)
    KC_MBON_PARAMS: Dict[str, Any] = field(default_factory=lambda: {"mod": -1.0})

    # ── Simulation ───────────────────────────────────────────────────────────
    DT: float = 1.0
    INPUT_SCALE: float = 0.5
    PRESENT_TIME_MS: float = 20.0
    target_kcs: int = 200

    # ── Infomax-specific ─────────────────────────────────────────────────────
    output_units: int = 100
    learning_rate: float = 0.01
    training_dataset_mean: Optional[float] = None
    training_dataset_std: Optional[float] = None

    # ── Training ─────────────────────────────────────────────────────────────
    train_step: int = 1

    # ── Paths ────────────────────────────────────────────────────────────────
    weightsPath: str = "./weights"
    trainingDatasetPath: str = "./dataset"
    parameters_path: str = "./parameters.json"
    plotsTrainPath: str = "./plots/train"
    plotsTestPath: str = "./plots/test"
    plotsSimulationPath: str = "./plots/simulation"

    # ── Runtime / tuning results (written back by tuning routines) ───────────
    mean_num_kc_fired: Optional[float] = None
    std_num_kc_fired: Optional[float] = None
    mean_num_spikes_kc: Optional[float] = None
    std_num_spikes_kc: Optional[float] = None

    # ── Dict-compatible interface ────────────────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def items(self):
        return asdict(self).items()

    # ── Serialization ────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str) -> NetworkConfig:
        """Load config from a JSON file; unknown keys are silently ignored."""
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, path: str) -> None:
        """Save config to JSON."""
        import os
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)
