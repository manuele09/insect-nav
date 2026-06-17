"""Backward-compatible JSON parameter I/O."""

import json
import os
from dataclasses import asdict, is_dataclass


def save_parameters_to_file(parameters, filename: str) -> None:
    """Save a NetworkConfig or plain dict to a JSON file."""
    data = asdict(parameters) if is_dataclass(parameters) else dict(parameters)
    parent = os.path.dirname(os.path.abspath(filename))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def load_parameters_from_file(filename: str) -> dict:
    """Load a JSON parameter file and return a plain dict."""
    with open(filename) as f:
        return json.load(f)
