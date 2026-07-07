"""JSON parameter I/O, with path rewriting so parameter files stay portable
across machines/users (weights/plots paths derived from the JSON's own
location, trainingDatasetPath rewritten to the current user's home)."""

import getpass
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path


def update_paths(parameters_dict: dict, filename: str) -> dict:
    """Rewrite user- and location-dependent paths in place and return the dict."""
    base_dir = Path(filename).resolve().parent
    current_user = getpass.getuser()

    parameters_dict["parameters_path"] = str(Path(filename).resolve())

    train_path = Path(parameters_dict["trainingDatasetPath"])
    parts = list(train_path.parts)
    parts[2] = current_user
    parameters_dict["trainingDatasetPath"] = str(Path(*parts))

    parameters_dict["weightsPath"] = str(base_dir / "weights")
    parameters_dict["plotsTrainPath"] = str(base_dir / "plots" / "training")
    parameters_dict["plotsTestPath"] = str(base_dir / "plots" / "testing")
    parameters_dict["plotsSimulationPath"] = str(base_dir / "plots" / "simulation")

    return parameters_dict


def save_parameters_to_file(parameters, filename: str) -> None:
    """Save a NetworkConfig or plain dict to a JSON file, refreshing portable paths first."""
    data = asdict(parameters) if is_dataclass(parameters) else dict(parameters)
    data = update_paths(data, filename)
    parent = os.path.dirname(os.path.abspath(filename))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def load_parameters_from_file(filename: str) -> dict:
    """Load a JSON parameter file, refresh portable paths, persist the update, and return it."""
    with open(filename) as f:
        parameters = json.load(f)
    updated = update_paths(parameters, filename)
    save_parameters_to_file(updated, filename)
    return updated
