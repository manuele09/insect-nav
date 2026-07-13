"""
Batch tooling for generating, training, and testing network parameter variants.

Typical workflow (see the CLI wrappers in a downstream project for a concrete
example, e.g. create_net_variants.py / train_variants.py / test_variants.py):

    1. generate_parameter_variants() / apply_parameter_transformation() build a
       cartesian product of parameter sweeps starting from a base parameters.json,
       writing one variant directory per combination.
    2. train_one_variant() trains a single variant in place (delegates the actual
       frame loop to NeuralModelBase.train_batch()).
    3. test_one_variant() evaluates a trained variant's navigation performance.
"""

import itertools
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from insect_nav.infomax import Infomax
from insect_nav.memory import PerfectMemory
from insect_nav.parallel import ParallelNavigator
from insect_nav.parameters import load_parameters_from_file, save_parameters_to_file
from insect_nav.vision import countFrames, loadFrame

try:
    from insect_nav.spiking import NeuralNetwork
except ImportError:
    NeuralNetwork = None

# =============================================================================
# Variant generation
# =============================================================================

# Parametri sweep-abili "sicuri": sono esattamente quelli usati da
# insect_nav.tuning.utilities.params_dict_to_vars/vars_to_name per costruire il
# nome univoco della variante. Sweepare un parametro diverso da questi non
# farebbe collidere i nomi delle cartelle (stesso nome per varianti diverse).
INTEGER_PARAMS = {"target_kcs", "PN_KC_FAN_IN", "train_step"}


def _set_nested(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Imposta d[a][b] = value a partire da una chiave puntata "a.b" (es. "IF_PARAMS.Vthresh")."""
    keys = dotted_key.split(".")
    target = d
    for key in keys[:-1]:
        if not isinstance(target.get(key), dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def cast_sweep_value(param_name: str, raw: str):
    """Converte un valore testuale nel tipo atteso dal parametro (int per i parametri interi noti)."""
    raw = raw.strip()
    if param_name in INTEGER_PARAMS:
        return int(round(float(raw)))
    return float(raw)


def generate_parameter_variants(
    base_params: Dict[str, Any],
    transformations: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """
    Generate parameter variants by creating all combinations of transformation values.

    Args:
        base_params: Base parameter dictionary
        transformations: Dictionary with parameter names as keys and list of values as values
                        Example: {"train_step": [2, 4, 6], "VERTICAL_WEIGHT": [0.1, 0.2]}
                        Dotted keys address nested dicts, e.g. "IF_PARAMS.Vthresh".

    Returns:
        List of parameter dictionaries with transformations applied
    """
    if not transformations:
        return [base_params.copy()]

    variants = []
    param_names = list(transformations.keys())
    param_values = list(transformations.values())

    for combo in itertools.product(*param_values):
        variant = json.loads(json.dumps(base_params))  # deep copy (base_params ha dict annidati)
        for param_name, value in zip(param_names, combo):
            _set_nested(variant, param_name, value)
        variants.append(variant)

    return variants


def apply_parameter_transformation(
    source_params_path: str,
    transformations: Dict[str, List[Any]],
    output_base_dir: str,
    train_dataset_path: Optional[str] = None,
    copy_weights: bool = True,
    halve: bool = False,
) -> List[str]:
    """
    Apply parameter transformations to a base configuration and create variants.

    Args:
        source_params_path: Path to source parameters.json
        transformations: Dictionary of parameters to modify with their value lists
        output_base_dir: Directory where variants will be saved
        train_dataset_path: Optional override for training dataset path. If given,
            sets training_poses_csv (<train_dataset_path>/poses.csv, the trajectory
            REALLY executed during acquisition — the single source of truth for
            navigation-test comparisons/metrics), and also looks for a dataset.json
            manifest in its parent directory and, if found, copies datasetJsonPath +
            panorama_method/total_distance_m/acquisition_step_m/
            waypoints_spline_csv/num_frames into the variant's parameters.
        copy_weights: Whether to copy weights from source network
        halve: If True, sets "halve": True in every created variant's
            parameters.json, switching NeuralNetwork.train() (spiking.py) to
            the extended KC-MBON training procedure (also trains the +/-
            DEGREES_PER_SHIFT presentations, halving instead of zeroing the
            synapse on coincidence). Default False = key omitted entirely,
            variants keep the classic training procedure unchanged. Also
            appends "_halve" to the variant's generated name/folder (after
            "_seed{N}" if present), same convention as seed -- the name is
            characteristic of the parameters that shape training, and halve
            is one of them.

    Returns:
        List of paths to newly created parameters.json files
    """
    source_path = Path(source_params_path)
    source_dir = source_path.parent

    base_params = load_parameters_from_file(source_params_path)
    network_type = base_params.get("network_type", "").lower()

    variants = generate_parameter_variants(base_params, transformations)

    from insect_nav.tuning.utilities import params_dict_to_vars, vars_to_name

    created_paths = []

    for variant_params in variants:
        variant_vars = params_dict_to_vars(variant_params, network_type)
        variant_name = vars_to_name(variant_vars, network_type)

        seed = variant_params.get("seeds", -1)
        variant_name = f"{variant_name}_seed{seed}" if seed >= 1 else variant_name
        variant_name = f"{variant_name}_halve" if halve else variant_name

        dest_dir = Path(output_base_dir) / variant_name
        os.makedirs(dest_dir, exist_ok=True)

        weights_info = "no weights copied"
        if copy_weights and network_type == "spiking":
            src_weights = source_dir / "weights"
            pn_kc_file = src_weights / "pn_kc_ind.npy"
            if pn_kc_file.exists():
                dest_weights = dest_dir / "weights"
                os.makedirs(dest_weights, exist_ok=True)
                shutil.copy2(pn_kc_file, dest_weights / pn_kc_file.name)
                weights_info = "with pn_kc_ind.npy"
            else:
                weights_info = "no pn_kc_ind.npy found"

        variant_params["name"] = variant_name
        if halve:
            variant_params["halve"] = True
        variant_params["parameters_path"] = str(dest_dir / "parameters.json")
        variant_params["weightsPath"] = str(dest_dir / "weights")
        variant_params["plotsTrainPath"] = str(dest_dir / "plots" / "training")
        variant_params["plotsTestPath"] = str(dest_dir / "plots" / "testing")
        variant_params["plotsSimulationPath"] = str(dest_dir / "plots" / "simulation")

        if seed >= 1:
            if NeuralNetwork is None:
                raise ImportError("insect_nav.spiking.NeuralNetwork richiede l'extra [genn] (pygenn).")
            nn = NeuralNetwork(
                variant_params,
                load_net={"pn_kc": False, "kc_mbon": False},
                tuneCurrent=False, connectivity_seed=seed, use_gpu=True)
            nn.save_weights()
            nn.model.unload()
            print(f"   Generated random connectivity with seed {seed}")

        if train_dataset_path:
            variant_params["trainingDatasetPath"] = train_dataset_path

            # training_poses_csv: <trainingDatasetPath>/poses.csv, la
            # traiettoria REALMENTE eseguita durante l'acquisizione (stesso
            # file scritto da DatasetWriter.finalize in ciascuna leaf dir
            # raw/panorama) — l'UNICA fonte di verita' per confronti/metriche
            # di un test di navigazione (vedi
            # control_navigator.resolve_training_poses_csv nel repo
            # Visual-Navigation-Biorobotics). Nessun campo generico
            # 'waypoints_csv' qui: quella era la spline IDEALE (o i waypoint
            # grezzi), non la traiettoria eseguita, e confondere le due era
            # fonte di bug (marker/plot di confronto contro la curva
            # sbagliata).
            variant_params["training_poses_csv"] = str(Path(train_dataset_path) / "poses.csv")

            # Cerca il manifest dataset.json nella cartella superiore (il
            # trainingDatasetPath punta tipicamente a una sottocartella
            # raw/panorama di un dataset acquisito e dataset.json vive nella
            # cartella padre, il manifest_dir). Non e' un errore se non
            # esiste (es. dataset creati a mano).
            dataset_json_path = Path(train_dataset_path).parent / "dataset.json"
            if dataset_json_path.is_file():
                with open(dataset_json_path) as f:
                    dataset_manifest = json.load(f)
                variant_params["datasetJsonPath"] = str(dataset_json_path.resolve())
                for key in ("panorama_method", "total_distance_m", "acquisition_step_m",
                            "waypoints_spline_csv", "num_frames"):
                    if key in dataset_manifest:
                        variant_params[key] = dataset_manifest[key]

        os.makedirs(variant_params["plotsTrainPath"], exist_ok=True)
        os.makedirs(variant_params["plotsTestPath"], exist_ok=True)
        os.makedirs(variant_params["plotsSimulationPath"], exist_ok=True)

        save_parameters_to_file(variant_params, dest_dir / "parameters.json")
        created_paths.append(str(dest_dir / "parameters.json"))

        print(f"Created {variant_name} ({weights_info})")

    return created_paths


# =============================================================================
# Training
# =============================================================================

def train_one_variant(parameters_path: Path) -> None:
    """
    Train a single network variant given the path to its parameters.json file.

    Loads parameters, instantiates the right network class for its
    `network_type`, and delegates the frame-by-frame training loop to
    NeuralModelBase.train_batch() (insect_nav/base.py), which reads
    num_frames/train_step from the network's own parameters, saves the
    weights, and (for spiking nets) tracks novelty and writes
    activated_kcs back into parameters.json.
    """
    parameters = load_parameters_from_file(parameters_path)
    num_frames = countFrames(parameters["trainingDatasetPath"])
    network_type = parameters["network_type"].lower()

    if not parameters.get("train_step"):
        print(f"'{parameters_path}': train_step mancante nel parameters.json, variante saltata.")
        return

    if network_type == "spiking":
        if NeuralNetwork is None:
            raise ImportError("insect_nav.spiking.NeuralNetwork richiede l'extra [genn] (pygenn).")
        nn = NeuralNetwork(
            parameters,
            load_net={"pn_kc": True, "kc_mbon": False},
            tuneCurrent=True, use_gpu=True)
        nn.model.unload()

        nn = NeuralNetwork(
            parameters,
            load_net={"pn_kc": True, "kc_mbon": False},
            tuneCurrent=False, use_gpu=False)

    elif network_type == "infomax":
        nn = Infomax(parameters, load_net=False, calculate_mean=True)

    elif network_type == "perfect_memory":
        nn = PerfectMemory(parameters)

    else:
        print(f"Unknown network type: {parameters['network_type']} in {parameters_path}")
        return

    print(f"\nTraining: {parameters_path}  |  Frames: {num_frames}  |  Step: {parameters['train_step']}")

    nn.train_batch()

    if network_type == "spiking":
        nn.model.unload()
        nn.delete_build_directory()


# =============================================================================
# Testing
# =============================================================================

def test_one_variant(
    parameters_path: Path,
    debug_mode: bool = True,
    parallel_navigation: bool = False,
    frame_ids: Optional[List[int]] = None,
) -> None:
    """
    Test a single network variant using its configuration file.

    Loads network parameters, detects the network type, iterates through
    test frames performing navigation tests, and computes the mean
    absolute angular deviation (degrees).

    Args:
        frame_ids: Explicit list of frame indices to test on. If None
            (default), falls back to the pre-existing behavior of stepping
            through the whole dataset with parameters["train_step"].
    """
    parameters = load_parameters_from_file(parameters_path)
    num_frames = countFrames(parameters["trainingDatasetPath"])

    if frame_ids is None:
        # Backward compatibility for naming
        test_step = parameters.get("train_step")
        if not test_step:
            print(f"'{parameters_path}': train_step mancante nel parameters.json, variante saltata.")
            return
        frame_ids = range(0, num_frames, test_step)

    network_type = parameters["network_type"].lower()

    if parallel_navigation:
        nn = ParallelNavigator(parameters)
    else:
        if network_type == "spiking":
            if NeuralNetwork is None:
                raise ImportError("insect_nav.spiking.NeuralNetwork richiede l'extra [genn] (pygenn).")
            nn = NeuralNetwork(parameters,
                                load_net={"pn_kc": True, "kc_mbon": True},
                                tuneCurrent=False)
        elif network_type == "infomax":
            nn = Infomax(parameters, load_net=True, calculate_mean=False)
        elif network_type == "perfect_memory":
            nn = PerfectMemory(parameters, load_net=True)
        else:
            print(f"Unknown network type: {parameters['network_type']} in {parameters_path}")
            return

    print(f"\nTesting: {parameters_path}  |  Frames: {num_frames}  |  Tested: {len(frame_ids)}")

    degrees_logged = []

    bar = tqdm(frame_ids, unit="frame")
    for frame_number in bar:
        frame = loadFrame(frame_number, frames_dir=parameters["trainingDatasetPath"])
        d = nn.testNavigation(frame, frame_number=frame_number, log_path=parameters["plotsTestPath"], debug_print=False)
        degree_value = math.degrees(d)
        degrees_logged.append(abs(degree_value))
        mean_abs_degree = sum(degrees_logged) / len(degrees_logged)
        bar.set_description(f"Current degree: {degree_value:.2f} deg, current error: {mean_abs_degree:.2f} deg")

        if debug_mode:
            nn.plot_test_results(frame_number, parameters["plotsTestPath"])
            nn.saveFiguresToCsv(frame, frame_number, parameters["plotsTestPath"])

    if degrees_logged:
        print(f"Mean |degree| for {parameters['name']}: {mean_abs_degree:.2f} deg")
    else:
        print(f"No degrees logged for {parameters['name']}.")

    frame_ids_list = list(frame_ids)
    summary = {
        "name": parameters.get("name"),
        "num_frames_total": num_frames,
        "num_frames_tested": len(frame_ids_list),
        "mean_abs_angular_error_deg": mean_abs_degree if degrees_logged else None,
        "frames": [
            {"frame_id": frame_id, "abs_angular_error_deg": degree}
            for frame_id, degree in zip(frame_ids_list, degrees_logged)
        ],
    }
    os.makedirs(parameters["plotsTestPath"], exist_ok=True)
    summary_path = os.path.join(parameters["plotsTestPath"], "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Test summary saved to {summary_path}")

    if NeuralNetwork is not None and isinstance(nn, NeuralNetwork):
        nn.model.unload()
    elif isinstance(nn, ParallelNavigator):
        nn.close()
