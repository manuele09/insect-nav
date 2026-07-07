from pathlib import Path
from typing import Any, Dict, List
import re

import pandas as pd

from insect_nav.parameters import load_parameters_from_file, save_parameters_to_file


def vars_to_dict(vars: List, network_type: str) -> Dict:
    """Converts a list of variables in a dictionary

    Args:
        vars (List):

    Returns:
        Dict:
    """
    if network_type == "spiking":
        vars_dict = {
            "target_kcs": int(round(vars[0])),
            "pn_kc_fan_in": int(round(vars[1])),
            "pn_kc_weight": vars[2],
            "vthresh": vars[3],
            "vertical_weight": vars[4],
            "horizontal_weight": vars[5],
            "train_step": int(round(vars[6]))
        }
    else:
        vars_dict = {
            "lr": vars[0],
            "ou": int(vars[1]),
            "ts": int(vars[2])
        }
    return vars_dict

def vars_to_name(vars: List, network_type: str) -> str:
    """Converts a list of variables in a name string.

    Args:
        vars (List):

    Returns:
        str:
    """
    vars_dict = vars_to_dict(vars, network_type)
    if network_type == "spiking":
        name = (
            f"t{vars_dict['target_kcs']}"
            f"_f{vars_dict['pn_kc_fan_in']}"
            f"_w{vars_dict['pn_kc_weight']}"
            f"_v{vars_dict['vthresh']}"
            f"_ver{vars_dict['vertical_weight']}"
            f"_hor{vars_dict['horizontal_weight']}"
            f"_ts{vars_dict['train_step']}"
            )
    else:
        name = (
            f"lr{vars_dict['lr']}"
            f"_ou{vars_dict['ou']}"
            f"_ts{vars_dict['ts']}"
        )
    return name

def name_to_vars(name: str, network_type: str) -> List:
    if network_type == "spiking":
        pattern = (
            r"t(?P<target_kcs>-?\d+)"
            r"_f(?P<pn_kc_fan_in>-?\d+)"
            r"_w(?P<pn_kc_weight>-?\d*\.?\d+)"
            r"_v(?P<vthresh>-?\d*\.?\d+)"
            r"_ver(?P<vertical_weight>-?\d*\.?\d+)"
            r"_hor(?P<horizontal_weight>-?\d*\.?\d+)"
            r"_ts(?P<train_step>-?\d+)"
        )

        match = re.fullmatch(pattern, name)
        if match is None:
            raise ValueError(f"Formato nome rete non valido: {name}")

        vars_list = [
            int(match.group("target_kcs")),
            int(match.group("pn_kc_fan_in")),
            float(match.group("pn_kc_weight")),
            float(match.group("vthresh")),
            float(match.group("vertical_weight")),
            float(match.group("horizontal_weight")),
            int(match.group("train_step")),
        ]

    else:
        pattern = (
            r"lr(?P<lr>-?\d*\.?\d+)"
            r"_ou(?P<ou>-?\d+)"
            r"_ts(?P<ts>-?\d+)"
        )

        match = re.fullmatch(pattern, name)
        if match is None:
            raise ValueError(f"Formato nome rete non valido: {name}")

        vars_list = [
            float(match.group("lr")),
            int(match.group("ou")),
            int(match.group("ts")),
        ]

    return vars_list

# Return parameters dict (used by neural networks)
def vars_to_params_dict(vars: List, network_type: str, base_net_param_path: str, output_path: str):
    vars_dict = vars_to_dict(vars, network_type)
    individual_name = vars_to_name(vars, network_type) # Define unique run ranme
    individual_folder = output_path / individual_name
    individual_folder.mkdir(exist_ok=True) # Folder where to save network data

    # Load base parameters
    params = load_parameters_from_file(base_net_param_path / "parameters.json")

    # Update them using variables data
    params.update({
        "base_folder": str(individual_folder),
        "parameters_path": str(individual_folder / "parameters.json"),
        "name": individual_name,
        "plotsTrainPath": str(individual_folder / "plots" / "training"),
        "plotsTestPath": str(individual_folder / "plots" / "testing"),
        "plotsSimulationPath": str(individual_folder / "plots" / "simulation"),
        "weightsPath": str(individual_folder / "weights")
        })

    if network_type == "spiking":
        params.update({
        "USE_VERTICAL_DIST": True,
        "VERTICAL_WEIGHT": vars_dict["vertical_weight"],
        "USE_HORIZONTAL_DIST": True,
        "HORIZONTAL_WEIGHT": vars_dict["horizontal_weight"],
        "train_step": vars_dict["train_step"],
        "target_kcs": vars_dict["target_kcs"],
        "PN_KC_FAN_IN": vars_dict["pn_kc_fan_in"],
        "PN_KC_WEIGHT": vars_dict["pn_kc_weight"],
        "IF_PARAMS": {**params["IF_PARAMS"], "Vthresh": vars_dict["vthresh"]}
        })
    else:
        params.update({
        "learning_rate": vars_dict["lr"],
        "USE_VERTICAL_DIST": True,
        "USE_HORIZONTAL_DIST": True,
        "output_units": vars_dict["ou"],
        "train_step": vars_dict["ts"],
        })
    save_parameters_to_file(params, params["parameters_path"])
    return params

def params_dict_to_vars(params: Dict[str, Any], network_type: str) -> List:
    """
    Extracts the optimization variables list from a parameters dictionary,
    consistent with vars_to_dict / vars_to_params_dict.

    Args:
        params (Dict[str, Any]): parameters dictionary (e.g., loaded from parameters.json)
        network_type (str): "spiking" or other

    Returns:
        List: variables list ordered exactly as expected by vars_to_dict / vars_to_name
    """
    if network_type == "spiking":
        # Required keys (with robust fallbacks where reasonable)
        target_kcs = params.get("target_kcs", None)
        pn_kc_fan_in = params.get("PN_KC_FAN_IN", None)
        pn_kc_weight = params.get("PN_KC_WEIGHT", None)

        if_params = params.get("IF_PARAMS", {}) or {}
        vthresh = if_params.get("Vthresh", None)

        # Weights for vertical/horizontal distances
        vertical_weight = params.get("VERTICAL_WEIGHT", None)
        horizontal_weight = params.get("HORIZONTAL_WEIGHT", None)

        train_step = params.get("train_step", None)

        missing = []
        for k, v in [
            ("target_kcs", target_kcs),
            ("PN_KC_FAN_IN", pn_kc_fan_in),
            ("PN_KC_WEIGHT", pn_kc_weight),
            ("IF_PARAMS.Vthresh", vthresh),
            ("VERTICAL_WEIGHT", vertical_weight),
            ("HORIZONTAL_WEIGHT", horizontal_weight),
            ("train_step", train_step),
        ]:
            if v is None:
                missing.append(k)

        if missing:
            raise KeyError(f"params_dict_to_vars(spiking): chiavi mancanti o None: {missing}")

        # IMPORTANT: keep the same ordering as vars_to_dict expects
        return [
            int(target_kcs),
            int(pn_kc_fan_in),
            float(pn_kc_weight),
            float(vthresh),
            float(vertical_weight),
            float(horizontal_weight),
            int(train_step),
        ]

    else:
        lr = params.get("learning_rate", None)
        ou = params.get("output_units", None)
        ts = params.get("train_step", None)

        missing = []
        for k, v in [("learning_rate", lr), ("output_units", ou), ("train_step", ts)]:
            if v is None:
                missing.append(k)

        if missing:
            raise KeyError(f"params_dict_to_vars(non-spiking): chiavi mancanti o None: {missing}")

        return [
            float(lr),
            int(ou),
            int(ts),
        ]


def get_best_individual_per_generation(csv_path):
    """
    Ritorna una lista di params_path, uno per ogni generazione (il migliore per Error).
    La lista ha lunghezza pari al numero di generazioni presenti nel CSV.
    """
    csv_path = Path(csv_path)
    parent_dir = csv_path.parent

    df = pd.read_csv(csv_path)

    required_cols = {"PopName", "Error", "Generation"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    df["Generation"] = pd.to_numeric(df["Generation"], errors="coerce")
    df["Error"] = pd.to_numeric(df["Error"], errors="coerce")

    df = df.dropna(subset=["PopName", "Error", "Generation"]).copy()
    if df.empty:
        return []

    # Ordina per (Generation asc, Error asc) per poter prendere il primo di ogni generazione
    df = df.sort_values(by=["Generation", "Error"], ascending=[True, True], na_position="last")

    # Se ci sono duplicati di PopName dentro la stessa generazione, tieni il migliore (Error min)
    df = df.drop_duplicates(subset=["Generation", "PopName"], keep="first")

    best_rows = df.groupby("Generation", as_index=False, sort=True).first()
    best_rows = best_rows.sort_values(by="Generation", ascending=True)

    # Rimuovi duplicati tra generazioni (stesso individuo ripetuto)
    best_rows = best_rows.drop_duplicates(subset=["PopName"], keep="first")

    candidates = [
        str(parent_dir / pop_name / "parameters.json")
        for pop_name in best_rows["PopName"].tolist()
    ]

    return candidates
