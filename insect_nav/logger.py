"""
NetworkLogger: logging system for GeNN-based spiking neural networks.

Handles spikes, voltages, currents, and synaptic weights with a flexible
query API, CSV export, and matplotlib visualization.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class NetworkLogger:
    """Logging system for GeNN spiking neural networks."""

    DEFAULT_CONFIG: Dict[str, Dict[str, bool]] = {
        "voltages": {"pn": False, "kc": False, "apln": False, "mbon": False},
        "currents": {"pn_kc": False, "kc_apln": False, "apln_kc": False, "kc_mbon": False},
        "spikes": {"pn": False, "kc": True, "apln": False, "mbon": True},
        "weights": {"kc_mbon": False},
    }

    def __init__(self, network, config: Optional[Dict] = None):
        self._network = network
        self._config = self._merge_config(config)

        self._data: Dict[str, Any] = {
            "voltages": {},
            "currents": {},
            "spikes": {},
            "weights": {},
        }

        self._simulation_active = False
        self._start_step = 0
        self._end_step = 0
        self._simulation_steps = 0
        self._dt = network.params.get("DT")

        self._novelty_data: Dict[str, Any] = {
            "enabled": False,
            "frames": [],
            "features": [],
            "global_kc_fired": set(),
            "cumulative_new_kcs": [],
            "novelty_per_frame": {"cosine": [], "pearson": [], "euclidean": []},
            "novelty_cumulative": {"cosine": [], "pearson": [], "euclidean": []},
            "cumulative_sum": {"cosine": 0.0, "pearson": 0.0, "euclidean": 0.0},
        }

    # ── Config ───────────────────────────────────────────────────────────────

    def _merge_config(self, user_config: Optional[Dict]) -> Dict:
        config = {}
        for key in self.DEFAULT_CONFIG:
            if user_config and key in user_config and isinstance(user_config[key], dict):
                config[key] = {**self.DEFAULT_CONFIG[key], **user_config[key]}
            elif user_config and key in user_config:
                config[key] = user_config[key]
            else:
                config[key] = self.DEFAULT_CONFIG[key].copy()
        return config

    def update_config(self, key: str, value: Dict) -> None:
        if key in self._config:
            if isinstance(value, dict):
                self._config[key].update(value)
            else:
                self._config[key] = value

    # ── Data collection ──────────────────────────────────────────────────────

    def start_logging(self, simulation_steps: int) -> None:
        self._simulation_active = True
        self._start_step = self._network.model.timestep
        self._end_step = self._start_step + simulation_steps
        self._simulation_steps = simulation_steps
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        steps = self._simulation_steps

        for pop_name, enabled in self._config["voltages"].items():
            if enabled and hasattr(self._network, pop_name):
                pop = getattr(self._network, pop_name)
                self._data["voltages"][pop_name] = np.zeros((pop.num_neurons, steps), dtype=np.float32)

        for syn_name, enabled in self._config["currents"].items():
            if enabled and hasattr(self._network, syn_name):
                post_pop_name = syn_name.split("_")[-1]
                if hasattr(self._network, post_pop_name):
                    post_pop = getattr(self._network, post_pop_name)
                    self._data["currents"][syn_name] = np.zeros((post_pop.num_neurons, steps), dtype=np.float32)

        for pop_name in self._config["spikes"]:
            if self._config["spikes"][pop_name]:
                self._data["spikes"][pop_name] = {"times": None, "ids": None}

        for syn_name in self._config["weights"]:
            if self._config["weights"][syn_name]:
                self._data["weights"][syn_name] = []

    def log_step(self, timestep: int) -> None:
        if not self._simulation_active or timestep < self._start_step or timestep >= self._end_step:
            return
        t_idx = timestep - self._start_step

        for pop_name, enabled in self._config["voltages"].items():
            if enabled:
                self._log_voltage(pop_name, t_idx)

        for syn_name, enabled in self._config["currents"].items():
            if enabled:
                self._log_current(syn_name, t_idx)

        for syn_name, enabled in self._config["weights"].items():
            if enabled and timestep == self._end_step - 1:
                self._log_weights(syn_name)

    def _log_voltage(self, pop_name: str, t_idx: int) -> None:
        pop = getattr(self._network, pop_name)
        pop.vars["V"].pull_from_device()
        self._data["voltages"][pop_name][:, t_idx] = pop.vars["V"].values

    def _log_current(self, syn_name: str, t_idx: int) -> None:
        syn = getattr(self._network, syn_name)
        syn.out_post.pull_from_device()
        self._data["currents"][syn_name][:, t_idx] = syn.out_post.view[:][0].copy()

    def _log_weights(self, syn_name: str) -> None:
        syn = getattr(self._network, syn_name)
        syn.vars["g"].pull_from_device()
        self._data["weights"][syn_name].append(syn.vars["g"].values)

    def finalize_logging(self) -> None:
        if not self._simulation_active:
            return
        if any(self._config["spikes"].values()):
            self._network.model.pull_recording_buffers_from_device()
            for pop_name, enabled in self._config["spikes"].items():
                if enabled and hasattr(self._network, pop_name):
                    pop = getattr(self._network, pop_name)
                    spike_times, spike_ids = pop.spike_recording_data[0]
                    self._data["spikes"][pop_name] = {
                        "times": np.array(spike_times),
                        "ids": np.array(spike_ids),
                    }
        self._simulation_active = False

    # ── Query API ────────────────────────────────────────────────────────────

    def get_spikes(self, population: str = "kc",
                   time_range: Optional[Tuple[float, float]] = None,
                   neuron_ids: Optional[Union[List, np.ndarray]] = None) -> Dict:
        empty = {"times": np.array([]), "ids": np.array([]),
                 "count": 0, "neurons_fired": 0,
                 "mean_spikes_per_neuron": 0.0, "std_spikes_per_neuron": 0.0,
                 "cumulative_spike_count": np.array([])}

        if population not in self._data["spikes"]:
            return empty
        spike_data = self._data["spikes"][population]
        if spike_data["times"] is None:
            return empty

        times = spike_data["times"].copy()
        ids = spike_data["ids"].copy()

        if time_range is not None:
            mask = (times >= time_range[0]) & (times <= time_range[1])
            times, ids = times[mask], ids[mask]

        if neuron_ids is not None:
            mask = np.isin(ids, np.asarray(neuron_ids))
            times, ids = times[mask], ids[mask]

        if len(ids) > 0:
            unique_neurons = np.unique(ids)
            counts_per = np.array([np.sum(ids == n) for n in unique_neurons])
            mean_spk = float(np.mean(counts_per))
            std_spk = float(np.std(counts_per))
        else:
            mean_spk = std_spk = 0.0

        return {
            "times": times,
            "ids": ids,
            "count": len(times),
            "neurons_fired": len(np.unique(ids)) if len(ids) > 0 else 0,
            "mean_spikes_per_neuron": mean_spk,
            "std_spikes_per_neuron": std_spk,
            "cumulative_spike_count": np.arange(1, len(times) + 1),
        }

    def get_voltages(self, population: str = "mbon",
                     time_range: Optional[Tuple[float, float]] = None,
                     neuron_ids: Optional[Union[List, np.ndarray]] = None) -> Dict:
        empty = {"voltages": np.array([]), "time_axis": np.array([]), "neuron_ids": np.array([])}
        if population not in self._data["voltages"]:
            return empty

        voltages = self._data["voltages"][population].copy()
        time_axis = np.arange(voltages.shape[1]) * self._dt

        if time_range is not None:
            s = int(time_range[0] / self._dt)
            e = min(int(time_range[1] / self._dt) + 1, voltages.shape[1])
            voltages, time_axis = voltages[:, s:e], time_axis[s:e]

        if neuron_ids is not None:
            neuron_ids = np.asarray(neuron_ids)
            voltages = voltages[neuron_ids, :]
        else:
            neuron_ids = np.arange(voltages.shape[0])

        return {"voltages": voltages, "time_axis": time_axis, "neuron_ids": neuron_ids}

    def get_currents(self, synapse: str = "kc_mbon",
                     time_range: Optional[Tuple[float, float]] = None,
                     neuron_ids: Optional[Union[List, np.ndarray]] = None) -> Dict:
        empty = {"currents": np.array([]), "time_axis": np.array([]), "neuron_ids": np.array([])}
        if synapse not in self._data["currents"]:
            return empty

        currents = self._data["currents"][synapse].copy()
        time_axis = np.arange(currents.shape[1]) * self._dt

        if time_range is not None:
            s = int(time_range[0] / self._dt)
            e = min(int(time_range[1] / self._dt) + 1, currents.shape[1])
            currents, time_axis = currents[:, s:e], time_axis[s:e]

        if neuron_ids is not None:
            neuron_ids = np.asarray(neuron_ids)
            currents = currents[neuron_ids, :]
        else:
            neuron_ids = np.arange(currents.shape[0])

        return {"currents": currents, "time_axis": time_axis, "neuron_ids": neuron_ids}

    def get_weights(self, synapse: str = "kc_mbon",
                    time_range: Optional[Tuple[float, float]] = None) -> Dict:
        if synapse not in self._data["weights"] or not self._data["weights"][synapse]:
            return {"weights": [], "time_axis": np.array([])}

        weights = self._data["weights"][synapse]
        time_axis = np.arange(len(weights)) * self._dt

        if time_range is not None:
            s = int(time_range[0] / self._dt)
            e = min(int(time_range[1] / self._dt) + 1, len(weights))
            weights, time_axis = weights[s:e], time_axis[s:e]

        return {"weights": weights, "time_axis": time_axis}

    # ── Novelty tracking ─────────────────────────────────────────────────────

    def enable_novelty_tracking(self, track_features: bool = True,
                                track_kc_activation: bool = True,
                                novelty_metrics: Optional[List[str]] = None) -> None:
        self._novelty_data["enabled"] = True
        valid = {"cosine", "pearson", "euclidean"}
        if novelty_metrics:
            novelty_metrics = [m for m in novelty_metrics if m in valid]
        else:
            novelty_metrics = list(valid)

        self._novelty_data["novelty_per_frame"] = {m: [] for m in novelty_metrics}
        self._novelty_data["novelty_cumulative"] = {m: [] for m in novelty_metrics}
        self._novelty_data["cumulative_sum"] = {m: 0.0 for m in novelty_metrics}

    def log_training_frame(self, frame_id: int, frame: np.ndarray,
                           preprocessed_frame: Optional[np.ndarray] = None) -> Dict:
        if not self._novelty_data["enabled"]:
            return {}

        from insect_nav.metrics import novelty_scores
        from insect_nav.vision import extractFeatures, preprocessFrame

        if frame_id is None:
            frame_id = len(self._novelty_data["frames"])

        if preprocessed_frame is None:
            preprocessed_frame = preprocessFrame(frame, 0, self._network.params)
        features = extractFeatures(preprocessed_frame, self._network.params)

        novelties = novelty_scores(features, self._novelty_data["features"])

        kc_spike_data = self.get_spikes("kc")
        current_kc_fired = set(kc_spike_data["ids"])
        newly_fired = current_kc_fired - self._novelty_data["global_kc_fired"]
        self._novelty_data["global_kc_fired"] |= newly_fired

        self._novelty_data["frames"].append(frame_id)
        self._novelty_data["cumulative_new_kcs"].append(len(self._novelty_data["global_kc_fired"]))
        self._novelty_data["features"].append(features)

        for metric in self._novelty_data["novelty_per_frame"]:
            self._novelty_data["novelty_per_frame"][metric].append(novelties[metric])
            self._novelty_data["cumulative_sum"][metric] += novelties[metric]
            self._novelty_data["novelty_cumulative"][metric].append(
                self._novelty_data["cumulative_sum"][metric]
            )

        return {
            "frame_id": frame_id,
            "new_kcs": len(newly_fired),
            "total_kcs": len(self._novelty_data["global_kc_fired"]),
            "novelty_cosine": novelties.get("cosine", 0.0),
            "novelty_pearson": novelties.get("pearson", 0.0),
            "novelty_euclidean": novelties.get("euclidean", 0.0),
        }

    def get_novelties(self) -> Dict:
        if not self._novelty_data["enabled"]:
            return {"enabled": False}

        num_frames = len(self._novelty_data["frames"])
        total_kcs = len(self._novelty_data["global_kc_fired"])

        cumulative_kcs = np.asarray(self._novelty_data["cumulative_new_kcs"], dtype=float)
        kc_recruitment_rate = np.diff(cumulative_kcs) if len(cumulative_kcs) > 1 else np.array([])

        return {
            "enabled": True,
            "frames": self._novelty_data["frames"].copy(),
            "num_frames": num_frames,
            "cumulative_new_kcs": self._novelty_data["cumulative_new_kcs"].copy(),
            "kc_recruitment_rate": kc_recruitment_rate,
            "novelty_per_frame": {m: np.array(v) for m, v in self._novelty_data["novelty_per_frame"].items()},
            "novelty_cumulative": {m: np.array(v) for m, v in self._novelty_data["novelty_cumulative"].items()},
            "correlation_cumulative": self._compute_cumulative_correlations(cumulative_kcs),
            "correlation_per_frame": self._compute_per_frame_correlations(kc_recruitment_rate),
            "final_cumulative_novelty": self._novelty_data["cumulative_sum"].copy(),
            "total_unique_kcs": total_kcs,
            "avg_new_kcs_per_frame": total_kcs / num_frames if num_frames > 0 else 0,
            "global_kc_fired": self._novelty_data["global_kc_fired"].copy(),
        }

    def _compute_cumulative_correlations(self, cumulative_kcs: np.ndarray) -> Dict[str, float]:
        return {
            metric: self._safe_pearson(
                np.asarray(self._novelty_data["novelty_cumulative"][metric], dtype=float),
                cumulative_kcs,
            )
            for metric in self._novelty_data["novelty_cumulative"]
        }

    def _compute_per_frame_correlations(self, kc_recruitment_rate: np.ndarray) -> Dict[str, float]:
        if len(kc_recruitment_rate) == 0:
            return {m: "nan" for m in self._novelty_data["novelty_per_frame"]}
        result = {}
        for metric in self._novelty_data["novelty_per_frame"]:
            vals = np.asarray(self._novelty_data["novelty_per_frame"][metric], dtype=float)
            aligned = vals[1: 1 + len(kc_recruitment_rate)]
            result[metric] = self._safe_pearson(aligned, kc_recruitment_rate)
        return result

    def _safe_pearson(self, x: np.ndarray, y: np.ndarray) -> Union[float, str]:
        if len(x) <= 1 or len(y) <= 1 or np.std(x) == 0 or np.std(y) == 0:
            return "nan"
        try:
            return float(np.corrcoef(x, y)[0, 1])
        except (ValueError, RuntimeWarning):
            return "nan"

    def reset_novelty_data(self) -> None:
        self._novelty_data["frames"] = []
        self._novelty_data["features"] = []
        self._novelty_data["global_kc_fired"] = set()
        self._novelty_data["cumulative_new_kcs"] = []
        for metric in self._novelty_data["novelty_per_frame"]:
            self._novelty_data["novelty_per_frame"][metric] = []
            self._novelty_data["novelty_cumulative"][metric] = []
            self._novelty_data["cumulative_sum"][metric] = 0.0

    # ── CSV export ───────────────────────────────────────────────────────────

    def export_to_csv(self, output_path: Union[str, Path],
                      what: Union[str, List[str]] = "all",
                      include_metadata: bool = False) -> None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        if what == "all":
            what = ["spikes", "voltages", "currents", "weights"]
        elif isinstance(what, str):
            what = [what]
        if "spikes" in what:
            self._export_spikes(output_path)
        if "voltages" in what:
            self._export_voltages(output_path)
        if "currents" in what:
            self._export_currents(output_path)
        if "weights" in what:
            self._export_weights(output_path)

    def _export_spikes(self, output_path: Path) -> None:
        spike_dir = output_path / "spikes"
        spike_dir.mkdir(parents=True, exist_ok=True)
        for pop_name in self._config["spikes"]:
            if not self._config["spikes"][pop_name]:
                continue
            spike_data = self._data["spikes"].get(pop_name)
            if spike_data is None or spike_data["times"] is None:
                continue
            with open(spike_dir / f"{pop_name}_spikes.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Time_ms", "Neuron_ID"])
                for t, n in zip(spike_data["times"], spike_data["ids"]):
                    writer.writerow([float(t), int(n)])

    def _export_voltages(self, output_path: Path) -> None:
        volt_dir = output_path / "voltages"
        volt_dir.mkdir(parents=True, exist_ok=True)
        for pop_name, voltages in self._data["voltages"].items():
            if voltages.size == 0:
                continue
            with open(volt_dir / f"{pop_name}_voltages.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Time_ms"] + [f"Neuron_{i}" for i in range(voltages.shape[0])])
                for t_idx in range(voltages.shape[1]):
                    writer.writerow([t_idx * self._dt] + voltages[:, t_idx].tolist())

    def _export_currents(self, output_path: Path) -> None:
        curr_dir = output_path / "currents"
        curr_dir.mkdir(parents=True, exist_ok=True)
        for syn_name, currents in self._data["currents"].items():
            if currents.size == 0:
                continue
            with open(curr_dir / f"{syn_name}_currents.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Time_ms"] + [f"Neuron_{i}" for i in range(currents.shape[0])])
                for t_idx in range(currents.shape[1]):
                    writer.writerow([t_idx * self._dt] + currents[:, t_idx].tolist())

    def _export_weights(self, output_path: Path) -> None:
        weight_dir = output_path / "weights"
        weight_dir.mkdir(parents=True, exist_ok=True)
        for syn_name, weights in self._data["weights"].items():
            if not weights:
                continue
            with open(weight_dir / f"{syn_name}_weights.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Time_ms"] + [f"w_{i}" for i in range(len(weights[0]))])
                for t_idx, w in enumerate(weights):
                    writer.writerow([t_idx * self._dt] + w.tolist())

    def _export_novelties(self, output_path: Union[str, Path]) -> None:
        novelty_data = self.get_novelties()
        if not novelty_data.get("enabled", False):
            return
        output_path = Path(output_path)
        novelty_dir = output_path / "novelties"
        novelty_dir.mkdir(parents=True, exist_ok=True)
        self._export_novelty_frames(novelty_dir, novelty_data)
        self._export_novelty_summary(novelty_dir, novelty_data)

    def _export_novelty_frames(self, output_dir: Path, novelty_data: Dict) -> None:
        csv_path = output_dir / "novelty_frames.csv"
        frames = novelty_data["frames"]
        cumulative_kcs = novelty_data["cumulative_new_kcs"]
        recruitment_rate = novelty_data["kc_recruitment_rate"]
        metrics = list(novelty_data["novelty_per_frame"].keys())

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["frame_id", "cumulative_new_kcs", "kc_recruitment_rate"]
            for m in metrics:
                header += [f"novelty_per_frame_{m}", f"novelty_cumulative_{m}"]
            writer.writerow(header)
            for idx, frame_id in enumerate(frames):
                row = [frame_id, cumulative_kcs[idx], 0 if idx == 0 else recruitment_rate[idx - 1]]
                for m in metrics:
                    row += [novelty_data["novelty_per_frame"][m][idx],
                            novelty_data["novelty_cumulative"][m][idx]]
                writer.writerow(row)

    def _export_novelty_summary(self, output_dir: Path, novelty_data: Dict) -> None:
        summary = {
            "num_frames": novelty_data["num_frames"],
            "total_unique_kcs": novelty_data["total_unique_kcs"],
            "avg_new_kcs_per_frame": novelty_data["avg_new_kcs_per_frame"],
            "final_cumulative_novelty": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in novelty_data["final_cumulative_novelty"].items()
            },
            "correlation_cumulative": {
                k: float(v) if isinstance(v, float) else v
                for k, v in novelty_data["correlation_cumulative"].items()
            },
            "correlation_per_frame": {
                k: float(v) if isinstance(v, float) else v
                for k, v in novelty_data["correlation_per_frame"].items()
            },
        }
        with open(output_dir / "novelty_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # ── Plotting ─────────────────────────────────────────────────────────────

    def plot_raster(self, population: str = "kc",
                    time_range: Optional[Tuple[float, float]] = None,
                    output_path=None, ax=None,
                    color: str = "blue", marker_size: float = 1.0,
                    alpha: float = 0.7, ylabel: str = "Neuron ID"):
        spike_data = self.get_spikes(population, time_range)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(spike_data["times"], spike_data["ids"], s=marker_size, color=color, alpha=alpha)
        ax.set_xlabel("Time [ms]", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{population.upper()} Spikes", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim((self._start_step * self._dt, self._end_step * self._dt))
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
        return ax

    def plot_voltage_traces(self, population: str = "mbon",
                            neuron_ids=None, time_range=None, output_path=None,
                            ax=None, color: str = "blue", alpha: float = 0.7):
        v_data = self.get_voltages(population, time_range, neuron_ids)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(min(v_data["voltages"].shape[0], 5)):
            ax.plot(v_data["time_axis"], v_data["voltages"][i, :], color=color, alpha=alpha,
                    label=f"Neuron {i}")
        ax.set_xlabel("Time [ms]", fontsize=12)
        ax.set_ylabel("Voltage [mV]", fontsize=12)
        ax.set_title(f"{population.upper()} Membrane Potential", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
        return ax

    def plot_currents(self, synapse: str = "kc_mbon",
                      neuron_ids=None, time_range=None, output_path=None,
                      ax=None, color: str = "red", alpha: float = 0.7):
        c_data = self.get_currents(synapse, time_range, neuron_ids)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(min(c_data["currents"].shape[0], 5)):
            ax.plot(c_data["time_axis"], c_data["currents"][i, :], color=color, alpha=alpha,
                    label=f"Neuron {i}")
        ax.set_xlabel("Time [ms]", fontsize=12)
        ax.set_ylabel("Current [a.u.]", fontsize=12)
        ax.set_title(f"{synapse.upper()} Post-Synaptic Current", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
        return ax

    def plot_cumulative_spike_count(self, population: str = "kc", output_path=None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        data = self.get_spikes(population)
        t_start = self._start_step * self._dt
        t_end = self._end_step * self._dt
        times_full = np.arange(t_start, t_end, self._dt)
        if len(data["times"]) > 0:
            cumulative_full = np.interp(times_full, data["times"], data["cumulative_spike_count"])
        else:
            cumulative_full = np.zeros_like(times_full)
        ax.plot(times_full, cumulative_full)
        ax.set_ylabel("Cumulative Spike Count", fontsize=12)
        ax.set_title(f"{population.upper()} - Cumulative Spike Count")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
        return fig

    def plot_activity_summary(self, output_path: Union[str, Path],
                              frame_number: Optional[int] = None) -> None:
        output_path = Path(output_path)
        if frame_number is not None:
            output_path = output_path / f"frame_{frame_number}"
        output_path.mkdir(parents=True, exist_ok=True)

        t_start, t_end = 0, self._simulation_steps * self._dt
        time_axis = np.arange(t_start, t_end, self._dt)

        total_plots = 5
        if hasattr(self._network, "mbon"):
            total_plots += self._network.NMBON

        fig, axes = plt.subplots(total_plots, figsize=(12, 2.5 * total_plots), sharex=True)

        self._plot_neuron_spikes(axes[0], self.get_spikes("pn")["times"],
                                 self.get_spikes("pn")["ids"],
                                 title="Projection Neurons (PN) Activity", color="blue", ylabel="PN ID")
        self._save_individual_plot(axes[0], output_path / "pn_spike_activity.png")

        self._plot_neuron_spikes(axes[1], self.get_spikes("kc")["times"],
                                 self.get_spikes("kc")["ids"],
                                 title="Kenyon Cells (KC) Activity", color="green",
                                 marker_size=2, ylabel="KC ID")
        self._save_individual_plot(axes[1], output_path / "kc_spike_activity.png")

        kc_spikes = self.get_spikes("kc")
        if len(kc_spikes["times"]) > 0:
            cumulative_full = np.interp(time_axis, kc_spikes["times"], kc_spikes["cumulative_spike_count"])
        else:
            cumulative_full = np.zeros_like(time_axis)
        axes[2].plot(time_axis, cumulative_full, color="purple")
        axes[2].set_ylabel("Cumulative Spike Count", fontsize=12)
        axes[2].set_title("KC Cumulative Spike Count Over Time", fontsize=14)
        axes[2].grid(True, linestyle="--", alpha=0.5)
        self._save_individual_plot(axes[2], output_path / "kc_active_count.png")

        apln_v = self.get_voltages("apln")["voltages"]
        if apln_v.size > 0:
            axes[3].plot(time_axis, apln_v[0, :], color="orange")
            axes[3].set_ylabel("Voltage [mV]", fontsize=12)
            axes[3].set_title("APL Neuron Voltage", fontsize=14)
            axes[3].grid(True, linestyle="--", alpha=0.5)
            self._save_individual_plot(axes[3], output_path / "apln_voltage_trace.png")

        c_data = self.get_currents("kc_mbon")
        if c_data["currents"].size > 0:
            for i in range(min(c_data["currents"].shape[0], 5)):
                axes[4].plot(time_axis, c_data["currents"][i, :], color="red", alpha=0.7)
            axes[4].set_ylabel("Current [a.u.]", fontsize=12)
            axes[4].set_title("KC→MBON Post-Synaptic Current", fontsize=14)
            axes[4].grid(True, linestyle="--", alpha=0.5)
        self._save_individual_plot(axes[4], output_path / "kc_mbon_currents.png")

        if hasattr(self._network, "mbon"):
            mbon_v = self.get_voltages("mbon")["voltages"]
            for i in range(self._network.NMBON):
                if i + 5 < len(axes) and mbon_v.size > 0:
                    axes[5 + i].plot(time_axis, mbon_v[i, :], color="red")
                    axes[5 + i].set_ylabel("Voltage [mV]", fontsize=12)
                    axes[5 + i].set_title(f"MBON {i} Voltage", fontsize=14)
                    axes[5 + i].grid(True, linestyle="--", alpha=0.5)
                    self._save_individual_plot(axes[5 + i], output_path / f"mbon_{i}_voltage_trace.png")

        plt.tight_layout()
        plt.savefig(output_path / "combined_activity_plot.png", dpi=300)
        plt.close(fig)

    def _plot_neuron_spikes(self, ax, spike_times, spike_ids, title="Spikes",
                            color="blue", marker_size=1, alpha=0.7, ylabel="Neuron ID"):
        ax.scatter(spike_times, spike_ids, s=marker_size, color=color, alpha=alpha)
        ax.set_xlabel("Time [ms]", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)

    def _save_individual_plot(self, ax, file_path):
        fig = plt.figure(figsize=(10, 6))
        new_ax = fig.add_subplot(111)
        for line in ax.lines:
            new_ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                        linestyle=line.get_linestyle(), linewidth=line.get_linewidth(),
                        label=line.get_label())
        for coll in ax.collections:
            offsets = coll.get_offsets()
            if len(offsets) > 0:
                new_ax.scatter(offsets[:, 0], offsets[:, 1],
                               c=coll.get_facecolor(), s=coll.get_sizes(), alpha=coll.get_alpha())
        new_ax.set_title(ax.get_title(), fontsize=14)
        new_ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        new_ax.set_ylabel(ax.get_ylabel(), fontsize=12)
        if ax.get_legend():
            new_ax.legend(loc="upper right")
        new_ax.grid(True, linestyle="--", alpha=0.5)
        new_ax.set_xlim((self._start_step * self._dt, self._end_step * self._dt))
        fig.tight_layout()
        fig.savefig(file_path, dpi=300)
        plt.close(fig)

    def plot_cumulative_novelty(self, output_path: Union[str, Path], metric: str = "cosine") -> None:
        novelty_data = self.get_novelties()
        if not novelty_data.get("enabled", False):
            return
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        frames = novelty_data["frames"]
        cum_novelty = novelty_data["novelty_cumulative"][metric].astype(float)
        cum_kcs = np.asarray(novelty_data["cumulative_new_kcs"], dtype=float)
        corr_val = novelty_data["correlation_cumulative"][metric]
        corr_str = f"{corr_val:.3f}" if isinstance(corr_val, float) else corr_val

        fig, ax1 = plt.subplots(figsize=(8, 6))
        sc1 = ax1.scatter(frames, cum_novelty, marker="o", s=4, color="blue", alpha=0.8,
                          label=f"Cumulative Novelty ({metric})")
        ax1.set_title(f"Cumulative {metric.capitalize()} Novelty vs. Newly Fired KCs", fontsize=14)
        ax1.set_xlabel("Frame ID", fontsize=12)
        ax1.set_ylabel("Cumulative Novelty", fontsize=12, color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        sc2 = ax2.scatter(frames, cum_kcs, marker="o", s=4, color="red", alpha=0.8,
                          label=f"Cumulative New KCs (corr={corr_str})")
        ax2.set_ylabel("Cumulative Newly Fired KCs", fontsize=12, color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax1.legend(handles=[sc1, sc2], loc="best", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path / f"combined_cumulative_novelty_{metric}.png", dpi=300)
        plt.close(fig)

    def plot_instant_novelty(self, output_path: Union[str, Path], metric: str = "cosine") -> None:
        novelty_data = self.get_novelties()
        if not novelty_data.get("enabled", False):
            return
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        frames = novelty_data["frames"]
        novelty_vals = novelty_data["novelty_per_frame"][metric].astype(float)
        recruitment_rate = novelty_data["kc_recruitment_rate"]
        if len(recruitment_rate) == 0:
            return
        frames_for_rate = frames[1:]
        corr_val = novelty_data["correlation_per_frame"][metric]
        corr_str = f"{corr_val:.3f}" if isinstance(corr_val, float) else corr_val

        fig, ax1 = plt.subplots(figsize=(10, 4))
        (line1,) = ax1.plot(frames, novelty_vals, color="green", alpha=0.7, label=f"Novelty ({metric})")
        ax1.set_xlabel("Frame ID", fontsize=12)
        ax1.set_ylabel("Novelty Score", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        (line2,) = ax2.plot(frames_for_rate, recruitment_rate, color="purple", alpha=0.7,
                             label=f"KC recruitment rate (corr={corr_str})")
        ax2.set_ylabel("New KCs Recruited", fontsize=12)
        ax1.set_title(f"Per-Frame {metric.capitalize()} Novelty + KC Recruitment Rate", fontsize=14)
        ax1.legend(handles=[line1, line2], loc="best", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path / f"novelty_and_kc_recruitment_{metric}.png", dpi=300)
        plt.close(fig)

    def plot_all_novelty(self, output_path: Union[str, Path]) -> None:
        novelty_data = self.get_novelties()
        if not novelty_data.get("enabled", False):
            return
        for metric in novelty_data["novelty_cumulative"]:
            self.plot_cumulative_novelty(output_path, metric)
            self.plot_instant_novelty(output_path, metric)
