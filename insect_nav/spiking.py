"""
NeuralNetwork: GeNN-based spiking mushroom body model.

Requires: pip install insect_nav[genn]
"""

import csv
import multiprocessing as mp
import os
import shutil
import time

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")

try:
    from pygenn import (
        GeNNModel,
        create_var_ref,
        create_out_post_var_ref,
        create_wu_pre_var_ref,
        create_wu_var_ref,
        init_postsynaptic,
        init_sparse_connectivity,
        init_weight_update,
    )
except ImportError as _err:
    raise ImportError(
        "NeuralNetwork requires pygenn. Install with: pip install insect_nav[genn]"
    ) from _err

from insect_nav.base import NeuralModelBase
from insect_nav.genn_models import (
    anti_hebbian,
    cs_model,
    if_model,
    reset_model_if,
    reset_model_lif,
    reset_model_syn,
    reset_model_wu_var,
)
from insect_nav.logger import NetworkLogger
from insect_nav.parameters import save_parameters_to_file
from insect_nav.vision import countFrames, extractFeatures, loadFrame, preprocessFrame


class NeuralNetwork(NeuralModelBase):
    """
    Spiking neural network inspired by the insect mushroom body.

    Architecture: PN → KC → MBON with APL inhibitory feedback.
    Uses GeNN for GPU-accelerated (or CPU) simulation.
    """

    def __init__(self, parameters, load_net=None, reducedNetwork=False,
                 num_shifts=None, tuneCurrent=False, connectivity_seed=-1, use_gpu=False,
                 batch_size=1, precompute_features=False):
        if load_net is None:
            load_net = {"pn_kc": False, "kc_mbon": False}

        if batch_size > 1 and not use_gpu:
            # GeNN's single_threaded_cpu backend hard-rejects batch_size>1 at
            # model.load() time ("only supports simulations with a batch
            # size of 1") -- fail fast with a clear message instead of
            # letting that generic RuntimeError surface after a build.
            raise ValueError("batch_size > 1 requires use_gpu=True (GeNN's single_threaded_cpu backend only supports batch_size=1)")

        self.NMBON = 1
        self.reducedNetwork = reducedNetwork
        self.connectivity_seed = connectivity_seed
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.precompute_features = precompute_features
        self._feature_cache = None
        self._cached_shift_degrees = None

        # Novelty output mode (opt-in via environment, read here so the
        # downstream project needs no code change):
        #   "spikes"         (default) -> per-shift novelty = MBON spike count
        #                                  (small discrete integers, original
        #                                  behaviour)
        #   "spikes_voltage" -> novelty = spike_count + V_norm, where V_norm is
        #                       the MBON's final membrane voltage normalized to
        #                       [0,1] between rest (Vrest/Vreset) and threshold
        #                       (Vthresh). This makes novelty CONTINUOUS, so it
        #                       should be paired with a continuous
        #                       INSECT_NAV_DEGREE_STRATEGY (e.g. strategy3).
        self.novelty_mode = os.environ.get("INSECT_NAV_NOVELTY_MODE", "spikes").strip() or "spikes"
        if self.novelty_mode not in ("spikes", "spikes_voltage"):
            raise ValueError(
                f"INSECT_NAV_NOVELTY_MODE must be 'spikes' or 'spikes_voltage', got {self.novelty_mode!r}"
            )

        self.loaded_weights = {
            "pn_kc": None,
            "kc_mbon": {i: parameters["KC_MBON_WEIGHT"] for i in range(self.NMBON)},
        }

        super().__init__(parameters, load_net, num_shifts)

        self.logger = NetworkLogger(self, config={
            "voltages": {"pn": False, "kc": False, "apln": False, "mbon": False},
            "currents": {"pn_kc": False, "kc_apln": False, "apln_kc": False, "kc_mbon": False},
            "spikes": {"pn": False, "kc": True, "apln": False, "mbon": True},
            "weights": {"kc_mbon": False},
        })

        self._build_network()
        if tuneCurrent:
            self.tuneInputCurrent()
        if self.precompute_features:
            self._build_feature_cache()

    # ── Weight persistence ───────────────────────────────────────────────────

    def save_weights(self):
        self.pn_kc.pull_connectivity_from_device()
        os.makedirs(self.params["weightsPath"], exist_ok=True)

        np.save(
            os.path.join(self.params["weightsPath"], "pn_kc_ind.npy"),
            np.vstack((self.pn_kc.get_sparse_pre_inds(), self.pn_kc.get_sparse_post_inds())),
        )
        print("PN→KC weights saved.")

        if not self.reducedNetwork:
            self.kc_mbon.vars["g"].pull_from_device()
            np.save(
                os.path.join(self.params["weightsPath"], "kc_mbon_g_0.npy"),
                self.kc_mbon.vars["g"].values,
            )
            print("KC→MBON weights saved.")

    def analyze_kc_mbon_connectivity(self) -> dict:
        """
        Inspect the KC→MBON weights currently on the device and determine
        whether they still reflect the untrained uniform initialization
        (KC_MBON_WEIGHT for every synapse) or have been modified by training.

        Call after the network has been built (e.g. right after __init__, or
        after load_weights()/_build_network()) — kc_mbon_g_*.npy is written
        with the uniform initial value both by a fresh (never-trained) build
        and by create_variants' random-connectivity-seed regeneration, so its
        mere existence on disk does not imply training actually happened;
        pulling the live weights and comparing them against KC_MBON_WEIGHT
        does.

        Returns a dict with min/max/mean/std of the weights, the fraction of
        synapses still exactly at the initial value, and a summary
        "is_trained" bool (True iff at least one weight differs from the
        initial value).
        """
        self.kc_mbon.vars["g"].pull_from_device()
        weights = self.kc_mbon.vars["g"].values
        initial_weight = self.params["KC_MBON_WEIGHT"]

        fraction_unchanged = float(np.mean(np.isclose(weights, initial_weight)))

        return {
            "min": float(weights.min()),
            "max": float(weights.max()),
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "initial_weight": float(initial_weight),
            "fraction_unchanged": fraction_unchanged,
            "is_trained": fraction_unchanged < 1.0,
        }

    def _generate_pn_kc_connectivity(self, num_pn: int, num_kc: int, fan_in: int):
        """
        Deterministic PN→KC connectivity using numpy RNG (seed-based).

        Bypasses GeNN's backend-specific RNGs (Philox/xorwow) so connectivity
        is identical across GPU and CPU runs when connectivity_seed is fixed.
        """
        np.random.seed(self.connectivity_seed)
        pre_inds, post_inds = [], []
        for kc_idx in range(num_kc):
            selected = np.random.choice(num_pn, size=fan_in, replace=True)
            for pn_idx in selected:
                pre_inds.append(pn_idx)
                post_inds.append(kc_idx)
        return (
            np.array(pre_inds, dtype=np.int64),
            np.array(post_inds, dtype=np.uint32),
        )

    def load_weights(self):
        weights_dir = self.params["weightsPath"]

        if self.load_net["pn_kc"]:
            path = os.path.join(weights_dir, "pn_kc_ind.npy")
            try:
                self.loaded_weights["pn_kc"] = np.load(path)
                print(f"PN→KC weights loaded from: {path}")
            except FileNotFoundError as e:
                print(f"Error loading PN→KC weights: {e}")
                raise
        elif self.connectivity_seed >= 0:
            pre_inds, post_inds = self._generate_pn_kc_connectivity(
                self.input_neurons, self.params["NUM_KC"], self.params["PN_KC_FAN_IN"]
            )
            self.loaded_weights["pn_kc"] = np.vstack((pre_inds, post_inds))
            print(f"PN→KC connectivity generated with seed={self.connectivity_seed}")

        if self.load_net["kc_mbon"]:
            for i in range(self.NMBON):
                path = os.path.join(weights_dir, f"kc_mbon_g_{i}.npy")
                try:
                    self.loaded_weights["kc_mbon"][i] = np.load(path)
                except FileNotFoundError as e:
                    print(f"Error loading KC→MBON weights: {e}")
                    raise
            print("KC→MBON weights loaded.")

    # ── Network construction ─────────────────────────────────────────────────

    def _build_network(self):
        start = time.time()
        backend = "cuda" if self.use_gpu else "single_threaded_cpu"
        self.model = GeNNModel("float", self.params["name"], backend=backend)
        self.model.dt = self.params["DT"]
        self.model.seed = 1
        # Must be set before any add_neuron_population/build() call. Verified
        # empirically (see plan) that batch_size=1 is bit-identical to never
        # setting it at all, on both single_threaded_cpu and cuda backends.
        self.model.batch_size = self.batch_size

        self.pn = self.model.add_neuron_population(
            "pn", self.input_neurons, "LIF",
            self.params["LIF_PARAMS"], self.params["LIF_INIT"],
        )
        self.kc = self.model.add_neuron_population(
            "kc", self.params["NUM_KC"], "LIF",
            self.params["LIF_PARAMS"], self.params["LIF_INIT"],
        )
        self.apln = self.model.add_neuron_population(
            "apln", 1, if_model,
            self.params["IF_PARAMS"], self.params["IF_INIT"],
        )

        self.neuron_populations = [self.pn, self.kc, self.apln]
        self.neuron_populations_names = ["pn", "kc", "apln"]
        self.neuron_initial_states = [
            self.params["LIF_INIT"], self.params["LIF_INIT"], self.params["IF_INIT"],
        ]

        if not self.reducedNetwork:
            self.mbon = self.model.add_neuron_population(
                "mbon", self.NMBON, "LIF",
                self.params["LIF_PARAMS"], self.params["LIF_INIT"],
            )
            self.neuron_initial_states.append(self.params["LIF_INIT"])
            self.neuron_populations.append(self.mbon)
            self.neuron_populations_names.append("mbon")
        else:
            self.logger.update_config("voltages", {"mbon": False})
            self.logger.update_config("spikes", {"mbon": False})
            self.logger.update_config("currents", {"kc_mbon": False})
            self.logger.update_config("weights", {"kc_mbon": False})

        for name, enabled in self.logger._config["spikes"].items():
            if enabled and hasattr(self, name):
                getattr(self, name).spike_recording_enabled = True

        self.pn_input = self.model.add_current_source("pn_input", cs_model, self.pn, {}, {"magnitude": 0.0})

        if self.load_net["pn_kc"] or self.connectivity_seed >= 0:
            self.pn_kc = self.model.add_synapse_population(
                "pn_kc", "SPARSE", self.pn, self.kc,
                init_weight_update("StaticPulseConstantWeight", {"g": self.params["PN_KC_WEIGHT"]}),
                init_postsynaptic("ExpCurr", {"tau": self.params["PN_KC_TAU"]}),
            )
            self.pn_kc.set_sparse_connections(
                self.loaded_weights["pn_kc"][0],
                self.loaded_weights["pn_kc"][1],
            )
        else:
            self.pn_kc = self.model.add_synapse_population(
                "pn_kc", "SPARSE", self.pn, self.kc,
                init_weight_update("StaticPulseConstantWeight", {"g": self.params["PN_KC_WEIGHT"]}),
                init_postsynaptic("ExpCurr", {"tau": self.params["PN_KC_TAU"]}),
                init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": self.params["PN_KC_FAN_IN"]}),
            )
            print("Random connectivity generated by GeNN.")

        self.kc_apln = self.model.add_synapse_population(
            "kc_apln", "DENSE", self.kc, self.apln,
            init_weight_update("StaticPulse", vars={"g": self.params["KC_APLN_WEIGHT"]}),
            init_postsynaptic("DeltaCurr"),
        )
        self.apln_kc = self.model.add_synapse_population(
            "apln_kc", "DENSE", self.apln, self.kc,
            init_weight_update("StaticPulse", vars={"g": self.params["APLN_KC_WEIGHT"]}),
            init_postsynaptic("ExpCurr", {"tau": self.params["APLN_KC_TAU"]}),
        )

        self.synapse_populations = [self.pn_kc, self.kc_apln, self.apln_kc]
        self.synapse_populations_names = ["pn_kc", "kc_apln", "apln_kc"]

        if not self.reducedNetwork:
            # Backward compat: parameters.json salvati prima dell'introduzione
            # di halve_g hanno KC_MBON_PARAMS = {"mod": ...} soltanto -- il
            # modello anti_hebbian ora richiede anche un valore iniziale per
            # halve_g.
            self.params["KC_MBON_PARAMS"].setdefault("halve_g", 0.0)

            kc_mbon_g_init = self.loaded_weights["kc_mbon"][0]
            if self.batch_size > 1 and np.ndim(kc_mbon_g_init) > 0:
                # anti_hebbian's "g" var is READ_WRITE (default access mode
                # for a plain ("g", "scalar") var, see genn_models.py), which
                # GeNN auto-duplicates across the batch dimension as soon as
                # model.batch_size > 1 -- an already-loaded 1D trained-weight
                # array must be tiled to (batch_size, NUM_KC) or GeNN rejects
                # it as under-shaped (a plain scalar initial value needs no
                # tiling: GeNN broadcasts scalars to any shape natively).
                # Every lane starts identical; train() refuses batch_size>1
                # (see below), so they stay identical for the network's
                # lifetime -- there is no plasticity to make them diverge.
                kc_mbon_g_init = np.tile(np.asarray(kc_mbon_g_init), (self.batch_size, 1))
            self.kc_mbon = self.model.add_synapse_population(
                "kc_mbon", "SPARSE", self.kc, self.mbon,
                init_weight_update(anti_hebbian, self.params["KC_MBON_PARAMS"],
                                   {"g": kc_mbon_g_init, "halved": 0.0}),
                init_postsynaptic("ExpCurr", {"tau": self.params["KC_MBON_TAU"]}),
            )
            self.kc_mbon.set_wu_param_dynamic("mod", True)
            self.kc_mbon.set_wu_param_dynamic("halve_g", True)
            self.kc_mbon.set_sparse_connections(
                np.arange(self.params["NUM_KC"]),
                np.zeros(self.params["NUM_KC"]),
            )
            self.synapse_populations.append(self.kc_mbon)
            self.synapse_populations_names.append("kc_mbon")

        os.makedirs("./builds_network", exist_ok=True)
        self._create_reset_custom_update()
        self.model.build("./builds_network", always_rebuild=True)
        self.model.load(num_recording_timesteps=int(round(self.params["PRESENT_TIME_MS"] / self.params["DT"])))
        print(f"\nNetwork built in {time.time() - start:.3f} seconds.")

    def delete_build_directory(self):
        build_dir = os.path.join("builds_network", f"{self.params['name']}_CODE")
        shutil.rmtree(build_dir)

    # ── Input current tuning ─────────────────────────────────────────────────

    def tuneInputCurrent(self):
        """
        Binary-search tuning of INPUT_SCALE to reach the target KC activation count.
        """
        current_values, mean_history, std_history = [], [], []
        mean_spk_history, std_spk_history = [], []
        lo, hi = 0.0, 1.0
        tolerance = max(1, int(self.params["target_kcs"] * 0.1))

        num_train_frames = countFrames(self.params["trainingDatasetPath"])
        ids = range(0, num_train_frames, self.params.get("train_step", 1))
        sampled_ids = list(ids) if len(ids) <= 10 else [ids[int(i * len(ids) / 10)] for i in range(10)]

        print(f"[Spiking] Tuning PN input current using {len(sampled_ids)} samples...")

        for _ in tqdm(range(15)):
            current = (lo + hi) / 2
            current_values.append(current)
            self.params["INPUT_SCALE"] = current

            kc_fired, kc_spk = [], []
            for frame_number in sampled_ids:
                frame = loadFrame(frame_number, self.params["trainingDatasetPath"])
                self.simulate(preprocessFrame(frame, 0, self.params), self.params["PRESENT_TIME_MS"], False)
                spikes = self.logger.get_spikes("kc")
                kc_fired.append(spikes["neurons_fired"])
                kc_spk.append(spikes["count"])

            mean_fired = float(np.mean(kc_fired))
            mean_history.append(mean_fired)
            std_history.append(float(np.std(kc_fired)))
            mean_spk_history.append(float(np.mean(kc_spk)))
            std_spk_history.append(float(np.std(kc_spk)))

            if mean_fired > self.params["target_kcs"]:
                hi = current
            else:
                lo = current

            if abs(self.params["target_kcs"] - mean_fired) <= tolerance:
                break

        best_idx = int(np.argmin(np.abs(np.array(mean_history) - self.params["target_kcs"])))
        self.params["INPUT_SCALE"] = current_values[best_idx]
        self.params["mean_num_kc_fired"] = mean_history[best_idx]
        self.params["std_num_kc_fired"] = std_history[best_idx]
        self.params["mean_num_spikes_kc"] = mean_spk_history[best_idx]
        self.params["std_num_spikes_kc"] = std_spk_history[best_idx]
        save_parameters_to_file(self.params, self.params["parameters_path"])
        print(f"Optimal input scale: {current_values[best_idx]:.4f}, mean active KCs: {mean_history[best_idx]:.2f}")

    # ── Reset ────────────────────────────────────────────────────────────────

    def _create_reset_custom_update(self):
        for pop_name, pop in zip(self.neuron_populations_names, self.neuron_populations):
            if pop_name == "apln":
                self.model.add_custom_update(
                    f"reset_{pop_name}", "RESET", reset_model_if, {}, {},
                    {"V": create_var_ref(pop, "V")}, {},
                )
            else:
                self.model.add_custom_update(
                    f"reset_{pop_name}", "RESET", reset_model_lif, {}, {},
                    {"V": create_var_ref(pop, "V"), "RefracTime": create_var_ref(pop, "RefracTime")}, {},
                )

        for syn_name, syn in zip(self.synapse_populations_names, self.synapse_populations):
            self.model.add_custom_update(
                f"reset_{syn_name}", "RESET", reset_model_syn, {}, {},
                {"Isyn": create_out_post_var_ref(syn)}, {},
            )

        if not self.reducedNetwork:
            # anti_hebbian's "halved" guard (see genn_models.py) caps each
            # synapse to at most one halving per FRAME, not per presentation:
            # deliberately its own custom-update group ("RESET_HALVED"),
            # separate from "RESET" (which reset_network() runs before every
            # single presentation) -- otherwise a synapse touched by both the
            # +step and -step flank presentations of the same frame (see
            # train()) would be halved twice (quarter, not half). Cleared
            # once per frame by _reset_halve_guard(), called from train()
            # before its first (center) presentation.
            self.model.add_custom_update(
                "reset_kc_mbon_halved", "RESET_HALVED", reset_model_wu_var, {}, {},
                {"var": create_wu_var_ref(self.kc_mbon, "halved")}, {},
            )

    def reset_network(self):
        self.model.timestep = 0
        self.model.custom_update("RESET")

    def _reset_halve_guard(self):
        """Clear anti_hebbian's per-synapse "halved" guard once per frame (see
        _create_reset_custom_update) -- called at the start of train(), not
        per presentation, so it protects across a frame's center + flank
        presentations without being touched by test()'s simulate() calls."""
        if not self.reducedNetwork:
            self.model.custom_update("RESET_HALVED")

    # ── Simulation ───────────────────────────────────────────────────────────

    def _features_from(self, item):
        """
        item is either an already-preprocessed image (2D array, needs
        extractFeatures), an already-extracted feature vector (1D array,
        e.g. from the precompute_features cache -- used as-is), or None
        (zero input).
        """
        if item is None:
            return 0
        arr = np.asarray(item)
        return arr if arr.ndim == 1 else extractFeatures(arr, self.params)

    def simulate(self, preprocessed_frame, present_time_ms: float, plasticity: bool = False, halve: bool = False):
        """
        Run the network for one stimulus presentation (batch_size == 1), or
        for up to batch_size presentations at once (batch_size > 1).

        Args:
            preprocessed_frame: Already-preprocessed image (from
                preprocessFrame), a feature vector, None to disable PN input,
                or -- only when batch_size > 1 -- a list of up to batch_size
                such items (one per batch lane; missing lanes are zeroed).
            present_time_ms: Stimulus duration in milliseconds.
            plasticity: Whether to enable KC→MBON weight modification.
                Requires batch_size == 1 (see train()) -- plasticity across
                independently-batched lanes has no defined semantics here.
            halve: When plasticity=True, a coincident KC-MBON spike halves
                the synapse's weight (g *= 0.5) instead of zeroing it
                outright. Ignored when plasticity=False.
        """
        if plasticity and self.batch_size != 1:
            raise ValueError("plasticity/training requires batch_size == 1")

        self.kc_mbon.set_dynamic_param_value("mod", 1 if plasticity else -1)
        self.kc_mbon.set_dynamic_param_value("halve_g", 1 if (plasticity and halve) else 0)

        magnitude = self.pn_input.vars["magnitude"]
        if self.batch_size == 1:
            features = self._features_from(preprocessed_frame)
            magnitude.view[:] = features * self.params["INPUT_SCALE"]
            magnitude.push_to_device()
        else:
            items = preprocessed_frame if isinstance(preprocessed_frame, (list, tuple)) else [preprocessed_frame]
            if len(items) > self.batch_size:
                raise ValueError(f"got {len(items)} inputs, exceeds batch_size={self.batch_size}")
            magnitude.view[:, :] = 0.0
            for lane, item in enumerate(items):
                magnitude.view[lane, :] = self._features_from(item) * self.params["INPUT_SCALE"]
            magnitude.push_to_device()

        self.reset_network()
        simulation_steps = int(round(present_time_ms / self.params["DT"]))
        if self.batch_size == 1:
            self.logger.start_logging(simulation_steps)
            for _ in range(simulation_steps):
                self.logger.log_step(self.model.timestep)
                self.model.step_time()
            self.logger.finalize_logging()
        else:
            # NetworkLogger assumes unbatched spike_recording_data[0] shape
            # and would silently read only lane 0 -- bypass it entirely here;
            # test() reads every lane's spikes directly once step_time() is
            # done (rich per-timestep logging isn't supported in batch mode).
            for _ in range(simulation_steps):
                self.model.step_time()
            self.model.pull_recording_buffers_from_device()

    # ── Feature cache (precompute_features=True) ────────────────────────────

    def _build_feature_cache(self):
        """
        Precompute PN feature vectors (preprocessFrame + extractFeatures) for
        every frame in trainingDatasetPath x the fixed (num_shifts+1) angular
        shift grid (see NeuralModelBase._shift_degrees), once, so test()/
        train() can skip that work at call time by passing frame_id.
        """
        num_frames = countFrames(self.params["trainingDatasetPath"])
        self._cached_shift_degrees = self._shift_degrees()
        self._feature_cache = np.zeros(
            (num_frames, len(self._cached_shift_degrees), self.input_neurons), dtype=np.float32,
        )
        for frame_id in range(num_frames):
            frame = loadFrame(frame_id, frames_dir=self.params["trainingDatasetPath"])
            for shift_idx, shift in enumerate(self._cached_shift_degrees):
                preprocessed = preprocessFrame(frame, shift, self.params)
                self._feature_cache[frame_id, shift_idx, :] = extractFeatures(preprocessed, self.params)

    def _lookup_cached_features(self, frame_id, shift_degree):
        if self._feature_cache is None or frame_id is None:
            return None
        try:
            shift_idx = self._cached_shift_degrees.index(shift_degree)
        except ValueError:
            return None
        if not (0 <= frame_id < self._feature_cache.shape[0]):
            return None
        return self._feature_cache[frame_id, shift_idx, :]

    # ── Train / Test ─────────────────────────────────────────────────────────

    def _mbon_voltage_norm(self):
        """Final MBON membrane voltage (after the last simulated timestep),
        normalized to [0, 1] between rest and threshold:

            V_norm = clip((V_final - V_rest) / (V_thresh - V_rest), 0, 1)

        V_rest is the built-in LIF's decay target ``Vrest`` (falls back to
        ``Vreset``), V_thresh is ``Vthresh`` -- both from LIF_PARAMS. The value
        is pulled straight off the device (cheaper than per-timestep voltage
        logging); reset_network() runs at the START of each simulate(), so there
        is no cross-shift leakage. Returns a 1-D array with one value per batch
        lane (lane 0 first). NMBON == 1, so reshape(-1) yields exactly one entry
        per lane.
        """
        self.mbon.vars["V"].pull_from_device()
        v = np.asarray(self.mbon.vars["V"].values, dtype=np.float64).reshape(-1)
        lif = self.params["LIF_PARAMS"]
        v_rest = lif.get("Vrest", lif.get("Vreset", 0.0))
        v_thresh = lif["Vthresh"]
        denom = (v_thresh - v_rest) or 1.0
        return np.clip((v - v_rest) / denom, 0.0, 1.0)

    def train(self, frame, frame_id=None):
        """
        Classic procedure (default, self.params["halve"] assente/False): un
        solo allenamento sul frame stesso, shift 0, coincidenza KC-MBON
        azzera il peso (halve=False).

        Procedura estesa (self.params["halve"] = True nel parameters.json):
        oltre al frame stesso (come sopra, azzerato), allena anche le due
        presentazioni ottenute shiftando il frame di +/-DEGREES_PER_SHIFT
        (stesso meccanismo di shift usato da testNavigation/_shift_degrees),
        dove la coincidenza KC-MBON dimezza il peso invece di azzerarlo --
        i due heading vicini vengono trattati come "meno familiari" invece
        che pienamente familiari. Il guard "halved" (vedi genn_models.py) e'
        per FRAME, non per presentazione: una sinapsi toccata da entrambe le
        presentazioni +/-step si dimezza una volta sola (non un quarto).
        """
        if self.batch_size != 1:
            raise ValueError("train() requires batch_size == 1 (plasticity is not batched)")
        self._reset_halve_guard()
        preprocessed = preprocessFrame(frame, 0, self.params)
        self.simulate(preprocessed, self.params["PRESENT_TIME_MS"], plasticity=True, halve=False)
        if self.logger._novelty_data["enabled"]:
            self.logger.log_training_frame(frame_id, frame, preprocessed)

        if self.params.get("halve"):
            step = self.params["DEGREES_PER_SHIFT"]
            for shift in (step, -step):
                shifted = preprocessFrame(frame, shift, self.params)
                self.simulate(shifted, self.params["PRESENT_TIME_MS"], plasticity=True, halve=True)

    def test(self, frame, shift_degree=0, frame_id=None):
        """
        Test the network on one stimulus, or -- when batch_size > 1 and
        `frame` is a list -- on up to batch_size stimuli at once (same
        shift_degree for every lane, matching this project's validated
        frame-batching strategy).

        Args:
            frame: a single raw frame, a list of up to batch_size raw frames,
                or None (per-item) when precompute_features=True and the
                matching frame_id is already cached -- callers relying on the
                cache can skip loading/decoding the image entirely (this is
                the whole point of precompute_features: avoid paying disk +
                cv2 preprocessing cost again at test time).
            shift_degree: angular shift applied to every frame in this call.
            frame_id: optional frame index (or list, parallel to `frame`)
                used to look up precomputed features when
                precompute_features=True was passed to __init__; ignored
                (falls back to on-the-fly preprocessing, which requires a
                real `frame`) otherwise or on a cache miss.

        Returns:
            A single MBON spike count (int) if `frame` is a single frame, or
            a list of counts (one per input) if `frame` is a list.
        """
        is_batch_call = isinstance(frame, (list, tuple))
        frames_in = list(frame) if is_batch_call else [frame]
        if is_batch_call and len(frames_in) > self.batch_size:
            raise ValueError(f"got {len(frames_in)} frames, exceeds batch_size={self.batch_size}")
        frame_ids_in = list(frame_id) if isinstance(frame_id, (list, tuple)) else [frame_id] * len(frames_in)
        if len(frame_ids_in) != len(frames_in):
            raise ValueError("frame_id list must match frame list length")

        inputs = []
        for f, fid in zip(frames_in, frame_ids_in):
            cached = self._lookup_cached_features(fid, shift_degree) if self.precompute_features else None
            if cached is not None:
                inputs.append(cached)
            elif f is not None:
                inputs.append(preprocessFrame(f, shift_degree, self.params))
            else:
                raise ValueError(
                    f"frame is None and no cached features for frame_id={fid} at shift={shift_degree} "
                    "(precompute_features cache miss) -- pass a real frame"
                )

        self.simulate(inputs if is_batch_call else inputs[0], self.params["PRESENT_TIME_MS"], False)

        if self.batch_size == 1:
            count = self.logger.get_spikes("mbon")["count"]
            if self.novelty_mode == "spikes_voltage" and hasattr(self, "mbon"):
                count = count + float(self._mbon_voltage_norm()[0])
            return count

        mbon_rec = self.mbon.spike_recording_data
        counts = [len(mbon_rec[lane][0]) for lane in range(len(frames_in))]
        if self.novelty_mode == "spikes_voltage":
            vnorm = self._mbon_voltage_norm()
            counts = [c + float(vnorm[lane]) for lane, c in enumerate(counts)]
        return counts if is_batch_call else counts[0]

    # ── CSV export ───────────────────────────────────────────────────────────

    def save_features_to_csv(self, frame, frame_number: int, csv_path: str):
        os.makedirs(csv_path, exist_ok=True)
        preprocessed = preprocessFrame(frame, 0, self.params)
        features = extractFeatures(preprocessed, self.params)
        filename = os.path.join(csv_path, "feature_vector.csv")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame_Number"] + [f"Feature_{i}" for i in range(len(features))])
            writer.writerow([frame_number, *features])

    def saveLogsToCsv(self, frame, frame_number: int, output_path: str):
        csv_path = os.path.join(output_path, f"frame_{frame_number}")
        self.save_features_to_csv(frame, frame_number, csv_path)
        self.logger.export_to_csv(csv_path, what="all", include_metadata=True)
