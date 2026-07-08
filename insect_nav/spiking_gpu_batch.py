"""
Native GeNN batch_size variant of insect_nav.spiking.NeuralNetwork.

Does NOT modify insect_nav.spiking or insect_nav.genn_models. Builds the same
PN -> KC -> APL(inhibitory) -> MBON topology directly with pygenn (mirroring
NeuralNetwork._build_network), but with GeNNModel.batch_size set to a value
>1 *before* build()/load(), so a single simulate() call advances `batch_size`
independent lanes at once, each lane driven by a different frame's PN input.

Batching axis (fixed by design, see module docstring in the task that
produced this file): batch over FRAMES, with the angular shift kept as a
sequential outer loop (NUM_SHIFTS+1 iterations, never batched). For each
shift, frame_ids are split into chunks of at most `batch_size`, each chunk
is simulated with exactly one simulate() call (one reset + PRESENT_TIME_MS
of stepping), and KC/MBON spike counts are read back per-lane from GeNN's
batched spike_recording_data.

This is pure inference: the KC->MBON weight update has no working
"plasticity" mode here at all (see kc_mbon_readonly below) -- this module
must never be used for training.

GPU usage note: every GeNNModel built here uses backend="cuda", so its
entire lifetime (construction/build/load, the whole test loop, and the final
model.unload()) MUST run inside insect_nav.benchmark.gpu_exclusive() -- the
single shared 6GB GPU is used by multiple concurrent worktrees/agents.
run_native_batch() below already does this; do not call _BatchNetwork
directly outside of a gpu_exclusive() block.
"""

import os
import time

import numpy as np

from pygenn import (
    GeNNModel,
    VarAccess,
    create_out_post_var_ref,
    create_var_ref,
    create_weight_update_model,
    init_postsynaptic,
    init_weight_update,
)

from insect_nav.benchmark import gpu_exclusive
from insect_nav.genn_models import cs_model, if_model, reset_model_if, reset_model_lif, reset_model_syn
from insect_nav.parameters import load_parameters_from_file
from insect_nav.vision import countFrames, extractFeatures, loadFrame, preprocessFrame

# ── Inference-only KC->MBON weight-update model ─────────────────────────────
#
# insect_nav.genn_models.anti_hebbian declares its "g" var as a plain
# ("g", "scalar") tuple, which pygenn defaults to VarAccess.READ_WRITE.
# Empirically (see get_var_access_dim in pygenn._genn), VarAccess.READ_WRITE
# carries the BATCH dimension, meaning GeNN would duplicate "g" once per
# batch lane -- i.e. batch_size independent copies of the trained weights,
# each separately writable. That's the opposite of what we want: ONE trained
# network, replicated *logically* (read-only) across batch_size test lanes.
#
# Since this module is pure inference (kc_mbon's "mod" is conceptually fixed
# at -1 forever -- no plasticity, ever, matching the task constraint of never
# touching learning), anti_hebbian's own state-update logic is provably inert
# at mod<=0: both `pre_spike_syn_code`'s "if (mod > 0) g = 0;" and the whole
# `post_spike_syn_code` block never execute. So a model that (a) drops the
# dead mod-gated writes entirely and (b) declares "g" as VarAccess.READ_ONLY
# (ELEMENT-only dims, no BATCH dim -> shared across every lane) is exactly
# numerically equivalent to anti_hebbian(mod=-1), while making GeNN keep a
# single shared copy of the weights instead of duplicating them per lane.
kc_mbon_readonly = create_weight_update_model(
    "kc_mbon_readonly",
    vars=[("g", "scalar", VarAccess.READ_ONLY)],
    pre_spike_syn_code="addToPost(g);",
)


def _shift_degrees(params: dict, num_shifts: int) -> list:
    """Identical formula to insect_nav.benchmark._shift_degrees / NeuralModelBase.testNavigation."""
    step = params["DEGREES_PER_SHIFT"]
    shifts = []
    for k in range(num_shifts + 1):
        shift = (-num_shifts / 2 + k) * step
        angle = (shift + 180) % 360 - 180
        shifts.append(angle)
    return shifts


def _find_optimal_degree(params: dict, degree_array: list, novelty_array: list) -> float:
    """Identical logic to insect_nav.benchmark._find_optimal_degree / NeuralModelBase.find_optimal_degree."""
    min_value = min(novelty_array)
    deg_min = [d for d, v in zip(degree_array, novelty_array) if v == min_value]

    step = params["DEGREES_PER_SHIFT"]
    groups, current = [], [deg_min[0]]
    for d in deg_min[1:]:
        if abs(d - current[-1]) == step:
            current.append(d)
        else:
            groups.append(current)
            current = [d]
    groups.append(current)

    max_length = max(len(g) for g in groups)
    longest = [g for g in groups if len(g) == max_length]
    best_group = longest[0] if len(longest) == 1 else min(longest, key=lambda g: abs(np.mean(g)))

    return float(np.mean(best_group))


class _BatchNetwork:
    """
    Builds the PN -> KC -> APL -> MBON network with an explicit
    model.batch_size > 1, mirroring NeuralNetwork._build_network
    (insect_nav/spiking.py) but:
      - always the full (non-reduced) network with a trained kc_mbon,
      - kc_mbon's weight-update model is kc_mbon_readonly (see above)
        instead of anti_hebbian, since this module never trains,
      - pn_kc / kc_mbon connectivity+weights are always loaded from disk
        (weightsPath/pn_kc_ind.npy, weightsPath/kc_mbon_g_0.npy) -- this
        module only ever runs inference against an already-trained net.

    Must be constructed and torn down (via .unload()) entirely inside
    insect_nav.benchmark.gpu_exclusive().
    """

    def __init__(self, params: dict, batch_size: int, build_dir: str):
        self.params = params
        self.batch_size = batch_size

        input_neurons = 0
        if params["USE_VERTICAL_DIST"]:
            input_neurons += params["WIDTH"]
        if params["USE_HORIZONTAL_DIST"]:
            input_neurons += params["HEIGHT"]
        self.input_neurons = input_neurons

        self.model = GeNNModel("float", f"{params['name']}_batch{batch_size}", backend="cuda")
        self.model.dt = params["DT"]
        self.model.seed = 1
        self.model.batch_size = batch_size  # MUST be set before build()/load()

        self.pn = self.model.add_neuron_population(
            "pn", input_neurons, "LIF", params["LIF_PARAMS"], params["LIF_INIT"],
        )
        self.kc = self.model.add_neuron_population(
            "kc", params["NUM_KC"], "LIF", params["LIF_PARAMS"], params["LIF_INIT"],
        )
        self.apln = self.model.add_neuron_population(
            "apln", 1, if_model, params["IF_PARAMS"], params["IF_INIT"],
        )
        self.mbon = self.model.add_neuron_population(
            "mbon", 1, "LIF", params["LIF_PARAMS"], params["LIF_INIT"],
        )

        self.neuron_populations = [self.pn, self.kc, self.apln, self.mbon]
        self.neuron_populations_names = ["pn", "kc", "apln", "mbon"]

        # Same logger spike config as NeuralNetwork's default: kc + mbon spikes recorded.
        self.kc.spike_recording_enabled = True
        self.mbon.spike_recording_enabled = True

        self.pn_input = self.model.add_current_source("pn_input", cs_model, self.pn, {}, {"magnitude": 0.0})

        pn_kc_ind = np.load(os.path.join(params["weightsPath"], "pn_kc_ind.npy"))
        self.pn_kc = self.model.add_synapse_population(
            "pn_kc", "SPARSE", self.pn, self.kc,
            init_weight_update("StaticPulseConstantWeight", {"g": params["PN_KC_WEIGHT"]}),
            init_postsynaptic("ExpCurr", {"tau": params["PN_KC_TAU"]}),
        )
        self.pn_kc.set_sparse_connections(pn_kc_ind[0], pn_kc_ind[1])

        self.kc_apln = self.model.add_synapse_population(
            "kc_apln", "DENSE", self.kc, self.apln,
            init_weight_update("StaticPulse", vars={"g": params["KC_APLN_WEIGHT"]}),
            init_postsynaptic("DeltaCurr"),
        )
        self.apln_kc = self.model.add_synapse_population(
            "apln_kc", "DENSE", self.apln, self.kc,
            init_weight_update("StaticPulse", vars={"g": params["APLN_KC_WEIGHT"]}),
            init_postsynaptic("ExpCurr", {"tau": params["APLN_KC_TAU"]}),
        )

        self.synapse_populations = [self.pn_kc, self.kc_apln, self.apln_kc]
        self.synapse_populations_names = ["pn_kc", "kc_apln", "apln_kc"]

        kc_mbon_g = np.load(os.path.join(params["weightsPath"], "kc_mbon_g_0.npy"))
        self.kc_mbon = self.model.add_synapse_population(
            "kc_mbon", "SPARSE", self.kc, self.mbon,
            init_weight_update(kc_mbon_readonly, {}, {"g": kc_mbon_g}),
            init_postsynaptic("ExpCurr", {"tau": params["KC_MBON_TAU"]}),
        )
        self.kc_mbon.set_sparse_connections(
            np.arange(params["NUM_KC"]),
            np.zeros(params["NUM_KC"]),
        )
        self.synapse_populations.append(self.kc_mbon)
        self.synapse_populations_names.append("kc_mbon")

        self._create_reset_custom_update()

        os.makedirs(build_dir, exist_ok=True)
        self.model.build(build_dir, always_rebuild=True)
        self.model.load(num_recording_timesteps=int(round(params["PRESENT_TIME_MS"] / params["DT"])))

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

    def reset_network(self):
        self.model.timestep = 0
        self.model.custom_update("RESET")

    def unload(self):
        self.model.unload()


def run_native_batch(params_path: str, batch_size: int, frame_ids=None) -> dict:
    """
    Run the native-batch GPU variant over `frame_ids` (default: all frames in
    trainingDatasetPath) x all angular shifts, batching up to `batch_size`
    frames per GeNN simulate() call (shift is a sequential outer loop, never
    batched). Returns the same dict shape as insect_nav.benchmark.run_reference
    so results are directly comparable via compare_kc_spike_counts.

    elapsed_seconds covers the frame x shift test loop only (build/load/unload
    excluded), consistent with run_reference.
    """
    params = load_parameters_from_file(params_path)
    num_shifts = params["NUM_SHIFTS"]
    shift_list = _shift_degrees(params, num_shifts)
    num_cols = len(shift_list)

    if frame_ids is None:
        frame_ids = list(range(countFrames(params["trainingDatasetPath"])))
    else:
        frame_ids = list(frame_ids)
    num_frames = len(frame_ids)

    # Cache raw (unpreprocessed) frames once so the shift-outer/frame-inner
    # loop below doesn't re-read the same PNGs from disk num_shifts+1 times
    # each (the reference's frame-outer/shift-inner loop only reads each
    # frame once; this keeps disk I/O comparable). preprocessFrame/
    # extractFeatures are still called fresh for every (frame, shift) pair,
    # exactly as in run_reference.
    raw_frames = [loadFrame(fid, frames_dir=params["trainingDatasetPath"]) for fid in frame_ids]

    kc_counts = np.zeros((num_frames, num_cols), dtype=np.int64)
    mbon_counts = np.zeros((num_frames, num_cols), dtype=np.int64)

    build_dir = os.path.join("builds_network_batch", f"{params['name']}_b{batch_size}")

    with gpu_exclusive():
        net = _BatchNetwork(params, batch_size, build_dir)
        try:
            input_scale = params["INPUT_SCALE"]
            input_neurons = net.input_neurons
            simulation_steps = int(round(params["PRESENT_TIME_MS"] / params["DT"]))
            magnitude = net.pn_input.vars["magnitude"]

            start = time.time()
            for shift_idx, shift in enumerate(shift_list):
                for chunk_start in range(0, num_frames, batch_size):
                    chunk_end = min(chunk_start + batch_size, num_frames)
                    lanes = chunk_end - chunk_start

                    features_batch = np.empty((lanes, input_neurons), dtype=np.float32)
                    for lane, idx in enumerate(range(chunk_start, chunk_end)):
                        preprocessed = preprocessFrame(raw_frames[idx], shift, params)
                        features_batch[lane] = extractFeatures(preprocessed, params) * input_scale

                    # Zero every lane first: on a partial (final) chunk, lanes
                    # >= `lanes` would otherwise keep whatever a *previous*,
                    # fully-populated chunk last wrote into that lane's
                    # magnitude buffer. Those unused lanes' results are never
                    # read back, but leaving stale nonzero input current
                    # flowing into them every subsequent shift/chunk is
                    # sloppy and unnecessary -- force a clean, deterministic
                    # zero-input state for anything we're not populating.
                    magnitude.view[:, :] = 0.0
                    magnitude.view[:lanes, :] = features_batch
                    magnitude.push_to_device()

                    net.reset_network()
                    for _ in range(simulation_steps):
                        net.model.step_time()
                    net.model.pull_recording_buffers_from_device()

                    kc_spikes = net.kc.spike_recording_data
                    mbon_spikes = net.mbon.spike_recording_data
                    for lane in range(lanes):
                        frame_row = chunk_start + lane
                        kc_counts[frame_row, shift_idx] = len(kc_spikes[lane][0])
                        mbon_counts[frame_row, shift_idx] = len(mbon_spikes[lane][0])

            elapsed = time.time() - start
        finally:
            net.unload()

    best_degree = np.zeros(num_frames, dtype=np.float64)
    for i in range(num_frames):
        novelty_row = [mbon_counts[i, j] for j in range(num_cols)]
        best_degree[i] = _find_optimal_degree(params, shift_list, novelty_row)

    return {
        "frame_ids": np.array(frame_ids, dtype=np.int64),
        "shift_degrees": np.array(shift_list, dtype=np.float64),
        "kc_spike_counts": kc_counts,
        "mbon_spike_counts": mbon_counts,
        "best_degree": best_degree,
        "elapsed_seconds": np.float64(elapsed),
    }
