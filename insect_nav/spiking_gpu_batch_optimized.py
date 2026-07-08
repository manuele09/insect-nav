"""
SpikeCount-optimized variant of insect_nav.spiking_gpu_batch (the native
GeNN model.batch_size=512 winner from Wave 2 GPU benchmarking).

Does NOT modify insect_nav.spiking, insect_nav.genn_models, or
insect_nav.spiking_gpu_batch. Same PN -> KC -> APL(inhibitory) -> MBON
topology, same batch-over-FRAMES-with-shift-as-outer-loop strategy, same
kc_mbon_readonly (VarAccess.READ_ONLY) weight-update model to share the
trained KC->MBON weights across batch lanes -- see spiking_gpu_batch.py's
module docstring for the rationale of that piece, reused verbatim here.

The one thing this module changes is HOW per-presentation KC/MBON spike
counts are obtained:

  - spiking_gpu_batch.py (and insect_nav.spiking/logger.py before it) uses
    GeNN's spike-RECORDING mechanism: kc.spike_recording_enabled = True,
    mbon.spike_recording_enabled = True, model.load(num_recording_timesteps=
    ...), then after the presentation model.pull_recording_buffers_from_
    device() and pop.spike_recording_data[lane] gives back (times, ids)
    arrays per lane, of which only len(times) (the total spike count) is
    ever used (see run_native_batch() and insect_nav.benchmark.run_reference:
    both only call len(...) on the recording, never read exact spike times).

  - This module instead follows the official GeNN MNIST-inference tutorial
    pattern (docs/tutorials/mnist_inference/tutorial_2.ipynb and tutorial_3.
    ipynb): KC and MBON use a custom LIF variant ("lif_spike_count", see
    below) that carries an extra unsigned-int state variable "SpikeCount",
    incremented directly in reset_code every time the neuron spikes. The
    per-presentation total spike count is then just pop.vars["SpikeCount"]
    pulled from device and summed over the neuron axis (per batch lane) --
    no spike-recording buffer, no spike_recording_enabled, no
    pull_recording_buffers_from_device() call at all. SpikeCount is zeroed
    every presentation by the same RESET custom-update mechanism already
    used for V/RefracTime (it's simply added as an extra var_ref on the
    kc/mbon reset custom updates).

  - PN and APL do not get spike counts (never needed here), so they keep
    the original "LIF" builtin / if_model neuron models unchanged.

The LIF dynamics themselves (sim_code, threshold_condition_code, reset_code
for V/RefracTime, and the ExpTC/Rmembrane derived params) are copied
UNCHANGED from GeNN's builtin "LIF" neuron model (dumped via
pygenn.neuron_models.LIF().get_sim_code() / get_threshold_condition_code() /
get_reset_code() / get_derived_params() on pygenn 5.4.0) -- this is a
read-mechanism change only, numerically identical to the unoptimized
variant.

GPU usage note: every GeNNModel built here uses backend="cuda", so its
entire lifetime (construction/build/load, the whole test loop, and the final
model.unload()) MUST run inside insect_nav.benchmark.gpu_exclusive() -- the
single shared 6GB GPU is used by multiple concurrent worktrees/agents.
run_native_batch_optimized() below already does this; do not call
_BatchNetworkOptimized directly outside of a gpu_exclusive() block.
"""

import os
import time

import numpy as np

from pygenn import (
    GeNNModel,
    VarAccess,
    VarAccessMode,
    create_current_source_model,
    create_custom_update_model,
    create_neuron_model,
    create_out_post_var_ref,
    create_var_ref,
    create_weight_update_model,
    init_postsynaptic,
    init_weight_update,
)

from insect_nav.benchmark import gpu_exclusive
from insect_nav.genn_models import if_model, reset_model_if, reset_model_lif
from insect_nav.parameters import load_parameters_from_file
from insect_nav.vision import countFrames, extractFeatures, loadFrame, preprocessFrame

# ── Inference-only KC->MBON weight-update model (identical to
# spiking_gpu_batch.kc_mbon_readonly; see that module's docstring for the
# full rationale of why VarAccess.READ_ONLY is used instead of the plain
# ("g", "scalar") READ_WRITE default of insect_nav.genn_models.anti_hebbian).
kc_mbon_readonly = create_weight_update_model(
    "kc_mbon_readonly",
    vars=[("g", "scalar", VarAccess.READ_ONLY)],
    pre_spike_syn_code="addToPost(g);",
)

# cs_model duplicated (rather than imported) from insect_nav.genn_models,
# byte-identical to the original, just so this module's import list stays
# scoped to what it actually needs beyond genn_models' shared pieces.
cs_model = create_current_source_model(
    "cs_model",
    vars=[("magnitude", "scalar")],
    injection_code="injectCurrent(magnitude);",
)

# ── Custom LIF neuron model with an extra SpikeCount state variable ────────
#
# sim_code / threshold_condition_code / reset_code (V, RefracTime) and the
# ExpTC/Rmembrane derived params below are copied verbatim from GeNN's
# builtin "LIF" neuron model (pygenn.neuron_models.LIF, dumped on pygenn
# 5.4.0 via get_sim_code()/get_threshold_condition_code()/get_reset_code()/
# get_derived_params()). The only addition is "SpikeCount" (unsigned int),
# incremented in reset_code exactly as in GeNN's own MNIST-inference
# tutorial's "if_model" -- see this module's docstring.
lif_spike_count = create_neuron_model(
    "lif_spike_count",
    params=["C", "TauM", "Vrest", "Vreset", "Vthresh", "Ioffset", "TauRefrac"],
    vars=[("V", "scalar"), ("RefracTime", "scalar"), ("SpikeCount", "unsigned int")],
    derived_params=[
        ("ExpTC", lambda pars, dt: np.exp(-dt / pars["TauM"])),
        ("Rmembrane", lambda pars, dt: pars["TauM"] / pars["C"]),
    ],
    sim_code="""
    if (RefracTime <= 0.0) {
      scalar alpha = ((Isyn + Ioffset) * Rmembrane) + Vrest;
      V = alpha - (ExpTC * (alpha - V));
    }
    else {
      RefracTime -= dt;
    }
    """,
    threshold_condition_code="RefracTime <= 0.0 && V >= Vthresh",
    reset_code="""
    V = Vreset;
    RefracTime = TauRefrac;
    SpikeCount++;
    """,
)

# Custom RESET update for kc/mbon: identical V/RefracTime reset to
# insect_nav.genn_models.reset_model_lif (hardcoded V=-60.0f, matching
# LIF_INIT/LIF_PARAMS' Vreset in refnet_source/parameters.json), plus
# zeroing SpikeCount every presentation.
reset_model_lif_spike_count = create_custom_update_model(
    "reset_lif_spike_count",
    var_refs=[
        ("V", "scalar", VarAccessMode.READ_WRITE),
        ("RefracTime", "scalar", VarAccessMode.READ_WRITE),
        ("SpikeCount", "unsigned int", VarAccessMode.READ_WRITE),
    ],
    update_code="""
    V = -60.0f;
    RefracTime = 0.0f;
    SpikeCount = 0;
    """,
)

# Synapse-output-current RESET custom update, byte-identical to
# insect_nav.genn_models.reset_model_syn (duplicated here so this module
# doesn't need to import it separately).
reset_model_syn = create_custom_update_model(
    "reset_syn",
    var_refs=[("Isyn", "scalar", VarAccessMode.READ_WRITE)],
    update_code="Isyn = 0.0f;",
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


class _BatchNetworkOptimized:
    """
    Builds the PN -> KC -> APL -> MBON network with an explicit
    model.batch_size > 1, mirroring spiking_gpu_batch._BatchNetwork but:
      - kc and mbon use the custom lif_spike_count neuron model (SpikeCount
        var) instead of builtin "LIF" + spike_recording_enabled,
      - no spike recording anywhere (no spike_recording_enabled, no
        num_recording_timesteps passed to model.load()),
      - kc/mbon RESET custom updates also zero SpikeCount each presentation.

    Must be constructed and torn down (via .unload()) entirely inside
    insect_nav.benchmark.gpu_exclusive().
    """

    def __init__(self, params: dict, batch_size: int, build_dir: str, timing_enabled: bool = False):
        self.params = params
        self.batch_size = batch_size

        input_neurons = 0
        if params["USE_VERTICAL_DIST"]:
            input_neurons += params["WIDTH"]
        if params["USE_HORIZONTAL_DIST"]:
            input_neurons += params["HEIGHT"]
        self.input_neurons = input_neurons

        self.model = GeNNModel("float", f"{params['name']}_batch{batch_size}_opt", backend="cuda")
        self.model.dt = params["DT"]
        self.model.seed = 1
        self.model.batch_size = batch_size  # MUST be set before build()/load()
        if timing_enabled:
            self.model.timing_enabled = True  # MUST be set before build()

        lif_params = params["LIF_PARAMS"]
        lif_init = params["LIF_INIT"]
        lif_spike_count_init = {**lif_init, "SpikeCount": 0}

        self.pn = self.model.add_neuron_population(
            "pn", input_neurons, "LIF", lif_params, lif_init,
        )
        self.kc = self.model.add_neuron_population(
            "kc", params["NUM_KC"], lif_spike_count, lif_params, lif_spike_count_init,
        )
        self.apln = self.model.add_neuron_population(
            "apln", 1, if_model, params["IF_PARAMS"], params["IF_INIT"],
        )
        self.mbon = self.model.add_neuron_population(
            "mbon", 1, lif_spike_count, lif_params, lif_spike_count_init,
        )

        self.neuron_populations = [self.pn, self.kc, self.apln, self.mbon]
        self.neuron_populations_names = ["pn", "kc", "apln", "mbon"]

        # NOTE: no spike_recording_enabled here -- SpikeCount replaces it
        # entirely for kc/mbon, and pn/apln never needed spike recording.

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
        self.model.load()  # no num_recording_timesteps: no spike-recording buffer needed at all

    def _create_reset_custom_update(self):
        for pop_name, pop in zip(self.neuron_populations_names, self.neuron_populations):
            if pop_name == "apln":
                self.model.add_custom_update(
                    f"reset_{pop_name}", "RESET", reset_model_if, {}, {},
                    {"V": create_var_ref(pop, "V")}, {},
                )
            elif pop_name in ("kc", "mbon"):
                self.model.add_custom_update(
                    f"reset_{pop_name}", "RESET", reset_model_lif_spike_count, {}, {},
                    {
                        "V": create_var_ref(pop, "V"),
                        "RefracTime": create_var_ref(pop, "RefracTime"),
                        "SpikeCount": create_var_ref(pop, "SpikeCount"),
                    }, {},
                )
            else:
                # pn: plain LIF, no SpikeCount.
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


def run_native_batch_optimized(
    params_path: str, batch_size: int, frame_ids=None, timing_enabled: bool = False,
) -> dict:
    """
    Run the SpikeCount-optimized native-batch GPU variant over `frame_ids`
    (default: all frames in trainingDatasetPath) x all angular shifts,
    batching up to `batch_size` frames per GeNN simulate() call (shift is a
    sequential outer loop, never batched). Returns the same dict shape as
    insect_nav.benchmark.run_reference / spiking_gpu_batch.run_native_batch
    so results are directly comparable via compare_kc_spike_counts.

    elapsed_seconds covers the frame x shift test loop only (build/load/
    unload excluded), consistent with run_reference.

    If timing_enabled is True, model.timing_enabled is set before build()
    and the returned dict additionally carries a "timing" sub-dict with
    GeNN's own timing counters (neuron_update_time, presynaptic_update_time,
    custom_update_reset_time, init_time, init_sparse_time) accumulated over
    the whole run, plus "build_seconds" (always present).
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

    raw_frames = [loadFrame(fid, frames_dir=params["trainingDatasetPath"]) for fid in frame_ids]

    kc_counts = np.zeros((num_frames, num_cols), dtype=np.int64)
    mbon_counts = np.zeros((num_frames, num_cols), dtype=np.int64)

    build_dir = os.path.join("builds_network_batch", f"{params['name']}_b{batch_size}_opt")

    timing = None
    with gpu_exclusive():
        build_start = time.time()
        net = _BatchNetworkOptimized(params, batch_size, build_dir, timing_enabled=timing_enabled)
        build_seconds = time.time() - build_start
        try:
            input_scale = params["INPUT_SCALE"]
            input_neurons = net.input_neurons
            simulation_steps = int(round(params["PRESENT_TIME_MS"] / params["DT"]))
            magnitude = net.pn_input.vars["magnitude"]
            kc_spike_count_var = net.kc.vars["SpikeCount"]
            mbon_spike_count_var = net.mbon.vars["SpikeCount"]

            start = time.time()
            for shift_idx, shift in enumerate(shift_list):
                for chunk_start in range(0, num_frames, batch_size):
                    chunk_end = min(chunk_start + batch_size, num_frames)
                    lanes = chunk_end - chunk_start

                    features_batch = np.empty((lanes, input_neurons), dtype=np.float32)
                    for lane, idx in enumerate(range(chunk_start, chunk_end)):
                        preprocessed = preprocessFrame(raw_frames[idx], shift, params)
                        features_batch[lane] = extractFeatures(preprocessed, params) * input_scale

                    # Same partial-chunk zeroing rationale as spiking_gpu_batch.py:
                    # unused lanes on the final (short) chunk must not keep stale
                    # nonzero input current from a previous, fully-populated chunk.
                    magnitude.view[:, :] = 0.0
                    magnitude.view[:lanes, :] = features_batch
                    magnitude.push_to_device()

                    net.reset_network()
                    for _ in range(simulation_steps):
                        net.model.step_time()

                    kc_spike_count_var.pull_from_device()
                    mbon_spike_count_var.pull_from_device()
                    # views are (batch_size, num_neurons); sum over neurons per
                    # lane gives the same total-spike-count-in-presentation-window
                    # value that len(spike_recording_data[lane][0]) gave before.
                    kc_lane_totals = kc_spike_count_var.view[:lanes, :].sum(axis=1)
                    mbon_lane_totals = mbon_spike_count_var.view[:lanes, :].sum(axis=1)

                    for lane in range(lanes):
                        frame_row = chunk_start + lane
                        kc_counts[frame_row, shift_idx] = int(kc_lane_totals[lane])
                        mbon_counts[frame_row, shift_idx] = int(mbon_lane_totals[lane])

            elapsed = time.time() - start

            if timing_enabled:
                timing = {
                    "neuron_update_time": float(net.model.neuron_update_time),
                    "presynaptic_update_time": float(net.model.presynaptic_update_time),
                    "init_time": float(net.model.init_time),
                    "init_sparse_time": float(net.model.init_sparse_time),
                }
                try:
                    timing["custom_update_reset_time"] = float(net.model.get_custom_update_time("RESET"))
                except Exception:
                    pass
                try:
                    timing["custom_update_reset_transpose_time"] = float(
                        net.model.get_custom_update_transpose_time("RESET")
                    )
                except Exception:
                    pass
        finally:
            net.unload()

    best_degree = np.zeros(num_frames, dtype=np.float64)
    for i in range(num_frames):
        novelty_row = [mbon_counts[i, j] for j in range(num_cols)]
        best_degree[i] = _find_optimal_degree(params, shift_list, novelty_row)

    result = {
        "frame_ids": np.array(frame_ids, dtype=np.int64),
        "shift_degrees": np.array(shift_list, dtype=np.float64),
        "kc_spike_counts": kc_counts,
        "mbon_spike_counts": mbon_counts,
        "best_degree": best_degree,
        "elapsed_seconds": np.float64(elapsed),
        "build_seconds": np.float64(build_seconds),
    }
    if timing is not None:
        result["timing"] = timing
    return result
