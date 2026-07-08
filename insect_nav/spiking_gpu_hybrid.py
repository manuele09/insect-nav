"""
Hybrid GPU variant of the PN->KC->APL(inhibitory)->MBON mushroom-body circuit
that ELIMINATES the sequential outer loop over angular shifts entirely, by
combining the two Wave-2 techniques already validated independently:

  * Wave-2 Experiment 1 ("native batch", insect_nav/spiking_gpu_batch.py,
    another worktree, read-only reference): GeNN's native `model.batch_size`
    mechanism, batching FRAMES into independent batch "lanes" so a single
    simulate() call processes many frames at once. Validated: batch_size=512
    gives 5.95x vs CPU / 3.75x vs GPU batch=1 (batch_size=64 was barely worth
    it -- there's a fixed per-chunk overhead that only pays off at large M).

  * Wave-2 Experiment 2 ("replicated", insect_nav/spiking_gpu_replicated.py,
    another worktree, read-only reference): manually replicating the KC/APLN/
    MBON populations B times with block-diagonal (structurally disjoint)
    SPARSE connectivity, so B independent copies of the circuit coexist in one
    GeNNModel and run in one simulate() call. Validated (adversarial isolation
    test): zero cross-replica leakage. At B=64: 4.06x vs CPU.

Hybrid strategy (this module): build the population-replication of Experiment
2 along the SHIFT axis (NUM_SHIFTS+1 = 21 replicas, one per angular shift,
block-diagonal, structurally isolated) -- this replication is baked into the
network topology ONCE -- and THEN apply Experiment 1's native
`model.batch_size = M` on top of that already-21x-replicated network, to
batch FRAMES into lanes. A single simulate() call for a chunk of M frames now
covers ALL 21 shifts x M frames = 21*M presentations at once. The only
remaining loop is over chunks of frame_ids (size M):

    for chunk in chunks(frame_ids, batch_size=M):
        for each frame f in chunk (lane m):
            for each shift j (0..20, physical replica j):
                preprocess f under shift_j, extract features, scale
                write into features_batch[m, j*num_pn:(j+1)*num_pn]
        simulate_chunk(features_batch)     # ONE simulate() for the whole
                                            # chunk: M frames x 21 shifts
        for each frame f in chunk (lane m), each shift j (replica j):
            kc_counts[frame, j]   = per-replica KC spike count in lane m
            mbon_counts[frame, j] = per-replica MBON spike count in lane m
    for frame in frame_ids:
        best_degree[frame] = same grouping/argmin logic as
            NeuralModelBase.find_optimal_degree (insect_nav/base.py)

This module does NOT modify insect_nav.spiking or insect_nav.genn_models --
it only imports neuron/synapse/custom-update model definitions from
genn_models.py and builds an independent GeNNModel.

Combining the two techniques required stacking BOTH surprises discovered
independently in Experiments 1 and 2:

  1. (From Experiment 2) All four synapse populations must use block-diagonal
     SPARSE connectivity with explicit index offsets per shift-replica r, so
     that NO synapse crosses a replica boundary -- this is built once, before
     model.batch_size is even set, exactly like Experiment 2's connectivity
     construction (only the replication axis is now "shift" instead of
     "frame").

  2. (From Experiment 1) Once `model.batch_size = M` is set (M > 1), GeNN
     automatically gives every per-neuron state variable (V, RefracTime, the
     current-source "magnitude" var, ...) a leading (M, ...) batch axis for
     free -- nothing manual needed there. BUT the KC->MBON `g` var of the
     custom `anti_hebbian` weight-update model is READ_WRITE by default (we
     are not allowed to change genn_models.py to mark it read-only), so GeNN
     ALSO duplicates it across the M batch lanes as soon as batch_size > 1.
     Combined with the shift-replica axis from point 1 (which ALSO tiled `g`
     21x with per-replica offsets, all replicas sharing the same trained
     weights), this means the initial value passed to
     init_weight_update(anti_hebbian, ..., {"g": ...}) needs to be built with
     TWO nested levels of replication:
         g_1d          shape (NUM_KC,)             -- the trained weights
         g_shift_tiled shape (21*NUM_KC,)           -- np.tile(g_1d, 21)
         g_batch_tiled shape (M, 21*NUM_KC)         -- np.tile(g_shift_tiled, (M, 1))
     Since `mod` is a permanent, never-toggled -1 (plasticity disabled, pure
     inference -- writes to g are gated by `if (mod > 0)` in anti_hebbian's
     pre/post_spike_syn_code), every (lane, replica) copy of g simply stays
     forever equal to the same trained values; the duplication is harmless
     "wasted" memory (NUM_KC floats x 21 x M), not a correctness issue.

  3. (From Experiment 1) pop.spike_recording_data, once batch_size > 1, is a
     list of M tuples (times, ids), one per batch lane. (From Experiment 2)
     within a single lane's ids array, replica r's KC neurons occupy ids in
     [r*NUM_KC, (r+1)*NUM_KC) and replica r's MBON neuron IS id r (1 neuron
     per replica, so id == replica index directly, same as Experiment 2's
     `per_replica_counts(..., neurons_per_replica=1)`). So reading counts
     requires: pick lane m's (times, ids) tuple, then bincount ids // NUM_KC
     (for kc) or ids directly (for mbon) into 21 per-replica counts.

  4. A single model.custom_update("RESET") call resets EVERY (lane, replica)
     combination at once -- it's just an elementwise reset over the whole,
     now doubly-larger, batched array. No per-lane or per-replica reset calls
     needed (same fact independently established in both source experiments).

Exposes:
    run_hybrid(params_path, batch_size, frame_ids=None) -> dict
        Same return schema as insect_nav.benchmark.run_reference.
    HybridNetwork
        Lower-level class building/holding the single hybrid GeNNModel.
    validate_hybrid_isolation(...)
        Small-scale adversarial self-test proving BOTH: (a) no cross-shift-
        replica leakage within a lane, and (b) no cross-frame-lane leakage
        across the batch dimension.
"""

import os
import time

import numpy as np

from pygenn import (
    GeNNModel,
    create_out_post_var_ref,
    create_var_ref,
    init_postsynaptic,
    init_weight_update,
)

from insect_nav.benchmark import _find_optimal_degree, _shift_degrees, gpu_exclusive
from insect_nav.genn_models import (
    anti_hebbian,
    cs_model,
    if_model,
    reset_model_if,
    reset_model_lif,
    reset_model_syn,
)
from insect_nav.parameters import load_parameters_from_file
from insect_nav.vision import countFrames, extractFeatures, loadFrame, preprocessFrame


def num_pn_neurons(params: dict) -> int:
    """Same computation as NeuralModelBase.__init__ for self.input_neurons."""
    n = 0
    if params["USE_VERTICAL_DIST"]:
        n += params["WIDTH"]
    if params["USE_HORIZONTAL_DIST"]:
        n += params["HEIGHT"]
    return n


def _sanitize_model_name(name: str) -> str:
    """GeNN model names must be safe C-ish identifiers; params['name'] contains
    dots (e.g. 't95_..._w0.169861...'). Strip anything non-alnum/underscore."""
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    if not out or not (out[0].isalpha() or out[0] == "_"):
        out = "m_" + out
    return out


# ── Hybrid network: shift-replicated topology + native frame batching ───────

class HybridNetwork:
    """
    A single GeNNModel that is BOTH:
      - physically replicated `num_shift_replicas` times (block-diagonal,
        structurally isolated), one replica per angular shift, AND
      - natively batched (`model.batch_size = batch_size`) over frames.

    Population sizes (R = num_shift_replicas, M = batch_size):
        pn   : R * num_pn   neurons, x M batch lanes (auto, via batch_size)
        kc   : R * NUM_KC   neurons, x M batch lanes
        apln : R            neurons, x M batch lanes (1 per shift-replica)
        mbon : R            neurons, x M batch lanes (1 per shift-replica)

    Inference-only: KC->MBON plasticity is permanently disabled (`mod=-1`,
    passed as a static weight-update param, never toggled).
    """

    def __init__(self, params: dict, num_shift_replicas: int, batch_size: int,
                 weights_dir: str = None, use_gpu: bool = True,
                 build_dir: str = "./builds_network_hybrid",
                 name_suffix: str = ""):
        self.params = params
        self.R = int(num_shift_replicas)
        self.M = int(batch_size)
        self.num_pn = num_pn_neurons(params)
        self.num_kc = params["NUM_KC"]
        self.use_gpu = use_gpu

        weights_dir = weights_dir or params["weightsPath"]
        pn_kc_ind = np.load(os.path.join(weights_dir, "pn_kc_ind.npy"))
        kc_mbon_g = np.load(os.path.join(weights_dir, "kc_mbon_g_0.npy")).astype(np.float32)
        if pn_kc_ind.shape[0] != 2:
            raise ValueError(f"Unexpected pn_kc_ind shape {pn_kc_ind.shape}, expected (2, n_synapses)")
        if kc_mbon_g.shape[0] != self.num_kc:
            raise ValueError(
                f"kc_mbon_g_0.npy has {kc_mbon_g.shape[0]} entries, expected NUM_KC={self.num_kc}"
            )

        self._build(pn_kc_ind, kc_mbon_g, build_dir, name_suffix)

    # ── Construction ─────────────────────────────────────────────────────────

    def _build(self, pn_kc_ind, kc_mbon_g, build_dir, name_suffix):
        params = self.params
        R, M, num_pn, num_kc = self.R, self.M, self.num_pn, self.num_kc
        backend = "cuda" if self.use_gpu else "single_threaded_cpu"

        model_name = _sanitize_model_name(f"{params['name']}_hyb_r{R}_b{M}{name_suffix}")
        self.model = GeNNModel("float", model_name, backend=backend)
        self.model.dt = params["DT"]
        self.model.seed = 1
        # Native batching over FRAMES, applied ON TOP of the shift-replicated
        # topology built below -- must be set before add_neuron_population /
        # build()/load(), exactly as in Experiment 1.
        self.model.batch_size = M

        self.pn = self.model.add_neuron_population(
            "pn", R * num_pn, "LIF", params["LIF_PARAMS"], params["LIF_INIT"])
        self.kc = self.model.add_neuron_population(
            "kc", R * num_kc, "LIF", params["LIF_PARAMS"], params["LIF_INIT"])
        self.apln = self.model.add_neuron_population(
            "apln", R, if_model, params["IF_PARAMS"], params["IF_INIT"])
        self.mbon = self.model.add_neuron_population(
            "mbon", R, "LIF", params["LIF_PARAMS"], params["LIF_INIT"])

        self.kc.spike_recording_enabled = True
        self.mbon.spike_recording_enabled = True

        self.neuron_populations = [self.pn, self.kc, self.apln, self.mbon]
        self.neuron_populations_names = ["pn", "kc", "apln", "mbon"]

        self.pn_input = self.model.add_current_source(
            "pn_input", cs_model, self.pn, {}, {"magnitude": 0.0})

        # ── PN -> KC: block-diagonal SPARSE over the R shift-replicas,
        # constant (non-plastic) weight. Built ONCE, shared across all M
        # batch lanes (native batching duplicates per-neuron/per-synapse
        # STATE, not this connectivity or the constant weight param). ──
        pre0 = pn_kc_ind[0].astype(np.int64)
        post0 = pn_kc_ind[1].astype(np.int64)
        n_syn = pre0.shape[0]
        pre_all = np.empty(n_syn * R, dtype=np.int64)
        post_all = np.empty(n_syn * R, dtype=np.int64)
        for r in range(R):
            sl = slice(r * n_syn, (r + 1) * n_syn)
            pre_all[sl] = pre0 + r * num_pn
            post_all[sl] = post0 + r * num_kc

        self.pn_kc = self.model.add_synapse_population(
            "pn_kc", "SPARSE", self.pn, self.kc,
            init_weight_update("StaticPulseConstantWeight", {"g": params["PN_KC_WEIGHT"]}),
            init_postsynaptic("ExpCurr", {"tau": params["PN_KC_TAU"]}),
        )
        self.pn_kc.set_sparse_connections(pre_all, post_all)

        # ── KC -> APLN: block-diagonal SPARSE (each replica's NUM_KC KCs -> its own APLN) ──
        kc_pre = np.empty(num_kc * R, dtype=np.int64)
        apln_post = np.empty(num_kc * R, dtype=np.int64)
        for r in range(R):
            sl = slice(r * num_kc, (r + 1) * num_kc)
            kc_pre[sl] = np.arange(num_kc, dtype=np.int64) + r * num_kc
            apln_post[sl] = r

        self.kc_apln = self.model.add_synapse_population(
            "kc_apln", "SPARSE", self.kc, self.apln,
            init_weight_update("StaticPulse", vars={"g": params["KC_APLN_WEIGHT"]}),
            init_postsynaptic("DeltaCurr"),
        )
        self.kc_apln.set_sparse_connections(kc_pre, apln_post)

        # ── APLN -> KC: block-diagonal SPARSE (each replica's APLN -> its own NUM_KC KCs) ──
        apln_pre = np.empty(num_kc * R, dtype=np.int64)
        kc_post = np.empty(num_kc * R, dtype=np.int64)
        for r in range(R):
            sl = slice(r * num_kc, (r + 1) * num_kc)
            apln_pre[sl] = r
            kc_post[sl] = np.arange(num_kc, dtype=np.int64) + r * num_kc

        self.apln_kc = self.model.add_synapse_population(
            "apln_kc", "SPARSE", self.apln, self.kc,
            init_weight_update("StaticPulse", vars={"g": params["APLN_KC_WEIGHT"]}),
            init_postsynaptic("ExpCurr", {"tau": params["APLN_KC_TAU"]}),
        )
        self.apln_kc.set_sparse_connections(apln_pre, kc_post)

        # ── KC -> MBON: block-diagonal SPARSE over R shift-replicas, PLUS
        # native batch duplication over M lanes for the `g` var (READ_WRITE
        # by default -- see module docstring). Two nested levels of
        # replication for the initial `g` array:
        #   (NUM_KC,) --tile x R--> (R*NUM_KC,) --tile x M--> (M, R*NUM_KC)
        kcm_pre = np.empty(num_kc * R, dtype=np.int64)
        kcm_post = np.empty(num_kc * R, dtype=np.int64)
        for r in range(R):
            sl = slice(r * num_kc, (r + 1) * num_kc)
            kcm_pre[sl] = np.arange(num_kc, dtype=np.int64) + r * num_kc
            kcm_post[sl] = r
        g_shift_tiled = np.tile(kc_mbon_g, R).astype(np.float32)  # (R*NUM_KC,)
        g_init = (g_shift_tiled if M == 1
                  else np.tile(g_shift_tiled, (M, 1)))  # (M, R*NUM_KC) if M>1

        # mod is a permanent -1 (no plasticity, pure inference) -- kept as a
        # static param (never toggled), matching Experiment 2's approach.
        self.kc_mbon = self.model.add_synapse_population(
            "kc_mbon", "SPARSE", self.kc, self.mbon,
            init_weight_update(anti_hebbian, {"mod": -1.0}, {"g": g_init}),
            init_postsynaptic("ExpCurr", {"tau": params["KC_MBON_TAU"]}),
        )
        self.kc_mbon.set_sparse_connections(kcm_pre, kcm_post)

        self.synapse_populations = [self.pn_kc, self.kc_apln, self.apln_kc, self.kc_mbon]
        self.synapse_populations_names = ["pn_kc", "kc_apln", "apln_kc", "kc_mbon"]

        self._create_reset_custom_update()

        os.makedirs(build_dir, exist_ok=True)
        self.model.build(build_dir, always_rebuild=True)
        self.model.load(num_recording_timesteps=int(round(params["PRESENT_TIME_MS"] / params["DT"])))

    def _create_reset_custom_update(self):
        """A single model.custom_update("RESET") call resets V/RefracTime/
        Isyn across ALL (batch lane, shift replica) combinations at once --
        it is just an elementwise reset over the whole, now doubly-larger,
        array (fact established independently in both source experiments)."""
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

    # ── Simulation ───────────────────────────────────────────────────────────

    def reset(self):
        self.model.timestep = 0
        self.model.custom_update("RESET")

    def simulate_chunk(self, features_batch: np.ndarray, present_time_ms: float):
        """
        features_batch: (n, R*num_pn) array, n <= M, already scaled by
        INPUT_SCALE. Row m is the concatenation of R per-shift feature
        vectors for frame m: features_batch[m, j*num_pn:(j+1)*num_pn] is
        shift-replica j's input for that frame. Only lanes [0, n) are
        meaningful; if n < M the remaining lanes are zeroed.

        Returns (kc_counts, mbon_counts): two (n, R) int arrays of per-
        (frame, shift) spike counts, in the same row order as features_batch.
        """
        n = features_batch.shape[0]
        view = self.pn_input.vars["magnitude"].view
        if self.M == 1:
            view[:] = features_batch[0]
        else:
            view[:n, :] = features_batch
            if n < self.M:
                view[n:, :] = 0.0
        self.pn_input.vars["magnitude"].push_to_device()

        self.reset()
        steps = int(round(present_time_ms / self.params["DT"]))
        for _ in range(steps):
            self.model.step_time()
        self.model.pull_recording_buffers_from_device()

        R, num_kc = self.R, self.num_kc
        kc_counts = np.zeros((n, R), dtype=np.int64)
        mbon_counts = np.zeros((n, R), dtype=np.int64)
        kc_rec = self.kc.spike_recording_data
        mbon_rec = self.mbon.spike_recording_data
        for lane in range(n):
            _kc_times, kc_ids = kc_rec[lane]
            kc_ids = np.asarray(kc_ids)
            if kc_ids.size:
                kc_counts[lane, :] = np.bincount(kc_ids // num_kc, minlength=R)
            _mbon_times, mbon_ids = mbon_rec[lane]
            mbon_ids = np.asarray(mbon_ids)
            if mbon_ids.size:
                mbon_counts[lane, :] = np.bincount(mbon_ids, minlength=R)
        return kc_counts, mbon_counts

    def unload(self):
        self.model.unload()


# ── High-level driver: same result schema as insect_nav.benchmark.run_reference ──

def run_hybrid(params_path: str, batch_size: int, frame_ids=None,
                weights_dir: str = None, use_gpu: bool = True,
                build_dir: str = "./builds_network_hybrid") -> dict:
    """
    Run the hybrid (shift-replicated + frame-batched) network over
    `frame_ids` (default: all frames) x all angular shifts, with NO
    sequential loop over shifts at all: shifts are physical replicas within
    one GeNNModel, and frames are native GeNN batch lanes. The only loop is
    over chunks of `batch_size` frames:

        for chunk in chunks(frame_ids, batch_size):
            for frame f in chunk (lane m), shift j (replica j):
                inject f's shift_j features into lane m's replica-j PN slice
            simulate() ONCE for the whole chunk (covers M frames x 21 shifts)
            for frame f in chunk (lane m), shift j (replica j):
                read per-replica KC/MBON spike counts from lane m
        for frame in frame_ids:
            best_degree[frame] = find_optimal_degree(...)

    Returns the same dict schema as insect_nav.benchmark.run_reference:
    frame_ids, shift_degrees, kc_spike_counts, mbon_spike_counts,
    best_degree, elapsed_seconds (loop only, build/load/unload excluded,
    mirroring run_reference's timing convention), PLUS an extra
    build_seconds field (wall-clock time of HybridNetwork construction,
    i.e. model.build()+model.load(), measured separately from the test
    loop, NOT included in elapsed_seconds).
    """
    params = load_parameters_from_file(params_path)
    num_shifts = params["NUM_SHIFTS"]
    shift_degrees = _shift_degrees(params, num_shifts)
    R = len(shift_degrees)
    num_pn = num_pn_neurons(params)

    if frame_ids is None:
        frame_ids = list(range(countFrames(params["trainingDatasetPath"])))
    else:
        frame_ids = list(frame_ids)

    num_frames = len(frame_ids)
    kc_counts_matrix = np.zeros((num_frames, R), dtype=np.int64)
    mbon_counts_matrix = np.zeros((num_frames, R), dtype=np.int64)

    present_time_ms = params["PRESENT_TIME_MS"]
    input_scale = params["INPUT_SCALE"]
    frames_dir = params["trainingDatasetPath"]

    def _run_loop(net: HybridNetwork):
        M = net.M
        start = time.time()
        for chunk_start in range(0, num_frames, M):
            chunk_positions = list(range(chunk_start, min(chunk_start + M, num_frames)))
            n = len(chunk_positions)
            features_batch = np.zeros((n, R * num_pn), dtype=np.float32)
            for lane, pos in enumerate(chunk_positions):
                frame_number = frame_ids[pos]
                frame = loadFrame(frame_number, frames_dir=frames_dir)
                for j, shift in enumerate(shift_degrees):
                    preprocessed = preprocessFrame(frame, shift, params)
                    features = extractFeatures(preprocessed, params)
                    features_batch[lane, j * num_pn:(j + 1) * num_pn] = features * input_scale

            kc_counts, mbon_counts = net.simulate_chunk(features_batch, present_time_ms)
            for lane, pos in enumerate(chunk_positions):
                kc_counts_matrix[pos, :] = kc_counts[lane, :]
                mbon_counts_matrix[pos, :] = mbon_counts[lane, :]
        elapsed = time.time() - start
        return elapsed

    if use_gpu:
        with gpu_exclusive():
            build_start = time.time()
            net = HybridNetwork(params, num_shift_replicas=R, batch_size=batch_size,
                                 weights_dir=weights_dir, use_gpu=True, build_dir=build_dir)
            build_seconds = time.time() - build_start
            try:
                elapsed = _run_loop(net)
            finally:
                net.unload()
    else:
        build_start = time.time()
        net = HybridNetwork(params, num_shift_replicas=R, batch_size=batch_size,
                             weights_dir=weights_dir, use_gpu=False, build_dir=build_dir)
        build_seconds = time.time() - build_start
        try:
            elapsed = _run_loop(net)
        finally:
            net.unload()

    best_degree = np.zeros(num_frames, dtype=np.float64)
    for i in range(num_frames):
        novelty_row = mbon_counts_matrix[i, :].tolist()
        best_degree[i] = _find_optimal_degree(params, shift_degrees, novelty_row)

    return {
        "frame_ids": np.array(frame_ids, dtype=np.int64),
        "shift_degrees": np.array(shift_degrees, dtype=np.float64),
        "kc_spike_counts": kc_counts_matrix,
        "mbon_spike_counts": mbon_counts_matrix,
        "best_degree": best_degree,
        "elapsed_seconds": np.float64(elapsed),
        "build_seconds": np.float64(build_seconds),
    }


# ── Isolation self-test: BOTH shift-replica axis AND frame-batch axis ───────

def validate_hybrid_isolation(params_path: str, num_shift_replicas: int = 4,
                               batch_size: int = 3, use_gpu: bool = True,
                               weights_dir: str = None,
                               build_dir: str = "./builds_network_hybrid_isotest") -> dict:
    """
    Build a small hybrid network (default R=4 shift-replicas x M=3 batch
    lanes) and prove there is NO leakage along either the shift-replica axis
    (within one lane) or the frame-batch axis (across lanes):

      1. "Config A": lane 0 replica 0 gets a real frame's features; every
         OTHER (lane, replica) slot in the whole (M, R*num_pn) input array
         is zero. Record lane 0 / replica 0's KC ids + MBON count.
      2. "Config B": lane 0 replica 0 gets the SAME features; every OTHER
         (lane, replica) slot -- i.e. every other replica within lane 0, AND
         every replica within every other lane -- gets a large adversarial
         drive (reversed + boosted features), designed to maximize activity
         elsewhere in the array so any leak into (lane 0, replica 0) would
         be detectable.
      3. Assert (lane 0, replica 0)'s recorded spikes are bit-for-bit
         identical between config A and config B (deterministic network:
         fixed seed, no stochastic elements in LIF/IF/StaticPulse/
         anti_hebbian at mod=-1) -- any cross-replica OR cross-lane leak
         would change (lane 0, replica 0)'s Isyn and hence its spikes.
      4. Sanity check: assert something DID fire elsewhere in config B (both
         some other replica within lane 0, and some replica in another lane),
         so the test isn't a trivial no-op.

    Returns a dict with the raw comparisons and an "isolated" bool.
    """
    params = load_parameters_from_file(params_path)
    num_pn = num_pn_neurons(params)
    num_kc = params["NUM_KC"]
    frames_dir = params["trainingDatasetPath"]
    present_time_ms = params["PRESENT_TIME_MS"]

    frame0 = loadFrame(0, frames_dir=frames_dir)
    preprocessed0 = preprocessFrame(frame0, 0.0, params)
    features0 = extractFeatures(preprocessed0, params)
    magnitude0 = (features0 * params["INPUT_SCALE"]).astype(np.float32)

    adversarial = (features0[::-1] * params["INPUT_SCALE"] * 50.0 + 5.0).astype(np.float32)

    net = HybridNetwork(params, num_shift_replicas=num_shift_replicas, batch_size=batch_size,
                         weights_dir=weights_dir, use_gpu=use_gpu, build_dir=build_dir)
    R, M = net.R, net.M
    try:
        def _read_lane_replica(kc_rec, mbon_rec, lane, replica):
            _t, ids = kc_rec[lane]
            ids = np.asarray(ids)
            kc_ids_r = np.sort(ids[(ids >= replica * num_kc) & (ids < (replica + 1) * num_kc)])
            _t2, mids = mbon_rec[lane]
            mids = np.asarray(mids)
            mbon_count_r = int(np.sum(mids == replica))
            return kc_ids_r, mbon_count_r

        # Config A: only (lane 0, replica 0) driven; everything else zero.
        feat_a = np.zeros((M, R * num_pn), dtype=np.float32)
        feat_a[0, 0:num_pn] = magnitude0
        kc_a, mbon_a = net.simulate_chunk(feat_a, present_time_ms)
        kc_ids_a, mbon_r0_a = _read_lane_replica(net.kc.spike_recording_data,
                                                  net.mbon.spike_recording_data, 0, 0)

        # Config B: (lane 0, replica 0) gets the SAME features; every other
        # (lane, replica) slot gets the adversarial drive.
        feat_b = np.zeros((M, R * num_pn), dtype=np.float32)
        for lane in range(M):
            for r in range(R):
                if lane == 0 and r == 0:
                    feat_b[lane, r * num_pn:(r + 1) * num_pn] = magnitude0
                else:
                    feat_b[lane, r * num_pn:(r + 1) * num_pn] = adversarial
        kc_b, mbon_b = net.simulate_chunk(feat_b, present_time_ms)
        kc_ids_b, mbon_r0_b = _read_lane_replica(net.kc.spike_recording_data,
                                                  net.mbon.spike_recording_data, 0, 0)

        # Sanity: something else must have fired in config B.
        other_replicas_in_lane0_fired = bool(mbon_b[0, 1:].sum() > 0) or bool(kc_b[0, 1:].sum() > 0)
        other_lanes_fired = bool(mbon_b[1:, :].sum() > 0) or bool(kc_b[1:, :].sum() > 0)

        kc_r0_identical = np.array_equal(kc_ids_a, kc_ids_b)
        mbon_r0_identical = mbon_r0_a == mbon_r0_b

        isolated = (kc_r0_identical and mbon_r0_identical
                    and other_replicas_in_lane0_fired and other_lanes_fired)

        return {
            "num_shift_replicas": R,
            "batch_size": M,
            "lane0_replica0_kc_count_config_a": int(kc_ids_a.shape[0]),
            "lane0_replica0_kc_count_config_b": int(kc_ids_b.shape[0]),
            "lane0_replica0_mbon_count_config_a": mbon_r0_a,
            "lane0_replica0_mbon_count_config_b": mbon_r0_b,
            "lane0_replica0_kc_ids_identical": bool(kc_r0_identical),
            "lane0_replica0_mbon_identical": bool(mbon_r0_identical),
            "other_replicas_in_lane0_fired_in_config_b": other_replicas_in_lane0_fired,
            "other_lanes_fired_in_config_b": other_lanes_fired,
            "isolated": bool(isolated),
        }
    finally:
        net.unload()
