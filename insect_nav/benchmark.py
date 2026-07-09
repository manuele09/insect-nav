"""
Shared benchmark / correctness harness for insect_nav spiking-network variants.

This module is the common infrastructure that "Wave 2" GPU-optimized variants
will be checked against. It deliberately does NOT modify insect_nav.spiking or
insect_nav.genn_models — it only drives NeuralNetwork as-is and records
KC/MBON spike counts per (frame, shift) presentation, so future variants can
be compared against a known-unmodified reference.

Provides:
    - gpu_exclusive(): process-safe GPU lock (fcntl-based flock) so that
      concurrent processes/worktrees never share the single GPU at once.
    - run_reference(): runs insect_nav.spiking.NeuralNetwork, unmodified, over
      a dataset (all frames x all angular shifts) and records KC/MBON spike
      counts plus the best heading per frame.
    - save_run() / load_run(): npz persistence for run results.
    - compare_kc_spike_counts(): correctness comparison between two runs
      (e.g. CPU reference vs GPU reference, or reference vs an optimized
      variant).
"""

import fcntl
import time
from contextlib import contextmanager

import numpy as np

# Fixed, absolute path OUTSIDE any git worktree/repo. Every process that wants
# to touch the GPU (use_gpu=True) must hold this lock for the entire lifetime
# of its NeuralNetwork (build/load through model.unload()) — the GPU has only
# 6GB and must never be shared by concurrent processes, regardless of how
# multiple worktrees/agents happen to get scheduled.
GPU_LOCK_PATH = "/home/emanuele/insect_nav_gpu_bench/gpu.lock"


@contextmanager
def gpu_exclusive(lock_path: str = GPU_LOCK_PATH):
    """
    Context manager granting exclusive access to the (single, shared) GPU.

    Acquires a blocking POSIX file lock (fcntl.flock, LOCK_EX) on `lock_path`
    before yielding, and releases it (LOCK_UN) afterwards, even on exception.
    Must wrap the *entire* lifetime of any NeuralNetwork built with
    use_gpu=True: construction (which builds/loads the GeNN model), the test/
    train loop, and the final nn.model.unload().
    """
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _shift_degrees(params: dict, num_shifts: int) -> list:
    """
    Same angular-shift formula as NeuralModelBase.testNavigation
    (insect_nav/base.py): num_shifts+1 angles, normalized to (-180, 180].
    """
    step = params["DEGREES_PER_SHIFT"]
    shifts = []
    for k in range(num_shifts + 1):
        shift = (-num_shifts / 2 + k) * step
        angle = (shift + 180) % 360 - 180
        shifts.append(angle)
    return shifts


def _find_optimal_degree(params: dict, degree_array: list, novelty_array: list) -> float:
    """
    Re-implementation, identical in logic, of
    NeuralModelBase.find_optimal_degree (insect_nav/base.py): groups the
    minimum-novelty angles, picks the longest consecutive run (closest to 0
    degrees on ties), and returns the group mean.

    Kept as a free function (operating on explicit arrays) rather than
    calling the method on a live NeuralModelBase/NeuralNetwork instance,
    since run_reference already owns per-frame degree/novelty arrays and
    reusing the instance's mutable self.degree_array/self.novelty_array
    buffers would just duplicate that state.
    """
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


def run_reference(params_path: str, use_gpu: bool, frame_ids=None) -> dict:
    """
    Run insect_nav.spiking.NeuralNetwork, unmodified, over `frame_ids`
    (default: all frames in trainingDatasetPath) x all angular shifts, and
    record KC/MBON spike counts for every presentation.

    Builds the network with load_net={"pn_kc": True, "kc_mbon": True} (i.e.
    loads the already-trained connectivity/weights from disk) and
    tuneCurrent=False, exactly as documented. If use_gpu is True, the whole
    lifetime of the NeuralNetwork (construction/build/load, the test loop,
    and the final nn.model.unload()) happens inside gpu_exclusive().

    Returns a dict with:
        frame_ids            (num_frames,)              int64
        shift_degrees         (num_shifts+1,)            float64
        kc_spike_counts       (num_frames, num_shifts+1) int64
        mbon_spike_counts     (num_frames, num_shifts+1) int64
        best_degree           (num_frames,)              float64
        elapsed_seconds       scalar float — wall-clock time for the
                              frame x shift test loop only (NOT including
                              network build/load/unload).
    """
    from insect_nav.parameters import load_parameters_from_file
    from insect_nav.spiking import NeuralNetwork
    from insect_nav.vision import countFrames, loadFrame

    params = load_parameters_from_file(params_path)
    num_shifts = params["NUM_SHIFTS"]
    shift_degrees = _shift_degrees(params, num_shifts)

    if frame_ids is None:
        frame_ids = list(range(countFrames(params["trainingDatasetPath"])))
    else:
        frame_ids = list(frame_ids)

    def _run_loop(nn):
        num_frames = len(frame_ids)
        num_cols = len(shift_degrees)
        kc_counts = np.zeros((num_frames, num_cols), dtype=np.int64)
        mbon_counts = np.zeros((num_frames, num_cols), dtype=np.int64)
        best_degree = np.zeros(num_frames, dtype=np.float64)

        start = time.time()
        for i, frame_number in enumerate(frame_ids):
            frame = loadFrame(frame_number, frames_dir=params["trainingDatasetPath"])
            novelty_row = []
            for j, angle in enumerate(shift_degrees):
                nn.test(frame, angle)
                kc_counts[i, j] = nn.logger.get_spikes("kc")["count"]
                mbon_counts[i, j] = nn.logger.get_spikes("mbon")["count"]
                novelty_row.append(mbon_counts[i, j])
            best_degree[i] = _find_optimal_degree(params, shift_degrees, novelty_row)
        elapsed = time.time() - start
        return kc_counts, mbon_counts, best_degree, elapsed

    if use_gpu:
        with gpu_exclusive():
            nn = NeuralNetwork(
                params, load_net={"pn_kc": True, "kc_mbon": True},
                tuneCurrent=False, use_gpu=True,
            )
            try:
                kc_counts, mbon_counts, best_degree, elapsed = _run_loop(nn)
            finally:
                nn.model.unload()
    else:
        nn = NeuralNetwork(
            params, load_net={"pn_kc": True, "kc_mbon": True},
            tuneCurrent=False, use_gpu=False,
        )
        try:
            kc_counts, mbon_counts, best_degree, elapsed = _run_loop(nn)
        finally:
            nn.model.unload()

    return {
        "frame_ids": np.array(frame_ids, dtype=np.int64),
        "shift_degrees": np.array(shift_degrees, dtype=np.float64),
        "kc_spike_counts": kc_counts,
        "mbon_spike_counts": mbon_counts,
        "best_degree": best_degree,
        "elapsed_seconds": np.float64(elapsed),
    }


def precompute_features(params_path: str, frame_ids=None) -> dict:
    """
    Precompute preprocessFrame + extractFeatures once for every (frame, shift)
    pair, so that repeated benchmark runs can skip CPU-side image
    preprocessing (cv2 crop/grayscale/shift/resize) entirely and read PN
    feature vectors directly from memory instead. This is what let us
    measure how much of a NeuralNetwork/variant's total wall time is CPU
    preprocessing vs actual GPU simulation.

    Returns a dict with:
        frame_ids        (num_frames,)                       int64
        shift_degrees     (num_shifts+1,)                     float64
        features           (num_frames, num_shifts+1, num_pn)  float32 --
                            NOT yet scaled by INPUT_SCALE (applied at
                            consumption time, since it can be re-tuned)
        elapsed_seconds    scalar float -- wall-clock time for this
                            precompute pass alone.
    """
    from insect_nav.parameters import load_parameters_from_file
    from insect_nav.vision import countFrames, extractFeatures, loadFrame, preprocessFrame

    params = load_parameters_from_file(params_path)
    num_shifts = params["NUM_SHIFTS"]
    shift_degrees = _shift_degrees(params, num_shifts)

    if frame_ids is None:
        frame_ids = list(range(countFrames(params["trainingDatasetPath"])))
    else:
        frame_ids = list(frame_ids)

    num_pn = 0
    if params["USE_VERTICAL_DIST"]:
        num_pn += params["WIDTH"]
    if params["USE_HORIZONTAL_DIST"]:
        num_pn += params["HEIGHT"]

    features = np.zeros((len(frame_ids), len(shift_degrees), num_pn), dtype=np.float32)

    start = time.time()
    for i, frame_number in enumerate(frame_ids):
        frame = loadFrame(frame_number, frames_dir=params["trainingDatasetPath"])
        for j, shift in enumerate(shift_degrees):
            preprocessed = preprocessFrame(frame, shift, params)
            features[i, j, :] = extractFeatures(preprocessed, params)
    elapsed = time.time() - start

    return {
        "frame_ids": np.array(frame_ids, dtype=np.int64),
        "shift_degrees": np.array(shift_degrees, dtype=np.float64),
        "features": features,
        "elapsed_seconds": np.float64(elapsed),
    }


def save_run(path: str, **arrays) -> None:
    """Persist run arrays (e.g. the dict returned by run_reference) to `path` via np.savez."""
    np.savez(path, **arrays)


def load_run(path: str) -> dict:
    """Load a run saved by save_run() back into a plain dict of arrays."""
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def compare_kc_spike_counts(ref: dict, candidate: dict) -> dict:
    """
    Compare KC spike counts (and best_degree) between two runs.

    Aligns rows by `frame_ids` (in case the two runs cover different/partial
    frame sets) and reports:
        num_common_frames
        missing_in_candidate         frame_ids present in ref but not candidate
        missing_in_ref                frame_ids present in candidate but not ref
        mean_abs_diff                 mean(|candidate - ref|) over common frames/shifts
        mean_rel_diff                 mean_abs_diff / mean(ref counts)
        max_abs_diff                  max(|candidate - ref|) over common frames/shifts
        best_degree_exact_match_rate  fraction of common frames where best_degree
                                       is exactly equal between ref and candidate
    """
    ref_frame_ids = [int(f) for f in np.asarray(ref["frame_ids"]).tolist()]
    cand_frame_ids = [int(f) for f in np.asarray(candidate["frame_ids"]).tolist()]

    ref_index = {f: i for i, f in enumerate(ref_frame_ids)}
    cand_index = {f: i for i, f in enumerate(cand_frame_ids)}

    common = [f for f in ref_frame_ids if f in cand_index]
    missing_in_candidate = [f for f in ref_frame_ids if f not in cand_index]
    missing_in_ref = [f for f in cand_frame_ids if f not in ref_index]

    ref_kc = np.asarray(ref["kc_spike_counts"])
    cand_kc = np.asarray(candidate["kc_spike_counts"])

    if common:
        ref_rows = np.array([ref_kc[ref_index[f]] for f in common], dtype=np.float64)
        cand_rows = np.array([cand_kc[cand_index[f]] for f in common], dtype=np.float64)
        abs_diff = np.abs(cand_rows - ref_rows)
        mean_abs_diff = float(np.mean(abs_diff))
        max_abs_diff = float(np.max(abs_diff))
        ref_mean = float(np.mean(ref_rows))
        mean_rel_diff = float(mean_abs_diff / ref_mean) if ref_mean != 0 else float("nan")

        ref_best = np.asarray(ref["best_degree"])
        cand_best = np.asarray(candidate["best_degree"])
        ref_best_common = np.array([ref_best[ref_index[f]] for f in common])
        cand_best_common = np.array([cand_best[cand_index[f]] for f in common])
        best_degree_exact_match_rate = float(np.mean(ref_best_common == cand_best_common))
    else:
        mean_abs_diff = float("nan")
        mean_rel_diff = float("nan")
        max_abs_diff = float("nan")
        best_degree_exact_match_rate = float("nan")

    return {
        "num_common_frames": len(common),
        "missing_in_candidate": missing_in_candidate,
        "missing_in_ref": missing_in_ref,
        "mean_abs_diff": mean_abs_diff,
        "mean_rel_diff": mean_rel_diff,
        "max_abs_diff": max_abs_diff,
        "best_degree_exact_match_rate": best_degree_exact_match_rate,
    }
