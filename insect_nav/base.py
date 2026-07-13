import csv
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from insect_nav.parameters import save_parameters_to_file
from insect_nav.plot_style import (
    COLORS,
    add_legend,
    apply_style,
    new_figure,
    save_figure,
    style_axes,
)
from insect_nav.vision import (
    countFrames,
    cropFrame,
    extractFeatures,
    loadFrame,
    preprocessFrame,
    saveFeaturesAsPNG,
    saveFrameAsPNG,
    saveVerticalWeightingHeatmap,
    visualize_vertical_weighting,
)

apply_style()


class NeuralModelBase:
    """
    Base class for biologically inspired visual navigation neural models.

    Provides:
        - Input preprocessing and feature extraction pipeline
        - Training and testing method stubs
        - Navigation evaluation routine (multi-angle scanning)
        - Visualization and CSV logging utilities
    """

    def __init__(self, parameters, load_net=False, num_shifts=None):
        self.params = parameters
        self.load_net = load_net
        self.num_shifts = self.params["NUM_SHIFTS"] if num_shifts is None else num_shifts

        self.degree_array = []
        self.novelty_array = []
        # Optional pluggable override for find_optimal_degree(); a callable
        # matching insect_nav.degree_strategies' shared contract:
        # (degree_array, novelty_array, step, prev_degree) -> (optimal_degree, uncertainty).
        # None (default) keeps the built-in grouping logic below unchanged.
        self.degree_strategy = None

        # Opt-in override via environment: INSECT_NAV_DEGREE_STRATEGY selects
        # one of insect_nav/degree_strategies/ by name ("strategy3",
        # "strategy3_parabolic_interp" or "3"). Read here (not in the caller)
        # so the downstream project can switch strategies without any code
        # change on its side. Unset => built-in find_optimal_degree unchanged.
        _strategy_env = os.environ.get("INSECT_NAV_DEGREE_STRATEGY", "").strip()
        if _strategy_env:
            from insect_nav.degree_strategies import load_strategy
            self.degree_strategy = load_strategy(_strategy_env)
            print(f"[insect_nav] degree strategy override: {_strategy_env}")

        os.makedirs(self.params["plotsTrainPath"], exist_ok=True)
        os.makedirs(self.params["plotsTestPath"], exist_ok=True)
        os.makedirs(self.params["plotsSimulationPath"], exist_ok=True)

        self.input_neurons = 0
        if self.params["USE_VERTICAL_DIST"]:
            self.input_neurons += self.params["WIDTH"]
        if self.params["USE_HORIZONTAL_DIST"]:
            self.input_neurons += self.params["HEIGHT"]

        self.load_weights()

    # ── Input processing ─────────────────────────────────────────────────────

    def input_pipeline(self, frame, shift_degrees):
        preprocessed = preprocessFrame(frame, shift_degrees, self.params)
        return extractFeatures(preprocessed, self.params)

    # ── Abstract stubs ───────────────────────────────────────────────────────

    def train(self, frame):
        pass

    def test(self, frame, shift_degree=0):
        pass

    def save_weights(self):
        pass

    def load_weights(self):
        pass

    def train_batch(self, frame_ids=None, plot_novelties=True):
        network_type = self.params.get("network_type", "").lower()
        if network_type == "spiking":
            self.logger.enable_novelty_tracking()

        if frame_ids is None:
            num_frames = countFrames(self.params["trainingDatasetPath"])
            train_step = self.params["train_step"]
            frame_ids = range(0, num_frames, train_step)

        for frame_number in tqdm(frame_ids, desc="Training"):
            frame = loadFrame(frame_number, frames_dir=self.params["trainingDatasetPath"])
            self.train(frame)

        self.save_weights()
        if network_type == "spiking":
            if plot_novelties:
                self.logger.plot_cumulative_novelty(self.params["plotsTrainPath"])
            novelties = self.logger.get_novelties()
            self.params["activated_kcs"] = novelties["total_unique_kcs"]
            print(f"Activated KCs: {novelties['total_unique_kcs']} / {self.params.get('NUM_KC', 'N/A')}")
            save_parameters_to_file(self.params, self.params["parameters_path"])

    def testNavigation_batch(self, frame_ids=None, debug_mode=True):
        if frame_ids is None:
            num_frames = countFrames(self.params["trainingDatasetPath"])
            train_step = self.params["train_step"]
            frame_ids = range(0, num_frames, train_step)
        frame_ids = list(frame_ids)

        if getattr(self, "batch_size", 1) <= 1:
            cumulative_error = 0
            for frame_number in frame_ids:
                frame = loadFrame(frame_number, frames_dir=self.params["trainingDatasetPath"])
                r = self.testNavigation(
                    frame,
                    frame_number=frame_number,
                    log_path=self.params["plotsTestPath"],
                    debug_print=False,
                )
                angle_rad = r[0] if isinstance(r, tuple) else r
                cumulative_error += math.degrees(angle_rad)
                self.plot_test_results(frame_number, self.params["plotsTestPath"])

            cumulative_error /= len(frame_ids)
            print(f"Cumulative error: {cumulative_error} degrees.")
            return cumulative_error

        # Batched path (self.batch_size > 1, NeuralNetwork only): shift as the
        # sequential outer loop, chunks of batch_size frames as the inner
        # loop, one self.test() call per chunk instead of one per (frame,
        # shift) pair.
        shift_degrees = self._shift_degrees()
        num_frames = len(frame_ids)
        novelty_matrix = [[None] * len(shift_degrees) for _ in range(num_frames)]

        # With precompute_features=True, test() resolves every (frame_id,
        # shift) pair straight from NeuralNetwork's in-memory cache -- skip
        # loadFrame()/disk+cv2 entirely rather than reloading each frame 21
        # times (once per shift) for no reason.
        use_cache = getattr(self, "precompute_features", False)
        for shift_idx, shift in enumerate(shift_degrees):
            for chunk_start in range(0, num_frames, self.batch_size):
                chunk_ids = frame_ids[chunk_start:chunk_start + self.batch_size]
                if use_cache:
                    chunk_frames = [None] * len(chunk_ids)
                else:
                    chunk_frames = [
                        loadFrame(fid, frames_dir=self.params["trainingDatasetPath"]) for fid in chunk_ids
                    ]
                counts = self.test(chunk_frames, shift, frame_id=chunk_ids)
                for i, count in enumerate(counts):
                    novelty_matrix[chunk_start + i][shift_idx] = count

        cumulative_error = 0
        for i, frame_number in enumerate(frame_ids):
            self.degree_array = list(shift_degrees)
            self.novelty_array = novelty_matrix[i]
            best_degree, uncertainty = self.find_optimal_degree()
            self.last_best_degree = best_degree
            self._log_navigation_results(frame_number, best_degree, uncertainty, self.params["plotsTestPath"])
            if debug_mode:
                self.plot_test_results(frame_number, self.params["plotsTestPath"])
            angle_rad = math.radians(-best_degree)
            cumulative_error += math.degrees(angle_rad)

        cumulative_error /= num_frames
        print(f"Cumulative error: {cumulative_error} degrees.")
        return cumulative_error

    # ── Navigation evaluation ────────────────────────────────────────────────

    def _shift_degrees(self):
        """
        The (num_shifts+1) angular shift values scanned by testNavigation, in
        order. Factored out of testNavigation's loop so other callers (e.g.
        NeuralNetwork's batched testNavigation_batch and feature cache) reuse
        the exact same formula instead of re-deriving it.
        """
        step = self.params["DEGREES_PER_SHIFT"]
        shifts = []
        for k in range(self.num_shifts + 1):
            shift = (-self.num_shifts / 2 + k) * step
            angle = (shift + 180) % 360 - 180
            shifts.append(angle)
        return shifts

    def testNavigation(self, frame, frame_number=-1, log_path=None, debug_print=True, return_timing=False):
        """
        Scan over angular shifts, find the minimum-novelty heading, and return
        the corresponding turning command in radians.
        """
        start_time = time.time()
        self.degree_array.clear()
        self.novelty_array.clear()

        for angle in self._shift_degrees():
            novelty = self.test(frame, angle)
            self.degree_array.append(angle)
            self.novelty_array.append(novelty)

        best_degree, uncertainty = self.find_optimal_degree()
        self.last_best_degree = best_degree

        if log_path:
            self._log_navigation_results(frame_number, best_degree, uncertainty, log_path)

        elapsed = time.time() - start_time
        if debug_print:
            print(f"Best_Degree: {best_degree:.2f}°, Uncertainty: {uncertainty:.4f}, Time: {elapsed:.3f}s")

        angle_rad = math.radians(-best_degree)
        if return_timing:
            return angle_rad, elapsed
        return angle_rad

    # ── Angle selection ──────────────────────────────────────────────────────

    def find_optimal_degree(self):
        """
        Find the optimal heading from novelty scores.

        Groups minimum-novelty angles, selects the longest consecutive group
        (closest to 0° on ties), and returns the group mean with an uncertainty metric.

        If self.degree_strategy is set, delegates to it instead (see
        insect_nav/degree_strategies/ for drop-in alternatives benchmarked
        against this default in insect_nav/tuning/degree_strategy_eval.py).
        """
        if self.degree_strategy is not None:
            prev_degree = getattr(self, "last_best_degree", None)
            return self.degree_strategy(
                self.degree_array, self.novelty_array, self.params["DEGREES_PER_SHIFT"], prev_degree,
            )

        min_value = min(self.novelty_array)
        deg_min = [d for d, v in zip(self.degree_array, self.novelty_array) if v == min_value]

        step = self.params["DEGREES_PER_SHIFT"]
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

        optimal_degree = float(np.mean(best_group))
        indecision_a = 1 - len(best_group) / len(deg_min)
        indecision_b = (len(best_group) - 1) / len(self.degree_array)
        uncertainty = (indecision_a + indecision_b) / 2

        return optimal_degree, uncertainty

    # ── Visualization ────────────────────────────────────────────────────────

    def saveFiguresToCsv(self, frame, frame_number, output_path):
        saveFrameAsPNG(frame, output_dir=os.path.join(output_path, f"frame_{frame_number}"),
                       frame_name="1_original_frame")

        import cv2
        cropped = cropFrame(frame, self.params["CROP_BOTTOM"], self.params["CROP_TOP"])
        saveFrameAsPNG(cropped, output_dir=os.path.join(output_path, f"frame_{frame_number}"),
                       frame_name="2_cropped_frame")

        preprocessed = preprocessFrame(frame, 0, self.params)
        scale = (10 * self.params["WIDTH"], 10 * self.params["HEIGHT"])
        preprocessed_resized = cv2.resize(preprocessed, scale, interpolation=cv2.INTER_NEAREST)
        saveFrameAsPNG(preprocessed_resized, output_dir=os.path.join(output_path, f"frame_{frame_number}"),
                       frame_name="3_prepro_frame")

        preprocessed_vertical = visualize_vertical_weighting(preprocessed, self.params)
        preprocessed_vertical_resized = cv2.resize(preprocessed_vertical, scale, interpolation=cv2.INTER_NEAREST)
        saveFrameAsPNG(preprocessed_vertical_resized, output_dir=os.path.join(output_path, f"frame_{frame_number}"),
                       frame_name="4_prepro_vertical")
        saveVerticalWeightingHeatmap(
            preprocessed_vertical_resized, self.params,
            output_dir=os.path.join(output_path, f"frame_{frame_number}"),
            frame_name=f"5_prepro_vertical_heatmap_{self.params['VERTICAL_WEIGHT']}",
        )

        saveFeaturesAsPNG(preprocessed, self.params,
                          output_dir=os.path.join(output_path, f"frame_{frame_number}"))

    def plot_test_results(self, frame_number, output_path, optimal_degree=None):
        """
        Args:
            optimal_degree: direzione "vera" (in gradi, stessa convenzione
                dell'asse Shift Degree) verso cui il robot dovrebbe girare per
                puntare alla traiettoria di training, calcolata dal chiamante
                a partire da posa attuale + traiettoria (la rete non conosce
                nessuna delle due). None (default) = nessuna riga di
                riferimento disegnata (es. nessuna traiettoria di training
                disponibile).
        """
        frame_dir = os.path.join(output_path, f"frame_{frame_number}")
        os.makedirs(frame_dir, exist_ok=True)

        fig, ax = new_figure("error_vs_x")
        ax.scatter(self.degree_array, self.novelty_array, s=80, color=COLORS["actual"],
                   edgecolor="black", linewidth=1.2)

        has_reference_lines = False
        if hasattr(self, "last_best_degree"):
            ax.axvline(self.last_best_degree, color="red", linestyle="--",
                       linewidth=2, label=f"Scelta dalla rete ({self.last_best_degree:.1f}°)")
            has_reference_lines = True
        if optimal_degree is not None:
            ax.axvline(optimal_degree, color="limegreen", linestyle="--",
                       linewidth=2, label=f"Direzione ottimale ({optimal_degree:.1f}°)")
            has_reference_lines = True

        style_axes(ax, xlabel="Shift Degree (°)", ylabel="Novelty", title=f"Frame {frame_number}")
        if has_reference_lines:
            add_legend(ax)
        save_figure(fig, os.path.join(frame_dir, "novelty_plot.png"))
        plt.close(fig)

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log_navigation_results(self, frame_number, best_degree, uncertainty, log_path):
        os.makedirs(log_path, exist_ok=True)
        csv_path = os.path.join(log_path, "test_log.csv")
        file_exists = os.path.isfile(csv_path)

        header = ["frame_number", "best_degree", "uncertainty"] + \
                 [f"degree_{int(d)}" for d in self.degree_array]
        row = [frame_number, best_degree, uncertainty] + self.novelty_array

        with open(csv_path, "a" if file_exists else "w", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
