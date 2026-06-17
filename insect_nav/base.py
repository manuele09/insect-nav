import csv
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from insect_nav.vision import (
    cropFrame,
    extractFeatures,
    preprocessFrame,
    saveFeaturesAsPNG,
    saveFrameAsPNG,
)


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

    # ── Navigation evaluation ────────────────────────────────────────────────

    def testNavigation(self, frame, frame_number=-1, log_path=None, debug_print=True):
        """
        Scan over angular shifts, find the minimum-novelty heading, and return
        the corresponding turning command in radians.
        """
        start_time = time.time()
        self.degree_array.clear()
        self.novelty_array.clear()

        for k in range(self.num_shifts + 1):
            shift = (-self.num_shifts / 2 + k) * self.params["DEGREES_PER_SHIFT"]
            angle = (shift + 180) % 360 - 180
            novelty = self.test(frame, angle)
            self.degree_array.append(angle)
            self.novelty_array.append(novelty)

        best_degree, uncertainty = self.find_optimal_degree()

        if log_path:
            self._log_navigation_results(frame_number, best_degree, uncertainty, log_path)

        elapsed = time.time() - start_time
        if debug_print:
            print(f"Best_Degree: {best_degree:.2f}°, Uncertainty: {uncertainty:.4f}, Time: {elapsed:.3f}s")

        return math.radians(-best_degree)

    # ── Angle selection ──────────────────────────────────────────────────────

    def find_optimal_degree(self):
        """
        Find the optimal heading from novelty scores.

        Groups minimum-novelty angles, selects the longest consecutive group
        (closest to 0° on ties), and returns the group mean with an uncertainty metric.
        """
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
        preprocessed_resized = cv2.resize(
            preprocessed,
            (10 * self.params["WIDTH"], 10 * self.params["HEIGHT"]),
            interpolation=cv2.INTER_NEAREST,
        )
        saveFrameAsPNG(preprocessed_resized, output_dir=os.path.join(output_path, f"frame_{frame_number}"),
                       frame_name="3_prepro_frame")
        saveFeaturesAsPNG(preprocessed, self.params,
                          output_dir=os.path.join(output_path, f"frame_{frame_number}"))

    def plot_test_results(self, frame_number, output_path):
        frame_dir = os.path.join(output_path, f"frame_{frame_number}")
        os.makedirs(frame_dir, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.scatter(self.degree_array, self.novelty_array, s=80, color="blue",
                    edgecolor="black", linewidth=1.2)
        plt.title(f"Frame {frame_number}", fontsize=20)
        plt.xlabel("Shift Degree (°)", fontsize=16)
        plt.ylabel("Novelty", fontsize=16)
        plt.grid(True, linestyle="-", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(frame_dir, "novelty_plot.png"), dpi=300)
        plt.close()

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
