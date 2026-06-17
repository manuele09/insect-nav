import os

import numpy as np

from insect_nav.base import NeuralModelBase
from insect_nav.parameters import save_parameters_to_file


class PerfectMemory(NeuralModelBase):
    """
    Snapshot-based memory model for visual navigation (baseline).

    Stores feature vectors during training and computes novelty as the
    minimum mean absolute error against all stored views.
    """

    def __init__(self, parameters, load_net=False, num_shifts=None):
        self.training_views = []
        super().__init__(parameters, load_net, num_shifts)

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, frame):
        self.training_views.append(self.input_pipeline(frame, 0))

    # ── Weight persistence ───────────────────────────────────────────────────

    def save_weights(self):
        if not self.training_views:
            print("[PerfectMemory] Warning: No training views to save.")
            return
        os.makedirs(self.params["weightsPath"], exist_ok=True)
        np.savez_compressed(
            os.path.join(self.params["weightsPath"], "training_data.npz"),
            features=np.array(self.training_views),
        )
        print(f"[PerfectMemory] Saved {len(self.training_views)} training views.")

    def load_weights(self):
        if self.load_net:
            data = np.load(os.path.join(self.params["weightsPath"], "training_data.npz"))
            self.training_views = list(data["features"])
            print(f"[PerfectMemory] Loaded {len(self.training_views)} stored views.")

    # ── Testing ──────────────────────────────────────────────────────────────

    def test(self, frame, shift_degree=0):
        features = self.input_pipeline(frame, shift_degree)
        return self.compute_perfect_memory_score(features)

    def compute_perfect_memory_score(self, current_view):
        """Minimum MAE between current_view and all stored training views."""
        if not len(self.training_views):
            print("[PerfectMemory] Warning: No stored memories found.")
            return float("inf")

        current_flat = current_view.flatten()
        return float(min(
            np.mean(np.abs(stored.flatten() - current_flat))
            for stored in self.training_views
        ))
