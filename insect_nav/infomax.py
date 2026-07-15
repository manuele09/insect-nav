import os

import numpy as np

from insect_nav.base import NeuralModelBase
from insect_nav.parameters import save_parameters_to_file
from insect_nav.vision import countFrames, loadFrame


class Infomax(NeuralModelBase):
    """
    Infomax unsupervised learning model for visual navigation.

    Learning rule: ΔW ∝ (W − (y + h) ⊗ (hW))
    where h = W·s, y = tanh(h), s = z-score normalized features.
    """

    def __init__(self, parameters, load_net=False, calculate_mean=True, num_shifts=None):
        self.M = parameters["output_units"]
        super().__init__(parameters, load_net, num_shifts)
        self.learning_rate = self.params["learning_rate"]

        if calculate_mean:
            self.calculate_mean_std()

    # ── Dataset statistics ───────────────────────────────────────────────────

    def calculate_mean_std(self):
        """Compute dataset-wide mean and std for z-score normalization."""
        print("[Infomax]: Calculating mean and std of dataset.")
        num_frames = countFrames(self.params["trainingDatasetPath"])
        feature_vectors = []

        for frame_number in range(0, num_frames, self.params.get("train_step", 1)):
            frame = loadFrame(frame_number, self.params["trainingDatasetPath"])
            features = self.input_pipeline(frame, 0, normalize=False)
            feature_vectors.append(features.flatten())

        data_matrix = np.array(feature_vectors)
        mean = float(np.mean(data_matrix))
        std = float(np.std(data_matrix) + 1e-8)

        self.params["training_dataset_mean"] = mean
        self.params["training_dataset_std"] = std
        print(f"[Infomax]: Mean = {mean:.6f}, Std = {std:.6f}")
        save_parameters_to_file(self.params, self.params["parameters_path"])

    # ── Input normalization ──────────────────────────────────────────────────

    def apply_z_score(self, features: np.ndarray) -> np.ndarray:
        mean = self.params["training_dataset_mean"]
        std = self.params["training_dataset_std"]
        return (features - mean) / std

    def input_pipeline(self, frame, shift_degrees, normalize=True):
        features = super().input_pipeline(frame, shift_degrees)
        if normalize:
            return self.apply_z_score(features.flatten())
        return features

    # ── Training and testing ─────────────────────────────────────────────────

    def train(self, frame, is_last_frame=False):
        s = self.input_pipeline(frame, 0)
        h = self.weights @ s
        y = np.tanh(h)
        proj = h @ self.weights
        correction = np.outer(y + h, proj)
        delta_w = (self.learning_rate / (self.M * self.input_neurons)) * (self.weights - correction)
        self.weights += delta_w

    def test(self, frame, shift_degree=0):
        s = self.input_pipeline(frame, shift_degree)
        h = self.weights @ s
        return float(np.sum(np.abs(h)))

    # ── Weight persistence ───────────────────────────────────────────────────

    def save_weights(self):
        os.makedirs(self.params["weightsPath"], exist_ok=True)
        np.save(os.path.join(self.params["weightsPath"], "weights.npy"), self.weights)

    def load_weights(self):
        if self.load_net:
            self.weights = np.load(os.path.join(self.params["weightsPath"], "weights.npy"))
            print(f"[Infomax]: Loaded weights from {self.params['weightsPath']}")
        else:
            w = np.random.uniform(-0.5, 0.5, (self.M, self.input_neurons))
            self.weights = (w - np.mean(w)) / np.std(w)
            print(f"[Infomax]: Initialized random weights ({self.M}x{self.input_neurons})")
