"""Visual preprocessing and feature extraction pipeline."""

import glob
import os

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


def loadFrame(frame_number: int, frames_dir: str) -> np.ndarray:
    frame_path = os.path.join(frames_dir, f"frame_{frame_number:06d}.png")
    return cv2.imread(frame_path)


def countFrames(frames_dir: str) -> int:
    pattern = os.path.join(frames_dir, "frame_*.png")
    return len(glob.glob(pattern))


def resizeFrame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def cropFrame(frame: np.ndarray, crop_bottom: int, crop_top: int) -> np.ndarray:
    """Remove pixels from the top and bottom of the frame."""
    if frame is None:
        raise ValueError("Input frame is None.")
    height = frame.shape[0]
    return frame[crop_top:(height - crop_bottom), :]


def shiftFrame(image: np.ndarray, degrees: float) -> np.ndarray:
    """Shift an image horizontally by the given angular amount (wrap-around)."""
    rows, cols = image.shape[:2]
    x_shift = int((degrees / 360.0) * cols)
    M = np.float32([[1, 0, x_shift], [0, 1, 0]])
    return cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_WRAP)


def preprocessFrame(frame: np.ndarray, shift_degrees: float, parameters_dict: dict) -> np.ndarray:
    """
    Full preprocessing pipeline: crop → grayscale+invert → shift → resize.
    """
    cropped = cropFrame(frame, parameters_dict["CROP_BOTTOM"], parameters_dict["CROP_TOP"])
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inverted = 255 - gray
    shifted = shiftFrame(inverted, shift_degrees)
    return resizeFrame(shifted, parameters_dict["WIDTH"], parameters_dict["HEIGHT"])


def extractFeatures(frame: np.ndarray, parameters_dict: dict) -> np.ndarray:
    """
    Compute weighted vertical and/or horizontal pixel distributions.

    Returns a 1-D feature vector (concatenation of selected distributions).
    """
    if not (parameters_dict["USE_VERTICAL_DIST"] or parameters_dict["USE_HORIZONTAL_DIST"]):
        return frame

    features = []

    if parameters_dict["USE_VERTICAL_DIST"]:
        row_weights = [1 + parameters_dict["VERTICAL_WEIGHT"] * i for i in range(frame.shape[0])]
        features.append(np.average(frame, axis=0, weights=row_weights))

    if parameters_dict["USE_HORIZONTAL_DIST"]:
        col_weights = [1 + parameters_dict["HORIZONTAL_WEIGHT"] * i for i in range(frame.shape[1])]
        features.append(np.average(frame, axis=1, weights=col_weights))

    return np.concatenate(features) if features else np.array([])


def computeNovelty(current_image: np.ndarray, previous_images: list) -> dict:
    """Return cosine, Pearson, and Euclidean novelty scores vs. a list of stored images."""
    novelties = {"cosine": 0.0, "pearson": 0.0, "euclidean": 0.0}
    if not previous_images:
        return novelties

    cosine_scores = [
        (1 - cosine_similarity([current_image], [prev])[0][0]) / 2
        for prev in previous_images
    ]
    novelties["cosine"] = float(np.min(cosine_scores))

    correlations = [np.corrcoef(current_image, prev)[0, 1] for prev in previous_images]
    novelties["pearson"] = float((1 - np.max(correlations)) / 2)

    distances = cdist([current_image], previous_images, metric="euclidean")[0]
    novelties["euclidean"] = float(distances.min())

    return novelties


def saveFrameAsPNG(frame: np.ndarray, output_dir: str = "SavedFrames",
                   frame_name: str = "original_frame") -> None:
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, frame_name + ".png"), frame)


def saveFeaturesAsPNG(frame: np.ndarray, parameters_dict: dict,
                      output_dir: str = "SavedFeatures") -> None:
    os.makedirs(output_dir, exist_ok=True)
    features = extractFeatures(frame, parameters_dict)

    if parameters_dict["USE_VERTICAL_DIST"]:
        vertical_dist = features[:parameters_dict["WIDTH"]]
        v_norm = cv2.normalize(vertical_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        v_img = np.repeat(np.repeat(np.expand_dims(v_norm, axis=0), 240, axis=0), 30, axis=1)
        cv2.imwrite(os.path.join(output_dir, "vertical_dist.png"), v_img)

    if parameters_dict["USE_HORIZONTAL_DIST"]:
        horizontal_dist = features[parameters_dict["WIDTH"]:]
        h_norm = cv2.normalize(horizontal_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        h_img = np.repeat(np.repeat(np.expand_dims(h_norm, axis=1), 30, axis=0), 300, axis=1)
        cv2.imwrite(os.path.join(output_dir, "horizontal_dist.png"), h_img)
