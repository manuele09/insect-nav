"""Visual preprocessing and feature extraction pipeline."""

import glob
import os

import cv2
import numpy as np


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



def visualize_vertical_weighting(
    preprocessed_frame: np.ndarray,
    parameters_dict: dict,
    out_dtype=np.uint8,
) -> np.ndarray:
    """
    Build an image showing the effect of VERTICAL_WEIGHT on each pixel.

    Each pixel is multiplied by (1 + VERTICAL_WEIGHT * h) where h=0 at the
    bottom row and increases upward. The result is gain-normalised to 255.
    """
    if preprocessed_frame is None:
        raise ValueError("preprocessed_frame is None.")
    if preprocessed_frame.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {preprocessed_frame.shape}.")

    img = preprocessed_frame.astype(np.float32, copy=False)
    H = img.shape[0]
    w = float(parameters_dict.get("VERTICAL_WEIGHT", 0.0))

    h = (H - 1) - np.arange(H, dtype=np.float32)  # h=0 at bottom row
    weights = (1.0 + w * h)[:, None]               # broadcast over columns
    weighted = img * weights

    max_val = float(np.max(weighted))
    if max_val > 0:
        weighted = weighted * (255.0 / max_val)
    else:
        weighted = np.zeros_like(weighted)

    weighted = np.clip(weighted, 0.0, 255.0)
    if out_dtype is not None:
        weighted = weighted.astype(out_dtype)
    return weighted


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
