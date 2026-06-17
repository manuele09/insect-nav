"""Novelty metrics on feature vectors."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


def novelty_scores(current: np.ndarray, references: list) -> dict:
    """
    Compute how different `current` is from a list of reference feature vectors.

    For each metric, returns the minimum distance between `current` and any
    vector in `references` (i.e. similarity to the closest known view).

    Args:
        current:    1-D feature vector of the current view.
        references: List of 1-D feature vectors seen previously.

    Returns:
        dict with keys:
            "cosine"    — in [0, 1]; 0 = identical direction
            "pearson"   — in [0, 1]; 0 = perfectly correlated profile
            "euclidean" — L2 distance; 0 = identical vector
        All values are 0 when `references` is empty.
    """
    scores = {"cosine": 0.0, "pearson": 0.0, "euclidean": 0.0}
    if not references:
        return scores

    cosine_dissimilarities = [
        (1 - cosine_similarity([current], [ref])[0][0]) / 2
        for ref in references
    ]
    scores["cosine"] = float(np.min(cosine_dissimilarities))

    correlations = [np.corrcoef(current, ref)[0, 1] for ref in references]
    scores["pearson"] = float((1 - np.max(correlations)) / 2)

    distances = cdist([current], references, metric="euclidean")[0]
    scores["euclidean"] = float(distances.min())

    return scores
