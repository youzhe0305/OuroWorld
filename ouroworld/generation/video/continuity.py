"""Start-continuity check of a generated loop.

A video model occasionally opens with a global jump (zoom, shift or roll)
between its first two frames. The loop starts on the reference image, so such
a jump would be visible at every seam; the video is rejected instead. The
check fits a similarity transform to ORB matches between frames 0 and 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

MIN_MATCHES = 20


@dataclass(frozen=True)
class ContinuityLimits:
    """Largest accepted change between the first two frames.

    Attributes:
        max_scale_change_percent: Zoom, in percent.
        max_translation_pixels: Displacement of the image centre.
        max_rotation_degrees: In-plane rotation.
    """

    max_scale_change_percent: float
    max_translation_pixels: float
    max_rotation_degrees: float


def start_continuity(
    first: np.ndarray, second: np.ndarray, limits: ContinuityLimits
) -> dict[str, Any]:
    """Measure the global transform between two ``(H, W)`` uint8 grey frames.

    Returns:
        The measured scale, translation and rotation, with ``passed``; a frame
        pair without enough texture to measure fails.
    """
    import cv2

    orb = cv2.ORB_create(  # type: ignore[attr-defined]
        nfeatures=10000, scaleFactor=1.1, nlevels=12, edgeThreshold=15
    )
    key0, desc0 = orb.detectAndCompute(first, None)
    key1, desc1 = orb.detectAndCompute(second, None)
    if desc0 is None or desc1 is None:
        return {"passed": False, "reason": "insufficient visual features"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc0, desc1, k=2)
    good = [
        pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
    ]
    if len(good) < MIN_MATCHES:
        return {"passed": False, "reason": "insufficient feature matches", "matches": len(good)}
    source = np.array([key0[match.queryIdx].pt for match in good], dtype=np.float32)
    target = np.array([key1[match.trainIdx].pt for match in good], dtype=np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=5000,
        confidence=0.999,
    )
    if matrix is None or inliers is None:
        return {"passed": False, "reason": "no global transform", "matches": len(good)}
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    height, width = first.shape
    centre = np.array([width / 2.0, height / 2.0, 1.0])
    translation = float(np.linalg.norm(matrix @ centre - centre[:2]))
    scale_change = abs(scale - 1.0) * 100.0
    return {
        "passed": bool(
            scale_change <= limits.max_scale_change_percent
            and translation <= limits.max_translation_pixels
            and abs(rotation) <= limits.max_rotation_degrees
        ),
        "scale_change_percent": scale_change,
        "translation_pixels": translation,
        "rotation_degrees": abs(rotation),
        "matches": len(good),
        "inliers": int(inliers.sum()),
    }
