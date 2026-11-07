
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------------------------
# Tunable configuration
# ---------------------------------------------------------------------------

# Visibility is read from MediaPipe's image-space pose landmarks. World
# landmarks are used for geometry.
VISIBILITY_THRESHOLD = 0.35

# EMA smoothing. Higher values react faster; lower values smooth more.
EMA_ALPHA = 0.25

# Hysteresis thresholds. A bad state is entered at the "bad" thresholds and
# exited only after all available features fall below the "good" thresholds.
BAD_TORSO_ANGLE = 18.0
GOOD_TORSO_ANGLE = 12.0

BAD_HEAD_RATIO = 0.40
GOOD_HEAD_RATIO = 0.30

BAD_SHOULDER_TILT = 15.0
GOOD_SHOULDER_TILT = 10.0

# Require sustained evidence before changing stable state.
BAD_CONFIRMATION_SECONDS = 0.8
GOOD_CONFIRMATION_SECONDS = 1.0

# Do not bridge long gaps between usable observations. Gaps above this value
# are treated as unobserved and reset an in-progress transition candidate.
MAX_FRAME_GAP_SECONDS = 0.75

# Fail if less than this amount of classified time was observed.
MIN_USABLE_SECONDS = 1.0

# Consecutive webcam read failures tolerated before stopping.
MAX_WEBCAM_READ_ERRORS = 30


# ---------------------------------------------------------------------------
# Landmark feature extraction
# ---------------------------------------------------------------------------

def _visibility(landmark) -> float:
    """Return landmark visibility, defaulting to zero if unavailable."""
    value = getattr(landmark, "visibility", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_features(
    world_landmarks: Sequence,
    image_landmarks: Optional[Sequence] = None,
) -> Optional[Dict[str, Optional[float]]]:
    """
    Extract scale-normalized posture features.

    The torso is estimated from shoulder/hip midpoints when both sides are
    visible. If one side is occluded (common in side-view video), the best
    visible same-side shoulder/hip pair is used.

    Returns None when the nose or a usable shoulder/hip pair is unavailable.
    Shoulder tilt is optional because it cannot be measured reliably from a
    single visible side.
    """
    if len(world_landmarks) <= 24:
        return None

    # Use image-space visibility where available. World landmark visibility
    # is used as a fallback.
    visibility_landmarks = (
        image_landmarks if image_landmarks is not None else world_landmarks
    )

    if len(visibility_landmarks) <= 24:
        return None

    nose_w = world_landmarks[0]
    nose_vis = _visibility(visibility_landmarks[0])

    left_sh_w = world_landmarks[11]
    right_sh_w = world_landmarks[12]
    left_hip_w = world_landmarks[23]
    right_hip_w = world_landmarks[24]

    left_sh_vis = _visibility(visibility_landmarks[11])
    right_sh_vis = _visibility(visibility_landmarks[12])
    left_hip_vis = _visibility(visibility_landmarks[23])
    right_hip_vis = _visibility(visibility_landmarks[24])

    # Nose is needed for the head displacement feature.
    if nose_vis < VISIBILITY_THRESHOLD:
        return None

    left_pair_quality = min(left_sh_vis, left_hip_vis)
    right_pair_quality = min(right_sh_vis, right_hip_vis)

    left_ok = left_pair_quality >= VISIBILITY_THRESHOLD
    right_ok = right_pair_quality >= VISIBILITY_THRESHOLD

    if not left_ok and not right_ok:
        return None

    def xyz(lm) -> np.ndarray:
        return np.array([lm.x, lm.y, lm.z], dtype=np.float64)

    # Prefer a midpoint when both sides are visible; otherwise use the
    # highest-confidence same-side pair.
    if left_ok and right_ok:
        shoulder = (xyz(left_sh_w) + xyz(right_sh_w)) / 2.0
        hip = (xyz(left_hip_w) + xyz(right_hip_w)) / 2.0
        used_both_sides = True
    elif left_ok and (
        not right_ok or left_pair_quality >= right_pair_quality
    ):
        shoulder = xyz(left_sh_w)
        hip = xyz(left_hip_w)
        used_both_sides = False
    else:
        shoulder = xyz(right_sh_w)
        hip = xyz(right_hip_w)
        used_both_sides = False

    torso_vector = shoulder - hip
    torso_length = float(np.linalg.norm(torso_vector))

    if not np.isfinite(torso_length) or torso_length < 1e-6:
        return None

    dx, dy, dz = torso_vector

    # Angle relative to the vertical axis. Ratios of coordinate differences
    # make this independent of overall body/image scale.
    torso_angle = math.degrees(
        math.atan2(math.sqrt(dx * dx + dz * dz), abs(dy) + 1e-9)
    )

    # Estimated head displacement along MediaPipe's relative depth axis,
    # normalized by torso length. This is a heuristic, not a clinical measure.
    head_ratio = abs(float(nose_w.z) - float(shoulder[2])) / torso_length

    shoulder_tilt = None
    if (
        used_both_sides
        and left_sh_vis >= VISIBILITY_THRESHOLD
        and right_sh_vis >= VISIBILITY_THRESHOLD
    ):
        shoulder_width = abs(float(left_sh_w.x) - float(right_sh_w.x))
        if shoulder_width >= 1e-6:
            shoulder_tilt = math.degrees(
                math.atan2(
                    abs(float(left_sh_w.y) - float(right_sh_w.y)),
                    shoulder_width,
                )
            )

    values = [torso_angle, head_ratio]
    if shoulder_tilt is not None:
        values.append(shoulder_tilt)

    if not all(np.isfinite(v) for v in values):
        return None

    return {
        "torso_angle": float(torso_angle),
        "head_ratio": float(head_ratio),
        "shoulder_tilt": (
            float(shoulder_tilt) if shoulder_tilt is not None else None
        ),
    }

