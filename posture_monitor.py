
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

# ---------------------------------------------------------------------------
# Feature smoothing
# ---------------------------------------------------------------------------

class FeatureSmoother:
    """Exponential moving average for numeric posture features."""

    def __init__(self, alpha: float = EMA_ALPHA):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EMA alpha must be in the interval (0, 1].")
        self.alpha = alpha
        self.values: Optional[Dict[str, Optional[float]]] = None

    def reset(self) -> None:
        self.values = None

    def update(
        self, features: Dict[str, Optional[float]]
    ) -> Dict[str, Optional[float]]:
        if self.values is None:
            self.values = features.copy()
            return self.values.copy()

        for key, value in features.items():
            if value is None:
                # If a feature is unavailable in this frame, preserve its
                # previous smoothed value rather than treating it as zero.
                continue

            previous = self.values.get(key)
            if previous is None:
                self.values[key] = float(value)
            else:
                self.values[key] = (
                    self.alpha * float(value)
                    + (1.0 - self.alpha) * float(previous)
                )

        return self.values.copy()


# ---------------------------------------------------------------------------
# Temporal classifier and time accounting
# ---------------------------------------------------------------------------

class TemporalPostureClassifier:
    """
    Stateful posture classifier.

    A candidate transition is held pending until it persists long enough.
    While pending, its time is not committed. If confirmed, the pending
    interval is attributed to the new state; if cancelled, it is attributed
    to the previous stable state.

    Initial posture is also confirmed over time. An initial bad episode is
    counted as a bad episode, but separately from breaks that occur after an
    observed good state.
    """

    def __init__(self):
        self.state: Optional[str] = None
        self.candidate: Optional[str] = None
        self.candidate_duration = 0.0
        self.last_timestamp: Optional[float] = None

        self.good_seconds = 0.0
        self.bad_seconds = 0.0
        self.observed_seconds = 0.0
        self.unobserved_gap_seconds = 0.0

        # Count each confirmed bad-posture episode, including one already
        # present when recording starts. This matches the session-level
        # interpretation of "how many poor-posture episodes occurred".
        self.posture_breaks = 0
        self.good_to_bad_transitions = 0

    def _add_time(self, state: Optional[str], seconds: float) -> None:
        if seconds <= 0 or state not in ("good", "bad"):
            return

        if state == "good":
            self.good_seconds += seconds
        else:
            self.bad_seconds += seconds

        self.observed_seconds += seconds

    def classify_candidate(
        self, features: Dict[str, Optional[float]]
    ) -> Optional[str]:
        torso = features.get("torso_angle")
        head = features.get("head_ratio")
        shoulder = features.get("shoulder_tilt")

        available = [
            value for value in (torso, head, shoulder) if value is not None
        ]
        if not available:
            return None

        # A feature only contributes when it is available.
        bad_evidence = (
            (torso is not None and torso > BAD_TORSO_ANGLE)
            or (head is not None and head > BAD_HEAD_RATIO)
            or (
                shoulder is not None
                and shoulder > BAD_SHOULDER_TILT
            )
        )

        good_evidence = (
            (torso is None or torso < GOOD_TORSO_ANGLE)
            and (head is None or head < GOOD_HEAD_RATIO)
            and (
                shoulder is None
                or shoulder < GOOD_SHOULDER_TILT
            )
        )

        if self.state == "bad":
            # Hysteresis: remain bad until there is clear recovery evidence.
            return "good" if good_evidence else "bad"

        if self.state == "good":
            # Hysteresis: remain good until a bad threshold is crossed.
            return "bad" if bad_evidence else "good"

        # No stable state yet. Require clear evidence to initialize.
        if bad_evidence:
            return "bad"
        if good_evidence:
            return "good"
        return None

    def _start_candidate(self, proposed: Optional[str]) -> None:
        self.candidate = proposed
        self.candidate_duration = 0.0

    def update(
        self,
        features: Dict[str, Optional[float]],
        timestamp: float,
    ) -> None:
        """
        Process one valid observation.

        Every short interval between valid observations is accounted for:
        - stable intervals go to the current state;
        - transition-candidate intervals are held temporarily;
        - if the transition is confirmed, held time goes to the new state;
        - if the transition is cancelled, held time goes to the old state.
        """
        timestamp = float(timestamp)
        proposed = self.classify_candidate(features)

        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            self._start_candidate(proposed)
            return

        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        if dt <= 0:
            return

        if dt > MAX_FRAME_GAP_SECONDS:
            # This interval is not reliable enough to infer posture.
            self.unobserved_gap_seconds += dt
            self._start_candidate(proposed)
            return

        # Initial state: wait for enough consecutive evidence before starting
        # the session's good/bad accounting.
        if self.state is None:
            if proposed is None:
                self._start_candidate(None)
                return

            if proposed != self.candidate:
                self._start_candidate(proposed)
                self.candidate_duration = dt
            else:
                self.candidate_duration += dt

            required = (
                BAD_CONFIRMATION_SECONDS
                if proposed == "bad"
                else GOOD_CONFIRMATION_SECONDS
            )

            if self.candidate_duration >= required:
                self.state = proposed
                self._add_time(self.state, self.candidate_duration)

                if self.state == "bad":
                    # The recording began in a confirmed bad posture.
                    self.posture_breaks += 1

                self._start_candidate(None)
            return

        # No clear evidence for either class: preserve the stable state and
        # treat this short interval as belonging to that state. Any previously
        # pending transition has been disproved.
        if proposed is None or proposed == self.state:
            if self.candidate is not None:
                self._add_time(self.state, self.candidate_duration)
                self._start_candidate(None)

            self._add_time(self.state, dt)
            return

        # We have evidence for the opposite state. Start or extend a pending
        # transition. Do not charge it until confirmed/cancelled.
        if proposed != self.candidate:
            if self.candidate is not None:
                # A different candidate replaced the old one; the old pending
                # interval was not confirmed, so assign it to stable state.
                self._add_time(self.state, self.candidate_duration)

            self._start_candidate(proposed)
            self.candidate_duration = dt
        else:
            self.candidate_duration += dt

        required = (
            BAD_CONFIRMATION_SECONDS
            if proposed == "bad"
            else GOOD_CONFIRMATION_SECONDS
        )

        if self.candidate_duration >= required:
            previous_state = self.state
            self.state = proposed

            # Count the full confirmed transition interval under the new state.
            self._add_time(self.state, self.candidate_duration)

            if previous_state == "good" and self.state == "bad":
                self.posture_breaks += 1
                self.good_to_bad_transitions += 1

            self._start_candidate(None)

    def finalize(self) -> None:
        """Commit any unconfirmed trailing interval to the stable state."""
        if self.candidate is not None and self.candidate_duration > 0:
            self._add_time(self.state, self.candidate_duration)
        self._start_candidate(None)

    def result(self) -> Optional[Dict[str, Union[float, int]]]:
        if self.observed_seconds <= 0:
            return None

        percentage = (
            self.good_seconds / self.observed_seconds
        ) * 100.0

        return {
            "good_posture_seconds": round(self.good_seconds, 2),
            "bad_posture_seconds": round(self.bad_seconds, 2),
            "observed_duration_seconds": round(self.observed_seconds, 2),
            "unobserved_gap_seconds": round(self.unobserved_gap_seconds, 2),
            "good_posture_percentage": round(percentage, 2),
            "posture_breaks": int(self.posture_breaks),
            "good_to_bad_transitions": int(self.good_to_bad_transitions),
            "posture_score": int(round(percentage)),
        }


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def _open_capture(source: Union[int, str]) -> cv2.VideoCapture:
    return cv2.VideoCapture(source)


def process_video(
    source: Union[int, str],
    duration: float,
    output: Union[str, Path],
    preview: bool = False,
) -> int:
    cap = _open_capture(source)

    if not cap.isOpened():
        print("ERROR: Unable to open video source.", file=sys.stderr)
        return 1

    is_file = isinstance(source, str)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    smoother = FeatureSmoother()
    classifier = TemporalPostureClassifier()

    pose_detected_frames = 0
    rejected_feature_frames = 0
    valid_pose_frames = 0
    total_read_errors = 0
    consecutive_read_errors = 0
    frame_count = 0

    wall_start = time.monotonic()
    stop_reason = "duration_reached"

    mp_pose = mp.solutions.pose

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            while True:
                # File timestamps are based on source frame index/FPS.
                # Webcam timestamps use monotonic wall-clock time.
                if is_file:
                    current_source_time = frame_count / fps
                    if current_source_time >= duration:
                        stop_reason = "duration_reached"
                        break
                else:
                    if time.monotonic() - wall_start >= duration:
                        stop_reason = "duration_reached"
                        break

                try:
                    success, frame = cap.read()
                except cv2.error:
                    total_read_errors += 1
                    consecutive_read_errors += 1

                    if is_file:
                        stop_reason = "video_read_error"
                        break

                    if consecutive_read_errors >= MAX_WEBCAM_READ_ERRORS:
                        stop_reason = "repeated_camera_read_errors"
                        break

                    continue

                if not success or frame is None:
                    if is_file:
                        # End-of-file is normal for a video file. A zero-frame
                        # or truncated file is rejected later if unusable.
                        stop_reason = "end_of_file"
                        break

                    total_read_errors += 1
                    consecutive_read_errors += 1

                    if consecutive_read_errors >= MAX_WEBCAM_READ_ERRORS:
                        stop_reason = "repeated_camera_read_errors"
                        break

                    continue

                consecutive_read_errors = 0
                frame_count += 1

                if (
                    frame.ndim != 3
                    or frame.shape[0] <= 0
                    or frame.shape[1] <= 0
                ):
                    total_read_errors += 1
                    continue

                timestamp = (
                    (frame_count - 1) / fps
                    if is_file
                    else time.monotonic() - wall_start
                )

                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    results = pose.process(rgb)

                    if not results.pose_world_landmarks:
                        if preview:
                            cv2.imshow("Posture Monitor", frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                stop_reason = "preview_stopped"
                                break
                        continue

                    pose_detected_frames += 1

                    world = results.pose_world_landmarks.landmark
                    image = (
                        results.pose_landmarks.landmark
                        if results.pose_landmarks
                        else None
                    )

                    features = extract_features(world, image)

                    if features is None:
                        rejected_feature_frames += 1
                        if preview:
                            cv2.imshow("Posture Monitor", frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                stop_reason = "preview_stopped"
                                break
                        continue

                    valid_pose_frames += 1
                    smoothed = smoother.update(features)
                    classifier.update(smoothed, timestamp)

                    if preview:
                        state_text = classifier.state or "calibrating"
                        color = (
                            (0, 200, 0)
                            if state_text == "good"
                            else (0, 0, 255)
                            if state_text == "bad"
                            else (0, 200, 255)
                        )

                        cv2.putText(
                            frame,
                            f"Posture: {state_text}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            color,
                            2,
                        )
                        cv2.putText(
                            frame,
                            (
                                f"Torso: {smoothed['torso_angle']:.1f} "
                                f"Head: {smoothed['head_ratio']:.2f}"
                            ),
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            2,
                        )
                        cv2.imshow("Posture Monitor", frame)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stop_reason = "preview_stopped"
                            break

                except (cv2.error, ValueError, TypeError, AttributeError):
                    # A single corrupt/unprocessable frame must not crash the
                    # complete session.
                    total_read_errors += 1
                    continue

    except Exception as exc:
        print(f"ERROR: Video processing failed: {exc}", file=sys.stderr)
        return 1

    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()

    if frame_count == 0:
        print("ERROR: No readable video frames.", file=sys.stderr)
        return 1

    if valid_pose_frames == 0:
        print(
            "ERROR: No usable person/landmark observations.",
            file=sys.stderr,
        )
        return 1

    classifier.finalize()
    metrics = classifier.result()

    if metrics is None or classifier.observed_seconds < MIN_USABLE_SECONDS:
        print(
            "ERROR: Insufficient continuous usable posture observations.",
            file=sys.stderr,
        )
        return 1

    if is_file:
        # Use the number of frames actually read, not the requested duration.
        actual_source_duration = frame_count / fps
    else:
        actual_source_duration = min(
            duration, max(0.0, time.monotonic() - wall_start)
        )

    unobserved_duration = max(
        0.0, actual_source_duration - classifier.observed_seconds
    )

    coverage_percentage = (
        classifier.observed_seconds / actual_source_duration * 100.0
        if actual_source_duration > 0
        else 0.0
    )

    output_data = {
        "status": "completed",
        "source": str(source),
        "session_duration_seconds": round(actual_source_duration, 2),
        "requested_duration_seconds": round(duration, 2),
        "video_fps": round(fps, 3),
        "total_frames_read": int(frame_count),
        "reported_video_frames": reported_frame_count if is_file else None,
        "pose_detected_frames": int(pose_detected_frames),
        "rejected_feature_frames": int(rejected_feature_frames),
        "valid_pose_frames": int(valid_pose_frames),
        "read_or_processing_errors": int(total_read_errors),
        "stop_reason": stop_reason,
        "coverage_percentage": round(coverage_percentage, 2),
        "unobserved_duration_seconds": round(unobserved_duration, 2),
        **metrics,
    }

    output_path = Path(output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=4)
    except OSError as exc:
        print(f"ERROR: Could not write JSON output: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(output_data, indent=4))

    # A very low coverage result is still a usable partial observation, but it
    # is made explicit in the JSON so downstream stages can reject it if
    # their own coverage requirement is stricter.
    return 0
