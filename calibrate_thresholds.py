
import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

try:
    from posture_monitor import extract_features
except ImportError as exc:
    raise SystemExit(
        "Could not import extract_features from posture_monitor.py. "
        "Place calibrate_thresholds.py beside posture_monitor.py."
    ) from exc


FEATURES = ("torso_angle", "head_ratio", "shoulder_tilt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate posture thresholds from labeled upright/slouched video segments."
    )
    parser.add_argument("--source", required=True,
                        help="Video path or webcam index (e.g. 0)")
    parser.add_argument("--upright-start", type=float, required=True)
    parser.add_argument("--upright-end", type=float, required=True)
    parser.add_argument("--slouched-start", type=float, required=True)
    parser.add_argument("--slouched-end", type=float, required=True)
    parser.add_argument("--out", default="calibration_report.json")
    parser.add_argument("--visibility", type=float, default=0.35,
                        help="Minimum landmark visibility; default: 0.35")
    return parser.parse_args()


def validate_segments(args):
    segments = [
        (args.upright_start, args.upright_end, "upright"),
        (args.slouched_start, args.slouched_end, "slouched"),
    ]
    for start, end, name in segments:
        if start < 0 or end <= start:
            raise SystemExit(f"Invalid {name} segment: require 0 <= start < end.")
    if max(args.upright_start, args.slouched_start) < min(
        args.upright_end, args.slouched_end
    ):
        raise SystemExit("Upright and slouched segments must not overlap.")


def robust_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    q10, q90 = np.percentile(arr, [10, 90])
    return {
        "count": int(arr.size),
        "median": median,
        "mad": mad,
        "p10": float(q10),
        "p90": float(q90),
    }


def estimate_thresholds(upright_values, slouched_values):
    """
    Assumes larger feature values indicate worse posture.

    The midpoint of the two medians is the candidate decision boundary.
    A small band around it provides separate entry/exit thresholds.
    """
    u = robust_stats(upright_values)
    s = robust_stats(slouched_values)

    separation = s["median"] - u["median"]
    if separation <= 0:
        return {
            "upright": u,
            "slouched": s,
            "separation": separation,
            "separable_by_median": False,
            "good_threshold": None,
            "bad_threshold": None,
            "warning": (
                "Slouched median is not higher than upright median. "
                "This feature does not separate these segments in the expected direction."
            ),
        }

    midpoint = (u["median"] + s["median"]) / 2.0
    # Hysteresis band is 10% of the median separation on each side.
    band = separation * 0.10
    good_threshold = midpoint - band
    bad_threshold = midpoint + band

    # Check whether the central 80% ranges overlap.
    overlap = not (u["p90"] < s["p10"])
    return {
        "upright": u,
        "slouched": s,
        "separation": float(separation),
        "separable_by_median": True,
        "central_ranges_overlap": bool(overlap),
        "candidate_midpoint": float(midpoint),
        "good_threshold": float(good_threshold),
        "bad_threshold": float(bad_threshold),
        "hysteresis_band_each_side": float(band),
        "warning": (
            "The central 80% ranges overlap; this feature alone may not reliably distinguish the two postures."
            if overlap else None
        ),
    }


def main():
    args = parse_args()
    validate_segments(args)

    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Could not open source: {args.source}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    mp_pose = mp.solutions.pose
    samples = {
        "upright": {name: [] for name in FEATURES},
        "slouched": {name: [] for name in FEATURES},
    }
    frame_counts = {"upright": 0, "slouched": 0}
    valid_counts = {"upright": 0, "slouched": 0}
    frame_index = 0

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp = frame_index / fps
                frame_index += 1

                if args.upright_start <= timestamp < args.upright_end:
                    label = "upright"
                elif args.slouched_start <= timestamp < args.slouched_end:
                    label = "slouched"
                else:
                    continue

                frame_counts[label] += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if not result.pose_landmarks or not result.pose_world_landmarks:
                    continue

                features = extract_features(
                    result.pose_world_landmarks.landmark,
                    result.pose_landmarks.landmark,
                )
                if features is None:
                    continue

                valid_counts[label] += 1
                for name in FEATURES:
                    value = features.get(name)
                    if value is not None and np.isfinite(value):
                        samples[label][name].append(float(value))

    finally:
        cap.release()

    report = {
        "source": args.source,
        "fps_used_for_timestamps": float(fps),
        "segments_seconds": {
            "upright": [args.upright_start, args.upright_end],
            "slouched": [args.slouched_start, args.slouched_end],
        },
        "frame_counts": frame_counts,
        "valid_pose_counts": valid_counts,
        "features": {},
        "notes": [
            "Candidate thresholds are estimated from this recording only.",
            "The midpoint between segment medians is a starting decision boundary, not a validated universal threshold.",
            "Good and bad thresholds form a hysteresis band; confirm their behavior by replaying the recording.",
            "Shoulder tilt may be unavailable in side-view footage when both shoulders are not visible.",
        ],
    }

    print("\nCalibration results")
    print("===================")
    for name in FEATURES:
        upright_values = samples["upright"][name]
        slouched_values = samples["slouched"][name]

        if len(upright_values) < 5 or len(slouched_values) < 5:
            report["features"][name] = {
                "error": "Not enough valid samples in both segments (need at least 5 each).",
                "upright_sample_count": len(upright_values),
                "slouched_sample_count": len(slouched_values),
            }
            print(f"\n{name}: INSUFFICIENT DATA "
                  f"(upright={len(upright_values)}, slouched={len(slouched_values)})")
            continue

        result = estimate_thresholds(upright_values, slouched_values)
        report["features"][name] = result

        print(f"\n{name}")
        print(f"  Upright:  n={result['upright']['count']}, "
              f"median={result['upright']['median']:.4f}, "
              f"MAD={result['upright']['mad']:.4f}, "
              f"P10-P90={result['upright']['p10']:.4f}–{result['upright']['p90']:.4f}")
        print(f"  Slouched: n={result['slouched']['count']}, "
              f"median={result['slouched']['median']:.4f}, "
              f"MAD={result['slouched']['mad']:.4f}, "
              f"P10-P90={result['slouched']['p10']:.4f}–{result['slouched']['p90']:.4f}")

        if result["separable_by_median"]:
            print(f"  Candidate GOOD threshold: {result['good_threshold']:.4f}")
            print(f"  Candidate BAD threshold:  {result['bad_threshold']:.4f}")
        else:
            print(f"  WARNING: {result['warning']}")

        if result.get("warning"):
            print(f"  WARNING: {result['warning']}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report to: {out_path.resolve()}")
    print("\nUse the candidate values as starting points, then validate them on a separate recording.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
