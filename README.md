# Seated Posture Monitor

A lightweight, **CPU-only, headless CLI** that uses Python, OpenCV, and
MediaPipe to analyze seated upper-body posture from a webcam or video
and generate a JSON session report.

## Features

-   Detects upper-body landmarks using MediaPipe.
-   Extracts torso-angle and normalized head-position features.
-   Classifies posture over time using geometric thresholds.
-   Reduces flicker with EMA smoothing, hysteresis, and time-based state
    confirmation.
-   Reports good/bad posture duration, posture breaks, observation
    coverage, and a 0--100 score.
-   Handles invalid landmarks and dropped frames.

## Installation

``` bash
git clone https://github.com/simranbajaj06/Seated-Posture-Monitor.git
cd Seated-Posture-Monitor

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

Process a video:

``` bash
python posture_monitor.py --source test.mp4 --output posture.json
```

Use the webcam:

``` bash
python posture_monitor.py --source 0 --output posture.json
```

See available CLI options:

``` bash
python posture_monitor.py --help
```

## Approach and Threshold Selection

OpenCV reads frames and MediaPipe extracts nose, shoulder, and hip
landmarks. The script derives geometric posture features, classifies
them, and applies temporal smoothing and confirmation before accepting a
state change. A single frame therefore does not immediately count as a
posture break.

Thresholds were calibrated from `test4.mp4`, using 0--18 seconds as the
upright segment and 20--40 seconds as the slouched segment. For each
feature, the midpoint between the segment medians was used as the
initial boundary, with a hysteresis band of 10% of the median separation
on each side.

  Feature                   Bad posture   Good/recovery
  ----------------------- ------------- ---------------
  Torso angle               Below 5.24°     Above 6.05°
  Normalized head ratio     Below 0.100     Above 0.135

In this recording, both features had lower values in the slouched
segment, so the classifier uses that observed direction. Shoulder tilt
was excluded from primary classification because its upright and
slouched distributions overlapped. These are recording-specific
heuristics, not universal or clinically validated limits; results may
vary with the person, camera angle, and setup.

## Score and Output

`Posture score = good posture seconds / observed seconds × 100`

Unobserved intervals are excluded. The JSON report includes session
status and duration, frame counts, good/bad durations, observation
coverage, posture breaks, score, and diagnostic information.

## Limitations

-   Designed for a single seated person and upper-body monitoring.
-   Accuracy depends on camera position, visibility, lighting, and user
    orientation.
-   Thresholds were calibrated on a limited sample and need validation
    on other recordings.
-   The score is a heuristic summary, not a medical assessment.

**Scope:** No UI, web service, Docker, multi-person tracking, or
separately trained model.
