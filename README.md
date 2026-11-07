# Seated Posture Monitor

A lightweight, CPU-only posture monitoring CLI built using **Python,
OpenCV, and MediaPipe**. It analyzes webcam or video input, extracts
upper-body pose landmarks, classifies posture over time, and generates a
JSON report containing posture statistics.

## Features

-   **Pose Detection:** Uses MediaPipe to detect upper-body landmarks.
-   **Geometric Analysis:** Evaluates torso inclination, forward head
    position, and shoulder alignment.
-   **Posture Classification:** Classifies posture as good or bad using
    configurable geometric thresholds.
-   **Temporal Smoothing:** Uses Exponential Moving Average (EMA),
    hysteresis, and minimum confirmation durations to reduce unstable
    classifications.
-   **Session Analytics:** Tracks good/bad posture duration, posture
    breaks, observation coverage, and an overall posture score.
-   **Robust Processing:** Handles invalid landmarks, dropped frames,
    and video processing errors.
-   **JSON Output:** Exports session statistics and diagnostic
    information.

## Tech Stack

-   Python
-   OpenCV
-   MediaPipe
-   NumPy

## Installation

Clone the repository:

``` bash
git clone https://github.com/simranbajaj06/Seated-Posture-Monitor.git
cd posture-monitor
```

Create and activate a virtual environment:

``` bash
python -m venv venv
```

Install dependencies:

``` bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

Monitor a video file:

``` bash
python posture_monitor.py --source test.mp4 --output posture.json
```

Monitor a webcam:

``` bash
python posture_monitor.py --source 0 --output posture.json
```

Use the CLI help to see available options:

``` bash
python posture_monitor.py --help
```

## How It Works

1.  **Frame Processing:** OpenCV reads frames from the webcam or video
    file.
2.  **Landmark Extraction:** MediaPipe detects pose landmarks, including
    the nose, shoulders, and hips.
3.  **Feature Extraction:** Geometric features are calculated to
    estimate torso inclination, head position, and shoulder alignment.
4.  **Classification:** Features are compared against configurable
    thresholds to determine posture.
5.  **Temporal Analysis:** EMA smoothing and time-based state
    transitions help prevent rapid classification changes.
6.  **Report Generation:** Session statistics are saved as a JSON file.

## Posture Score

The posture score represents the percentage of classified observation
time spent in good posture:

`Posture Score = (Good Posture Duration / Observed Duration) × 100`

Unobserved intervals are excluded from this calculation. The score is a
heuristic indicator, not a medical assessment.

## Threshold Rationale

The geometric thresholds are heuristic starting points based on torso
inclination, normalized head position, and shoulder alignment.
Hysteresis uses separate thresholds for entering and leaving a
bad-posture state to reduce classification flickering. Minimum
confirmation durations help avoid treating brief movements as sustained
posture changes. These values are configurable and may require
calibration for different camera angles and users.

## Output

The generated `posture.json` contains:

-   Session duration and processing status
-   Total and valid pose frames
-   Good and bad posture durations
-   Observation coverage and unobserved duration
-   Good posture percentage
-   Number of posture breaks
-   Overall posture score
-   Processing diagnostics

## Limitations

-   Results depend on camera placement, visibility, and lighting.
-   Geometric thresholds are heuristic and are not clinically validated.
-   The system is designed for upper-body posture monitoring in a seated
    position.

------------------------------------------------------------------------

**Note:** This project is designed as a headless, CPU-only command-line
application. It does not require a graphical interface, web server,
Docker, or a separately trained machine learning model.
