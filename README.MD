# 🐙 Octopus Behavioral Analysis
### A Computer Vision Pipeline for Octopus Behavior Understanding

> **GSoC 2026 Entry Task** — Extracting interpretable behavioral signals from video data using pure classical computer vision, without specialized models such as C3D, SlowFast, or I3D.

---
## 💬 MY Note

While this entry task focuses on simple and interpretable features, I am comfortable working with more advanced approaches, including deep learning and multi-modal modeling. I chose to keep the implementation straightforward to align with the task requirements and ensure clarity and reproducibility.

If given the opportunity to work on the full project, I would be excited to extend this pipeline with more sophisticated techniques — spatiotemporal models, improved feature extraction, and multi-modal integration — to achieve higher accuracy and deeper behavioral understanding.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Extracted Features](#-extracted-features)
- [Behavioral Interpretation Framework](#-behavioral-interpretation-framework)
- [Results & Visualizations](#-results--visualizations)
- [Limitations](#-limitations)
- [Future Work & Extensions](#-future-work--extensions)
- [Author's Note](#-authors-note)
- [References](#-references)

---

## 🔬 Project Overview

This project presents a simple, interpretable computer vision pipeline for analyzing octopus behavior from video data. Rather than relying on black-box deep learning models, the approach uses **pure OpenCV and NumPy** to extract two complementary signals per frame — motion magnitude and contour area — and overlays them on a shared time axis to reveal behavioral patterns.

**Core Research Questions:**
- How much is the octopus moving at any given moment?
- What is the animal's body posture (expanded vs. contracted)?
- Can we identify distinct behavioral states — such as resting, active exploration, or stress responses — without any learned model?

**Design Philosophy:**
Behavioral neuroscience and ethology benefit most from methods where the link between raw sensor data and behavioral inference is transparent and reproducible. This pipeline prioritizes interpretability so that biologists and researchers can verify, critique, and extend the analysis without needing to understand neural network internals.

---

## 📁 Repository Structure

```
OCTOPUS-BEHAVIORAL-ANALYSIS
│
├── DATASET/
│   └── 3836204455-preview.mp4        # Octopus video footage (Shutterstock)
│
├── notebooks/
│   └── cephalopod_analysis.ipynb     # Full analysis notebook (feature extraction + plots)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd cephalopod-behavior-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python==4.13.0.92 numpy==2.3.4 matplotlib==3.10.7
```

### 3. Verify Installation

```python
import cv2
import numpy as np
import matplotlib
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
```

### Dependency Overview

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | 4.13.0.92 | Video I/O, frame differencing, contour detection |
| `numpy` | 2.3.4 | Array operations, signal processing |
| `matplotlib` | 3.10.7 | Feature visualization and plotting |

---

## 🎬 Dataset

The analysis uses a short underwater footage clip of an octopus sourced from Shutterstock.

```
DATASET/
 └── 3836204455-preview.mp4
```

**Extracted Video Metadata (via OpenCV):**

| Property | How It Is Read |
|---|---|
| Resolution | `cap.get(cv2.CAP_PROP_FRAME_WIDTH)` × `cap.get(cv2.CAP_PROP_FRAME_HEIGHT)` |
| Frame Rate | `cap.get(cv2.CAP_PROP_FPS)` |
| Total Frames | `int(cap.get(cv2.CAP_PROP_FRAME_COUNT))` |
| Duration | `total_frames / fps` (seconds) |

> **Note:** Frames are resized to **224 × 224** before processing for faster computation. For production-grade analysis, higher resolution and longer recordings are recommended.

---

## 🧪 Methodology

The notebook processes the video sequentially, reading frame pairs to extract two signals, then normalizes and overlays both on a shared time axis.

```
Video File (MP4)
    │
    ▼
cap = cv2.VideoCapture(video_path)
    │
    ├──► Read metadata (FPS, resolution, total frames, duration)
    │
    ├──► Read first frame → resize to (224, 224) → convert to grayscale (prev_gray)
    │
    └──► Frame loop (while cap.read()):
              │
              ├── Resize frame → convert to grayscale (gray)
              │
              ├── FEATURE 1: cv2.absdiff(prev_gray, gray) → np.sum()  ──► motion_feats[]
              │
              ├── FEATURE 2: cv2.threshold(gray, 50, 255, THRESH_BINARY_INV)
              │              → cv2.findContours() → max contourArea()  ──► contour_areas[]
              │
              └── Append frame_idx / fps                               ──► time_stamps[]

    ▼
Min-max normalize both signals → overlay plot (matplotlib)
```

---

## 📊 Extracted Features

### 1. Motion Magnitude

**Definition:** The sum of absolute pixel differences between consecutive grayscale frames.

**Code (from notebook):**
```python
motion_val = np.sum(cv2.absdiff(prev_gray, gray))
motion_feats.append(motion_val)
prev_gray = gray
```

**Interpretation:**

| Signal Level | Behavioral Meaning |
|---|---|
| High | Active movement — exploration, escape, stress response |
| Moderate | Slow locomotion or tentacle activity |
| Low | Resting, hiding, or camouflage behavior |

This feature answers: **"How much is the octopus moving?"**

---

### 2. Contour Area

**Definition:** The pixel area of the largest detected contour in a thresholded frame, used as a proxy for body size/posture.

**Code (from notebook):**
```python
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    max_area = max(cv2.contourArea(c) for c in contours)
else:
    max_area = 0
contour_areas.append(max_area)
```

> **Note:** `THRESH_BINARY_INV` inverts the threshold — darker regions (the octopus body against a lighter background) become the foreground. The threshold value of `50` was chosen empirically.

**Interpretation:**

| Signal Level | Behavioral Meaning |
|---|---|
| Large | Body fully extended — alert, foraging, or inking |
| Moderate | Partial extension — transitioning between states |
| Small | Body contracted — hiding, resting, or camouflaged |

This feature answers: **"What is the body posture of the octopus?"**

---

### Normalization & Visualization

Both signals are min-max normalized before plotting so they share a common [0, 1] scale:

```python
motion_norm = (motion_feats - motion_feats.min()) / (motion_feats.max() - motion_feats.min())
contour_norm = (contour_areas - contour_areas.min()) / (contour_areas.max() - contour_areas.min())

plt.figure(figsize=(12, 6))
plt.plot(time_stamps, motion_norm, label="Motion Magnitude (normalized)", color="blue")
plt.plot(time_stamps, contour_norm, label="Contour Area (normalized)", color="red")
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Feature Value")
plt.title("Octopus Behavioral Features Over Time")
plt.legend()
plt.show()
```

---

## 🧠 Behavioral Interpretation Framework

When used together, motion magnitude and contour area form a 2D behavioral state space. Each quadrant maps to a distinct behavioral interpretation:

| Motion Magnitude | Contour Area | Behavioral State | Ethological Interpretation |
|---|---|---|---|
| **High** | **High** | Active & Expanded | Exploration, foraging, or stress response |
| **High** | **Low** | Active & Contracted | Rapid escape, cautious locomotion |
| **Low** | **High** | Stationary & Expanded | Alert watch, territorial display |
| **Low** | **Low** | Stationary & Contracted | Resting, hiding, or deep camouflage |

**Key Insight:**
- Motion magnitude encodes **activity level** — *is the octopus moving?*
- Contour area encodes **postural state** — *is the body open or closed?*
- Together, they provide a compact, interpretable behavioral descriptor without any trained model

These two dimensions roughly correspond to established ethological constructs: **arousal** (activity level) and **approach-avoidance** (expanded vs. contracted posture).

---

## 📈 Results & Visualizations

Running the notebook produces a single dual-signal time-series plot:

- **Blue line** — normalized motion magnitude over time, showing activity bursts and quiet periods
- **Red line** — normalized contour area over time, showing body expansion and contraction events

Regions where both lines are elevated together indicate active, expanded behavior. Regions where both are suppressed indicate resting or hiding. Divergence between the two lines reveals postural changes without locomotion (or vice versa).

---

## ⚠️ Limitations

**Signal Quality:**
- Motion magnitude cannot distinguish octopus movement from background noise such as water ripples or lighting flicker.
- Contour detection degrades significantly with dynamic backgrounds, poor lighting, or camouflage — ironically, when octopus behavior is most interesting.
- The fixed threshold value (`50`) is not adaptive and may perform inconsistently across different lighting conditions.

**Feature Scope:**
- Fine-grained tentacle movement is not captured — the signals reflect whole-body changes only.
- Color and texture changes (chromatophore activity) — primary behavioral signals in cephalopods — are not modeled.
- Spatial location of the octopus within the frame is discarded.

**Interpretation:**
- Behavioral state labels are rule-based approximations, not ground-truth ethological annotations.
- No calibration against expert-labeled behavioral data has been performed.

---

## 🚀 Future Work & Extensions

### Short-Term (Classical CV)

| Extension | Benefit |
|---|---|
| **Audio analysis** | Captures water disturbance and external noise; helps answer *"Did something in the environment trigger this reaction?"* |
| **Optical Flow (Farneback / Lucas-Kanade)** | Captures local motion direction; distinguishes tentacle activity from whole-body movement |
| **Adaptive thresholding** | More robust contour detection across varying lighting conditions |
| **HSV Color Space Analysis** | Tracks chromatophore-driven color changes as an independent behavioral signal |
| **Background Subtraction (MOG2 / KNN)** | Reduces false motion detections from dynamic water backgrounds |

### Medium-Term (Deep Learning)

| Extension | Benefit |
|---|---|
| **Pose Estimation (DeepLabCut / SLEAP)** | Tracks individual tentacle keypoints for fine-grained postural analysis |
| **Semantic Segmentation** | Precisely separates octopus pixels from background for robust feature extraction |
| **Temporal Models (LSTM / Transformer)** | Learns temporal dependencies and transition patterns between behavioral states |

### Long-Term (Multi-Modal)

| Extension | Benefit |
|---|---|
| **Environmental Context (Object Detection)** | Links behavioral states to environmental triggers — prey, predators, shelter |
| **3D Reconstruction** | Enables volumetric body shape analysis from multi-camera setups |
| **Automated Ethogram Generation** | Produces annotated behavioral timelines comparable to expert human annotation |

---

## 💬 Author's Note

While this entry task focuses on simple and interpretable features, I am comfortable working with more advanced approaches, including deep learning and multi-modal modeling. I chose to keep the implementation straightforward to align with the task requirements and ensure clarity and reproducibility.

If given the opportunity to work on the full project, I would be excited to extend this pipeline with more sophisticated techniques — spatiotemporal models, improved feature extraction, and multi-modal integration — to achieve higher accuracy and deeper behavioral understanding.


---

*Built with OpenCV, NumPy, and Matplotlib — no specialized video understanding models required.*
