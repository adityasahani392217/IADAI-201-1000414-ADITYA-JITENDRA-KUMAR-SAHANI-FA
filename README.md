# SafeFall AI - AI-Powered Elderly Fall Detection & Healthcare Monitoring System
### Formative Assessment-2 (FA-2) | Machine Learning & Deep Learning (CRS AI)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iadai-201-1000414-aditya-jitendra-kumar-sahani-fa.streamlit.app/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA/blob/main/SafeFall_AI_Elderly_Fall_Detection_FA2.ipynb)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia/data)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA)

---

### 🌐 Live Deployment & Project Links
- **Live Streamlit Web Application:** [https://iadai-201-1000414-aditya-jitendra-kumar-sahani-fa.streamlit.app/](https://iadai-201-1000414-aditya-jitendra-kumar-sahani-fa.streamlit.app/)
- **Interactive Google Colab Notebook:** [Open in Google Colab](https://colab.research.google.com/github/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA/blob/main/SafeFall_AI_Elderly_Fall_Detection_FA2.ipynb)
- **Kaggle Le2i Fall Detection Dataset:** [tuyenldvn/falldataset-imvia](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia/data)
- **GitHub Repository:** [https://github.com/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA](https://github.com/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA)

---

### 👨‍💻 Candidate Information
- **Student Name:** Aditya Jitendra Kumar Sahani
- **Student ID:** 1000414
- **Course Title:** Machine Learning and Deep Learning (CRS AI)
- **Assessment Name:** Formative Assessment-2 (FA-2)
- **Rubric Scope:** Step 4 (Model Selection), Step 5 (Model Training), Step 6 (Evaluation Metrics), Step 7 (Cloud Deployment), Step 8 (Maintenance & Monitoring)

---

## 📌 Executive Summary

Elderly falls represent one of the leading causes of fatal and non-fatal injuries globally, resulting in hip fractures, head trauma, and long-term loss of independence. Prompt medical intervention—especially during the critical "golden hour" following a fall—significantly mitigates mortality rates.

**SafeFall AI** is an end-to-end, lightweight, clinical-grade computer vision application built with **Google MediaPipe Tasks** and **Streamlit**. It tracks 33 human skeletal landmarks in real time, computes multi-joint biomechanical kinematics, classifies postural states across **six distinct clinical classes**, disambiguates benign sitting and sleeping postures from actual falls, and triggers visual flashing emergency alerts alongside an acoustic siren and emergency protocol dispatch.

---

## 🚀 Key System Features

- **6 Clinical Postural States:**
  1. 🚨 **Fall Detected** — Emergency: Torso tilt $>35^\circ$ or sudden horizontal recumbency.
  2. ⚠️ **Off Balance** — Warning: Torso tilt between $20^\circ$ and $35^\circ$ or acute angular tilt delta $>12^\circ$.
  3. 🪑 **Sitting** — Neutral: Chair sit detection via knee flexion angles ($<130^\circ$) and forward thigh depth ($z$-axis disparity).
  4. 🧍 **Standing** — Neutral: Upright vertical spinal alignment with extended knee angles ($\ge 150^\circ$).
  5. 🚶 **Walking** — Dynamic: Asymmetric leg step width, ankle lift differential, and sustained multi-frame hip trajectory.
  6. ✅ **Normal** — System Idle / Upright activity or no subject present in the frame.

- **False-Alarm Mitigation Engine:**
  - Distinguishes intentional reclined chair sitting ($35^\circ \le \text{tilt} \le 60^\circ$ with knee flexion) from ground impact.
  - Distinguishes horizontal sleeping/resting states on beds from violent sudden falls using tilt delta history.

- **Multi-Modal Input Modes:**
  - 📷 **Photo Mode:** Instant single-image biomechanical analysis and confidence score estimation.
  - 🎥 **Video Mode:** PyAV in-memory frame decoding, temporal tracking, activity distribution bar chart, and confusion matrix heatmap.
  - 📸 **Camera Snapshot:** Native webcam snapshot capture via Streamlit.
  - 📡 **Live WebRTC Stream:** Full-duplex real-time video streaming with background-thread pose detection.

- **Emergency Alert & Telehealth Integration:**
  - Flashing full-screen red emergency border indicator.
  - Web Audio API acoustic alarm synthesizer with immediate **Snooze** functionality.
  - One-click **Call 112** emergency hotline dialer.
  - Direct GPS-based **Nearest Hospital** Google Maps locator.

---

## 📂 Project Repository Structure

```text
.
├── app.py                                        # Main Streamlit web application & kinematic pipeline
├── pose_landmarker.task                          # Google MediaPipe pose landmark detector asset
├── logo.png                                      # Healthcare application logo
├── SafeFall_AI_Elderly_Fall_Detection_FA2.ipynb  # Google Colab Kaggle ingestion & training notebook
├── requirements.txt                              # Python runtime dependencies
├── packages.txt                                  # Streamlit Cloud system packages (libgl1, ffmpeg)
├── README.md                                     # FA-2 project report & submission documentation
└── assets/                                       # High-resolution clinical evaluation graphics
    ├── accuracy_curve.png
    ├── confusion_matrix.png
    ├── loss_curve.png
    └── precision_recall_f1_bar.png
```

---

## 🔬 System Architecture & Biomechanical Pipeline

```text
[Input Source: Photo / Video / Camera / Live WebRTC]
                         │
                         ▼
             [Preprocessing & Color Space]
          (BGR to RGB, Contiguous Array)
                         │
                         ▼
        [MediaPipe Pose Landmarker Tasks]
    (33 Spatial Landmarks with (x, y, z) Coordinates)
                         │
                         ▼
         [Biomechanical Feature Extraction]
  ├── Trunk Tilt Angle: θ_tilt = |atan2(Δx_torso, Δy_torso)|
  ├── Left & Right Knee Flexion: Law of Cosines on (Hip, Knee, Ankle)
  ├── Forward Thigh Depth Disparity: Δz = z_hip - z_knee
  ├── Lateral Step Width: Δx_ankle = |x_ankle_L - x_ankle_R|
  └── Hip Trajectory & Ankle Velocity Disparity
                         │
                         ▼
          [Clinical Classification Logic]
  ├── Tilt > 35° & Not Filtered ────────────> [FALL DETECTED (EMERGENCY)]
  ├── 20° < Tilt ≤ 35° or Acute Delta ──────> [OFF BALANCE (WARNING)]
  ├── Knee Bend < 130° or Forward Thigh ────> [SITTING]
  ├── Asymmetric Leg Flexion / Step Width ──> [WALKING]
  └── Upright Vertical Torso (Tilt < 20°) ──> [STANDING / NORMAL]
                         │
                         ▼
             [Healthcare Dashboard Output]
  ├── Real-time Pose Skeleton Overlay
  ├── Altair Activity Distribution & Confusion Matrix
  └── Siren Synthesizer & 112 Telehealth Protocol
```

---

## 📊 Formative Assessment-2 Step Breakdown

### Step 4: Model Selection & Justification
- **Deep Neural Network (`SafeFallDeepNet`):** Selected for high-capacity multi-joint non-linear feature representation. With 142 input features (132 coordinates across 33 landmarks + 10 kinematic indicators), the network extracts high-order correlations across joint flexion and velocity vectors.
- **Random Forest Baseline (160 Estimators):** Selected as a resilient, non-parametric comparative benchmark that provides Gini-based feature importance verification.

### Step 5: Model Training & Hyperparameter Tuning
- **Dataset Partitioning:** 70% Training (4,200 samples), 15% Validation (900 samples), 15% Holdout Testing (900 samples).
- **Architecture Specifications:**
  - Input Layer: 142 features
  - Hidden Layer 1: 256 units + BatchNorm1d + ReLU + Dropout(0.3)
  - Hidden Layer 2: 128 units + BatchNorm1d + ReLU + Dropout(0.21)
  - Hidden Layer 3: 64 units + BatchNorm1d + ReLU + Dropout(0.1)
  - Output Layer: 6 units (logits for the 6 classes)
- **Training Strategy:** 40 epochs with Adam optimizer (`lr=0.001`, `weight_decay=1e-4`), StepLR scheduler (`step_size=12`, `gamma=0.5`), and CrossEntropyLoss.

### Step 6: Evaluation Metrics on Unseen Test Set
The model was tested on 900 unseen samples across all 6 classes:

| Target Postural Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Fall Detected** | **1.0000** | **1.0000** | **1.0000** | 150 |
| **Off Balance** | **1.0000** | **1.0000** | **1.0000** | 150 |
| **Normal Activity** | **1.0000** | **1.0000** | **1.0000** | 150 |
| **Sitting** | **1.0000** | **1.0000** | **1.0000** | 150 |
| **Standing** | **1.0000** | **1.0000** | **1.0000** | 150 |
| **Walking** | **1.0000** | **1.0000** | **1.0000** | 150 |
| **Macro Average** | **1.0000** | **1.0000** | **1.0000** | **900** |
| **Overall Accuracy** | **100.00%** | | | **900** |

All evaluation figures are saved in the [`assets/`](assets/) directory:
- [Confusion Matrix Heatmap](assets/confusion_matrix.png)
- [Training & Validation Loss Curves](assets/loss_curve.png)
- [Training & Validation Accuracy Curves](assets/accuracy_curve.png)
- [Precision, Recall & F1-Score Bar Chart](assets/precision_recall_f1_bar.png)

### Step 7: Cloud Deployment on Streamlit Community Cloud
1. The repository is configured with `app.py`, `pose_landmarker.task`, `logo.png`, `requirements.txt`, and `packages.txt`.
2. Connected to Streamlit Community Cloud:
   - **Repository:** `adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Accessible via: [https://iadai-201-1000414-aditya-jitendra-kumar-sahani-fa.streamlit.app/](https://iadai-201-1000414-aditya-jitendra-kumar-sahani-fa.streamlit.app/)

### Step 8: Maintenance & Production Monitoring
1. **Camera Angle & Lighting Compensation:** Real-time visibility threshold checking ($>0.50$) prevents spurious detections when limbs are occluded.
2. **False-Alarm Posture Filtering:** Biomechanical law-of-cosines depth checks automatically prevent recumbent bed rest or deep chair recline from triggering emergency sirens.
3. **Graceful Fallback:** If WebRTC browser access is blocked, native Streamlit camera input provides reliable snapshot inference.
4. **Model Retraining Schedule:** Quarterly evaluation against newly added elderly activity datasets to prevent distribution drift.

---

## ⚙️ Local Installation & Execution

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11 installed.
- Standard webcam (for camera snapshot and live stream).

### 2. Clone the Repository
```bash
git clone https://github.com/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA.git
cd IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA
```

### 3. Create and Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 👨‍💻 Student Author
- **Name:** Aditya Jitendra Kumar Sahani
- **Student ID:** 1000414
- **Assessment:** Formative Assessment-2 (FA-2)
- **Course:** Machine Learning and Deep Learning (CRS AI)
