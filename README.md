# 🛡️ SafeFall AI: Real-Time Human Posture & Fall Detection System
### Comprehensive Deep Learning & Biomechanical Vision Pipeline (FA-2 Solution)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![YOLOv8 Pose](https://img.shields.io/badge/YOLOv8-17%20Keypoints-green)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SafeFall AI** is an enterprise-grade, privacy-first computer vision and deep learning system engineered for non-invasive patient monitoring, automated fall detection, and rapid emergency dispatch in geriatric care, hospitals, and assisted-living residences.

By combining **17-Keypoint Skeletal Pose Estimation**, **Bidirectional Temporal LSTMs (BiLSTM)**, and **Biomechanical Kinematic Heuristics**, the platform achieves **94.6% Fall Precision** and **95.2% Fall Recall** at zero-lag **15 FPS** real-time speeds.

---

## 📑 Alignment with FA-2 Assignment Brief (Steps 1 – 8)

| Assignment Phase | Implementation in SafeFall AI | Key Module |
| :--- | :--- | :--- |
| **Step 1: Problem Definition** | Elderly fall detection & injury mitigation in private homes & elder care | [app.py](file:///app.py) |
| **Step 2: Data Collection** | Le2i Fall Detection benchmark + real-world kinematic pose sequences (15,336 frames) | [data/](file:///data) |
| **Step 3: Feature Engineering** | 17 COCO landmarks, torso inclination, aspect ratio, knee flexion, descent velocity | [core/kinematics.py](file:///core/kinematics.py) |
| **Step 4: Model Selection** | YOLOv8-Pose detector + Deep Temporal BiLSTM Sequence Classifier | [model/pose_detector.py](file:///model/pose_detector.py) |
| **Step 5: Model Training** | Stratified 70/15/15 train/val/test split across 6 activity classes with Cosine Annealing | [model/safefall_nn_model.pth](file:///model/safefall_nn_model.pth) |
| **Step 6: Model Evaluation** | 88.81% Test Accuracy, 95.15% Fall Recall, Confusion Matrix, and 4 Deployment Mitigations | [assets/evaluation_summary.json](file:///assets/evaluation_summary.json) |
| **Step 7: Web Deployment** | Fast Streamlit dashboard with Live Stream, Photo, Video, Snapshot, and SOS Center | [ui/dashboard_view.py](file:///ui/dashboard_view.py) |
| **Step 8: Retraining & Maint.** | Active learning feedback loop, automated drift retraining pipeline & CCTV/IR roadmap | [ui/dashboard_view.py](file:///ui/dashboard_view.py) |

---

## 🦴 1. Pose Estimation & Skeletal Kinematics (Steps 3 & 4)

SafeFall AI uses a high-efficiency YOLOv8 Pose model to detect and track human subjects, extracting **17 COCO anatomical landmarks** per frame in under 12 ms:

```
                  (0) Nose
               /           \
      (1) L-Eye             (2) R-Eye
         |                     |
      (3) L-Ear             (4) R-Ear
               \           /
           (5) L-Shoulder ── (6) R-Shoulder
            /     │      │     \
           /      │      │      \
  (7) L-Elbow     │      │     (8) R-Elbow
         │        │      │        │
  (9) L-Wrist     │      │    (10) R-Wrist
           (11) L-Hip ── (12) R-Hip
            /                  \
           /                    \
    (13) L-Knee              (14) R-Knee
         │                        │
    (15) L-Ankle             (16) R-Ankle
```

### Biomechanical Pose Telemetry

From these 17 normalized keypoints $(x_i, y_i, c_i)$, the engine computes real-time anatomical vectors:

1. **Torso Inclination Angle ($\theta_{\text{torso}}$)**:
   $$\text{Shoulder}_{\text{mid}} = \frac{KP_5 + KP_6}{2}, \quad \text{Hip}_{\text{mid}} = \frac{KP_{11} + KP_{12}}{2}$$
   $$\theta_{\text{torso}} = \arctan2(|\text{Hip}_x - \text{Shoulder}_x|, |\text{Hip}_y - \text{Shoulder}_y|) \times \frac{180^\circ}{\pi}$$
   - **Nominal Upright (Standing/Walking):** $\theta < 25^\circ$
   - **Leaning / Transitioning:** $25^\circ \le \theta \le 55^\circ$
   - **Horizontal / Collapsed (Fall):** $\theta > 55^\circ$

2. **Bounding Box Aspect Ratio ($AR$)**:
   $$AR = \frac{x_{\max} - x_{\min}}{\max(1.0, y_{\max} - y_{\min})}$$
   - **Standing:** $AR < 0.50$ (tall & narrow)
   - **Sitting:** $0.55 \le AR \le 0.95$
   - **Fallen Ground Posture:** $AR > 1.15$ (wide & horizontal)

3. **3-Point Knee Joint Flexion Angle**:
   Measures interior joint angle between Hip $\rightarrow$ Knee $\rightarrow$ Ankle to differentiate controlled chair sitting from uncontrolled floor collapse.

4. **Dynamic Gait Oscillation**:
   Tracks Euclidean separation velocity between Left Ankle ($KP_{15}$) and Right Ankle ($KP_{16}$) across temporal sliding windows to identify active walking vs standing still.

---

## 🏃 2. Complete 6 Clinical Activity Classes (Step 5)

The dual-engine architecture classifies human posture across all **6 clinical activity categories**:

| Activity Class | Primary Visual & Biomechanical Markers | Alert Status |
| :--- | :--- | :--- |
| 🚨 **Fall Detected** | Rapid descent velocity + Torso angle $> 55^\circ$ + Aspect ratio $> 1.15$ + Ground proximity | **CRITICAL ALERT (SOS + Siren)** |
| ⚠️ **Off-Balance** | High torso oscillation ($35^\circ - 55^\circ$) + Irregular centroid acceleration + Narrow foot base | **WARNING (Pre-fall advisory)** |
| 🚶 **Walking** | Cyclic stride oscillation between ankles + Upright torso ($< 25^\circ$) | **NOMINAL (Active locomotion)** |
| 🧍 **Standing** | Stable vertical alignment + Ankle-to-shoulder linearity + Zero gait cycle | **NOMINAL (Upright stationary)** |
| 🪑 **Sitting** | Knee flexion $\approx 90^\circ$ + Torso inclination $< 30^\circ$ + Elevated chair plane | **NOMINAL (Resting posture)** |
| 🛡️ **Normal Activity** | Nominal domestic movement / room vacant sentinel | **NOMINAL (All clear)** |

---

## 📊 3. Model Evaluation & Performance Curves (Step 6)

The PyTorch BiLSTM neural classifier was rigorously evaluated on an unseen test dataset split of **2,707 samples** from the Le2i Fall Detection benchmark:

### Quantitative Test Split Metrics

| Metric | Measured Value | Standard Benchmark Target | FA-2 Target Status |
| :--- | :---: | :---: | :---: |
| **Fall Detection Recall (Sensitivity)** | **95.15%** | $> 85.0\%$ | ✅ **EXCEEDED (+10.15%)** |
| **Fall Detection Precision** | **94.58%** | $> 85.0\%$ | ✅ **EXCEEDED (+9.58%)** |
| **Fall Class F1-Score** | **94.86%** | $> 85.0\%$ | ✅ **EXCEEDED (+9.86%)** |
| **Overall Test Accuracy** | **88.81%** | $> 80.0\%$ | ✅ **EXCEEDED (+8.81%)** |
| **Macro Average F1-Score** | **86.93%** | $> 80.0\%$ | ✅ **EXCEEDED (+6.93%)** |
| **Weighted F1-Score** | **88.97%** | $> 80.0\%$ | ✅ **EXCEEDED (+8.97%)** |

### Per-Class Detailed Performance

| Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| 🚨 **Fall Detected** | **94.58%** | **95.15%** | **94.86%** | 330 frames |
| ⚠️ **Off-Balance** | **94.25%** | 85.06% | 89.42% | 810 frames |
| 🪑 **Sitting** | 93.31% | **95.64%** | 94.46% | 642 frames |
| 🧍 **Standing** | 87.90% | 81.34% | 84.49% | 509 frames |
| 🚶 **Walking** | 62.77% | 84.80% | 72.14% | 171 frames |
| 🛡️ **Normal Activity** | 80.28% | 93.06% | 86.20% | 245 frames |

### Real-World Deployment Challenges & Technical Mitigations

1. **💡 Lighting Variations (Day, Night, Shadows)**:
   - *Challenge*: Low-light domestic bedrooms and night-time bathroom trips degrade RGB appearance.
   - *Mitigation*: The vision pipeline converts pixel frames into normalized skeletal keypoint coordinates $(x/w, y/h)$ that are mathematically invariant to lux and illumination levels. CLAHE + power-law gamma correction ($\gamma=1.6$) preprocess low-light inputs.
2. **📐 Camera Angle Differences (Ceiling, Wall, Desk)**:
   - *Challenge*: Oblique perspective transforms distort aspect ratios and subject heights.
   - *Mitigation*: Desk Mode calibration applies upper-body kinematic normalization when lower extremities are occluded by tables, chairs, or bed frames.
3. **🪑 Posture Ambiguity (Sitting vs Falling)**:
   - *Challenge*: Sitting down onto a low couch or floor mat resembles the final state of a fall.
   - *Mitigation*: The 30-frame temporal BiLSTM analyzes continuous vertical descent velocity. Controlled sitting shows deceleration ($< 0.8\text{ m/s}$), whereas falls exhibit rapid gravitational acceleration followed by sudden impact arrest.
4. **🔒 Privacy Preservation in Sensitive Spaces**:
   - *Challenge*: Patients refuse video cameras in bedrooms, dressing areas, and bathrooms.
   - *Mitigation*: SafeFall AI processes keypoint geometry entirely in edge memory. Raw video streams are never written to disk or transmitted across external networks—only anonymized skeletal coordinates and emergency incident metadata are retained.

---

## 🔄 4. Continuous Monitoring & Model Retraining Pipeline (Step 8)

To prevent concept drift and maintain clinical reliability over long-term deployments, SafeFall AI implements an active learning and automated retraining pipeline:

```
┌────────────────────────────────────────────────────────┐
│             Real-Time Vision Stream                    │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│   Active Uncertainty Sampler & Telemetry Monitor       │
│   - Confidence marginality: 0.40 <= Confidence <= 0.65 │
│   - Caregiver False-Positive Override Feedback         │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│     Automated Edge Data Annotation & Staging           │
│     - Stratified 70% Train / 15% Val / 15% Test Split  │
│     - Cosine Annealing Learning Rate Schedule          │
│     - Early Stopping on Validation Loss (Patience = 5) │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│       Automated Regression & Delta Verification        │
│       - Fall Recall Delta >= +0.0% (Zero Regression)   │
│       - Fall Precision Delta >= +0.0%                  │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│    Zero-Downtime Hot Swapping of PyTorch Checkpoint    │
│    (safefall_nn_model.pth automatically reloaded)      │
└────────────────────────────────────────────────────────┘
```

### Continuous Maintenance Roadmap
- **Multi-Camera CCTV Feeds**: Native RTSP and ONVIF stream ingest for whole-facility elder care monitoring.
- **Thermal & Infrared Low-Light Adaptation**: Fine-tuning keypoint extractors on nocturnal thermal imagery.
- **Mobility Aid Invariance**: Specialized kinematic normalization for residents utilizing walkers, canes, and wheelchairs.
- **Embedded Edge Acceleration**: Quantization to INT8 TensorRT and OpenVINO for deployment on Jetson Nano and Coral Edge TPUs.

---

## 🚀 5. Performance Optimizations & Zero-Lag Architecture

SafeFall AI is engineered for high throughput and instant UI responsiveness:

- **15 FPS WebRTC Background Worker**: Camera frames are processed in dedicated background daemon threads with `video_receiver_size=2` to eliminate input lag.
- **Non-Blocking `@st.fragment(run_every=1)`**: Telemetry updates asynchronously every 1.0 second without triggering full-page Streamlit reruns.
- **Zero Double-Rerun Ping-Pong**: Clean single horizontal mode selector eliminating competing state navigation loops.
- **Instant Typography (0.0 ms Font Lag)**: Native OS system fonts (`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto`) replacing network-blocking Google Fonts.
- **Single Fast STUN Gatherer**: Uses Google's primary STUN endpoint (`stun:stun.l.google.com:19302`) for instantaneous ICE candidate gathering.

---

## 🏗️ 6. Repository Architecture

```
.
├── app.py                         # Application entrypoint & mode router
├── yolov8n-pose.pt                # YOLOv8 nano pose model weights
├── requirements.txt               # Locked production dependencies
├── packages.txt                   # Linux runtime dependencies
├── run_app.bat                    # 1-click Windows launch script
├── push_to_github.bat             # 1-click GitHub push script
├── core/                          # Machine learning & analytics modules
│   ├── security.py                # Salted PBKDF2-HMAC-SHA256 authentication vault
│   ├── kinematics.py              # Biomechanical angle and posture heuristics
│   ├── body_tracker.py            # Spatial IoU & centroid person tracker
│   ├── temporal_filter.py         # Exponential smoothing & hysteresis gate
│   └── vision_pipeline.py         # YOLOv8 pipeline & WebRTC streaming worker
├── ui/                            # Presentation and user interface
│   ├── styles.py                  # High-performance CSS and themes
│   ├── components.py              # KPI cards, SVG badges, and siren player
│   ├── auth_view.py               # Login, registration, and guest access
│   └── dashboard_view.py          # Monitor, Photo, Video, SOS, Retraining views
├── model/                         # Neural network checkpoints
│   ├── safefall_nn_model.pth      # PyTorch BiLSTM sequence classifier
│   ├── feature_scaler.joblib      # Robust kinematic feature standardizer
│   └── pose_detector.py           # Pose estimation abstraction
├── assets/                        # Benchmark evaluation artifacts
│   └── evaluation_summary.json    # Exact test split metrics & confusion stats
├── training/                      # Training history & logs
│   └── training_history.json      # 45-epoch loss & accuracy progression
├── outputs/                       # Runtime outputs
│   ├── users.json                 # Salted user credentials
│   └── falls/                     # Full-resolution incident snapshots
└── tests/                         # Comprehensive automated test suites
    ├── test_all_6_classes.py      # 6 clinical activity verification
    ├── test_sos_alarm.py          # Acoustic siren & SOS dialer verification
    └── test_pose_and_retraining.py# 17-keypoint pose & Step 8 retraining tests
```

---

## ⚙️ 7. Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA.git
cd "ML & DL FA 2"
```

### 2. Set Up Python Virtual Environment
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Tests
```bash
python tests/test_all_6_classes.py
python tests/test_sos_alarm.py
python tests/test_pose_and_retraining.py
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🧪 Default Demo Credentials
- **Email:** `demo@safefall.ai`
- **Password:** `safefall123`
*(Or click **Continue as guest** for instant access without signing in).*

---

## 📄 License
This project is licensed under the MIT License.
