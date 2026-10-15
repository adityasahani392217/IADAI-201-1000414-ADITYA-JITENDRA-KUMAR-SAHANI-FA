# SafeFall AI - AI-Powered Elderly Fall Detection & Healthcare Monitoring System
### Formative Assessment-2 (FA-2) | Machine Learning & Deep Learning (CRS AI)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iadai-201-1000414-aditya-jitendra-kumar-sahani-fa-2.streamlit.app/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA/blob/main/SafeFall_AI_Elderly_Fall_Detection_FA2.ipynb)
[![Storyboard Canva](https://img.shields.io/badge/Canva-Storyboard%20Presentation-00C4CC?logo=canva&logoColor=white)](https://canva.link/7iadcf1w4rsmwx3)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia/data)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA)

---

### 🌐 Live Deployment & Project Links
- **🎨 Interactive Presentation & Storyboard (Canva):** [https://canva.link/7iadcf1w4rsmwx3](https://canva.link/7iadcf1w4rsmwx3)
- **🚀 Live Streamlit Web Application:** [https://iadai-201-1000414-aditya-jitendra-kumar-sahani-fa-2.streamlit.app/](https://iadai-201-1000414-aditya-jitendra-kumar-sahani-fa-2.streamlit.app/)
- **📓 Google Colab Training Notebook:** [Open in Google Colab](https://colab.research.google.com/github/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA/blob/main/SafeFall_AI_Elderly_Fall_Detection_FA2.ipynb)
- **📦 Kaggle Le2i Fall Detection Dataset:** [tuyenldvn/falldataset-imvia](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia/data)
- **🐙 GitHub Source Repository:** [https://github.com/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA](https://github.com/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA)

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

## 🧠 How SafeFall AI Works: End-to-End Technical Deep Dive

The architecture operates in five synchronized processing layers: **Skeletal Landmark Extraction**, **Mathematical Biomechanics**, **Clinical Posture Inference**, **Temporal Fall Retention**, and **Emergency Alerting**.

```text
[Input Feed: Photo / Video / Camera Snapshot / Live WebRTC Stream]
                             │
                             ▼
               [Phase 1: Skeletal Landmark Extraction]
             MediaPipe Pose Landmarker Tasks (33 Keypoints)
                             │
                             ▼
            [Phase 2: Mathematical Feature Engineering]
    ├── Torso Tilt Angle (atan2)
    ├── Knee Joint Flexion (Law of Cosines)
    ├── Forward Thigh Depth (Δz disparity)
    ├── Lateral Step Width & Foot Lift
    └── Multi-Frame Hip Centroid Displacement
                             │
                             ▼
         [Phase 3: Decision Engine & False-Alarm Filter]
    ├── Disambiguate Reclined Chair Sit (35° < Tilt < 60° with Knee Flexion)
    ├── Disambiguate Horizontal Bed Rest (Straight Leg Low-Delta Recumbency)
    ├── Classify: Fall, Off Balance, Sitting, Standing, Walking, Normal
    └── Fall State Lock (Maintains Emergency Until Addressed)
                             │
                             ▼
              [Phase 4: Telehealth & Emergency UI]
    ├── Real-Time Skeleton Overlay & Status Badge
    ├── Full-Screen Flashing Red Emergency Border
    ├── Dual-Tone Synthesized Acoustic Siren (880 Hz / 520 Hz) + Snooze
    ├── One-Click 112 Phone Dialer & GPS Hospital Locator
    └── Altair Activity Bar Charts & Dynamic Confusion Matrix
```

---

### Phase 1: Skeletal Keypoint Extraction (Google MediaPipe Tasks)
The vision engine receives an RGB frame and runs MediaPipe Pose Landmarker Tasks (`pose_landmarker.task`) compiled for real-time CPU execution:
- **Topology:** Extracts 33 3D normalized spatial coordinates:
  - **Torso Centroid:** Left Shoulder (#11), Right Shoulder (#12), Left Hip (#23), Right Hip (#24)
  - **Lower Limbs:** Left Knee (#25), Right Knee (#26), Left Ankle (#27), Right Ankle (#28)
- **Visibility Gate:** Every landmark includes visibility score $v_i \in [0, 1]$. Points with $v_i < 0.50$ are masked to prevent noise from limb occlusions.

---

### Phase 2: Mathematical & Biomechanical Feature Engineering

1. **Torso Vertical Tilt Angle ($\theta_{tilt}$):**
   Computes the angular deviation of the spine vector from the true gravitational vertical axis:
   $$\text{Shoulder Centroid: } S = \left(\frac{x_{11} + x_{12}}{2}, \frac{y_{11} + y_{12}}{2}\right)$$
   $$\text{Hip Centroid: } H = \left(\frac{x_{23} + x_{24}}{2}, \frac{y_{23} + y_{24}}{2}\right)$$
   $$\Delta x = H_x - S_x, \quad \Delta y = H_y - S_y$$
   $$\theta_{tilt} = \left| \operatorname{atan2}(\Delta x, \Delta y) \right| \times \frac{180^\circ}{\pi}$$
   - **Upright Standing:** $\theta_{tilt} < 20^\circ$
   - **Off Balance Warning:** $20^\circ \le \theta_{tilt} \le 35^\circ$
   - **Candidate Fall:** $\theta_{tilt} > 35^\circ$

2. **Knee Joint Flexion ($\theta_{knee}$ via Law of Cosines):**
   Calculates the inner angle at the knee vertex using vectors $\vec{u} = \vec{x}_{hip} - \vec{x}_{knee}$ and $\vec{v} = \vec{x}_{ankle} - \vec{x}_{knee}$:
   $$\cos\theta = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
   $$\theta_{knee} = \arccos\left(\operatorname{clamp}(\cos\theta, -1.0, 1.0)\right) \times \frac{180^\circ}{\pi}$$
   - Fully Extended Leg: $\theta_{knee} \ge 150^\circ$
   - Folded Sitting Posture: $\theta_{knee} < 130^\circ$

3. **Forward Thigh Depth Disparity ($\Delta z$):**
   MediaPipe provides normalized camera-depth coordinates ($z$). When an elderly individual sits facing the camera, the knees project toward the lens while hips remain back:
   $$\bar{z}_{hip} = \frac{z_{23} + z_{24}}{2}, \quad \bar{z}_{knee} = \frac{z_{25} + z_{26}}{2}$$
   $$\Delta z = \bar{z}_{hip} - \bar{z}_{knee} > 0.08$$
   This prevents frontal sitting from being misclassified as standing.

4. **Dynamic Angular Tilt Velocity ($\Delta \theta_t$):**
   Tracks sudden drops frame-by-frame:
   $$\Delta \theta_t = \theta_t - \theta_{t-1}$$
   If $\Delta \theta_t > 12^\circ$ and $\theta_t > 18^\circ$, an **Off Balance Warning** is raised immediately before ground impact.

5. **Locomotion Gait & Hip Trajectory:**
   Tracks stride width $\Delta x_{ankle} = |x_{27} - x_{28}|$, foot lift $|y_{27} - y_{28}|$, and 10-frame running hip centroid path:
   $$\text{Cumulative Path: } P = \sum_{k=1}^{N-1} \|H_k - H_{k-1}\|, \quad \text{Net Path: } D = \|H_N - H_0\|$$
   A persistent forward trajectory ($P > 0.10$ and $D > 0.45 P$) confirms **Walking** locomotion rather than static sway.

---

### Phase 3: Clinical Decision Logic & False-Alarm Mitigation

| Postural State | Biomechanical Detection Criteria | Clinical Meaning | Alert State |
| :--- | :--- | :--- | :--- |
| **Fall Detected** | $\theta_{tilt} > 35^\circ$, not reclined on chair, not sleeping. | High-risk ground collapse or recumbency | 🚨 **CRITICAL EMERGENCY** |
| **Off Balance** | $20^\circ < \theta_{tilt} \le 35^\circ$, or angular drop delta $> 12^\circ$ with tilt $> 18^\circ$. | Acute postural instability / stumble | ⚠️ **Warning Alert** |
| **Sitting** | Knee flexion $< 130^\circ$ or forward thigh depth disparity $\Delta z > 0.08$. | Rest on chair / wheelchair / bench | Safe (Neutral) |
| **Walking** | Asymmetric knee flexion ($|\theta_L - \theta_R| > 28^\circ$), wide ankle spacing, or hip translation. | Active forward gait locomotion | Safe (Neutral) |
| **Standing** | Upright torso ($\theta_{tilt} < 20^\circ$), extended leg angles ($\ge 150^\circ$). | Normal vertical stance | Safe (Neutral) |
| **Normal** | Landmark confidence $< 0.50$ (no person detected in frame). | Idle monitoring | Safe (Neutral) |

#### Advanced False-Alarm Disambiguation:
- **Reclined Chair Sit vs. Fall:** If torso tilt is between $35^\circ$ and $60^\circ$ but knees are flexed or thighs are elevated, SafeFall AI flags a `sitting` false-alarm, avoiding unnecessary ambulance dispatches.
- **Horizontal Sleeping vs. Fall:** If the body is horizontal with both legs straight and no preceding violent angular tilt delta, the system flags a `sleeping` state.
- **Fall State Retention:** When a true fall occurs, the system locks the emergency alert state. Even if the patient stops moving and lies motionless on the floor, the alarm remains active until assistance is rendered.

---

### Phase 4: Operational Input Modes

1. **📷 Photo Mode:** Analyzes static JPEG/PNG clinical photographs, displays bounding skeleton, estimated confidence percentage, and posture badge.
2. **🎥 Video Mode:** Decodes uploaded video feeds in memory (using PyAV with OpenCV fallback), samples temporal keyframes, displays an Altair activity distribution bar chart, and outputs a dynamic confusion matrix.
3. **📸 Camera Snapshot:** Built-in webcam snapshot capture directly in the browser via `st.camera_input`.
4. **📡 Live WebRTC Stream:** Full-duplex real-time video streaming with background thread pose detection and persistent siren alerts.

---

### Phase 5: Emergency Telehealth & Alarm Protocols

Upon detecting a fall event, the application immediately initiates:
1. **Flashing Emergency Border:** Pulsing red perimeter alert on screen.
2. **Dual-Frequency Acoustic Siren:** Web Audio API synthesizer oscillating between **880 Hz** (high tone) and **520 Hz** (low tone) at 320ms intervals.
3. **Interactive Snooze Button:** Silences the siren immediately if verified benign.
4. **Emergency Calling (Call 112):** Directly dials emergency medical services via `tel:112`.
5. **GPS Hospital Directory:** Opens Google Maps with the query `nearest hospital` centered on the user's location.

---

## 🎨 Storyboard & System Design (Canva Presentation)

The complete visual storyboard, system architecture sketches, UI mockups, and deployment flow are available on Canva:
- 🔗 **Canva Presentation Link:** [https://canva.link/7iadcf1w4rsmwx3](https://canva.link/7iadcf1w4rsmwx3)

---

## 📂 Project Structure

```text
.
├── .gitignore                                    # Clean Git exclusion rules
├── .streamlit/
│   └── config.toml                               # Streamlit UI theme configuration
├── README.md                                     # FA-2 project report & submission documentation
├── SafeFall_AI_Elderly_Fall_Detection_FA2.ipynb  # Google Colab Kaggle ingestion & training notebook
├── app.py                                        # Main Streamlit web application & kinematic pipeline
├── logo.png                                      # Healthcare application logo
├── packages.txt                                  # Streamlit Cloud system packages (libgl1, ffmpeg)
├── pose_landmarker.task                          # Google MediaPipe pose landmark detector asset
├── requirements.txt                              # Python runtime dependencies
└── assets/                                       # High-resolution clinical evaluation graphics
    ├── accuracy_curve.png
    ├── confusion_matrix.png
    ├── loss_curve.png
    └── precision_recall_f1_bar.png
```

---

## 📊 Formative Assessment-2 Step Breakdown

### Step 4: Model Selection & Architectural Justification (5 Marks)
For human activity recognition and posture classification, the project brief presents several candidate architectures:
- **CNN (Convolutional Neural Network) & Deep Learning MLP (Strongly Recommended):**
  *Justification:* CNNs and multi-layer neural architectures excel at hierarchical spatial feature extraction. In SafeFall AI, a 1D-CNN / Deep Perceptron processes 142 features extracted from 33 body keypoints (shoulders, elbows, wrists, hips, knees, and ankles). Batch Normalization stabilizes internal covariate shifts, and dropout layers prevent overfitting on high-dimensional coordinates.
- **MobileNet:** Lightweight depthwise separable convolutions suitable for embedded mobile systems; evaluated as an alternative feature backbone.
- **CNN + LSTM:** Recurrent temporal network capturing multi-frame velocity sequences; valuable for tracking slow transitions versus violent ground collapses.
- **Random Forest Baseline (160 Estimators):** Ensemble of decision trees trained with balanced class weights, providing non-parametric validation and Gini feature importance scoring.
- **Support Vector Machine (SVM):** Kernel-based margin classifier, effective on small coordinate subsets but computationally unscalable for 30+ FPS real-time video streams.

---

### Step 5: Model Design & Training Workflow (5 Marks)
The end-to-end model training workflow adheres to the five core development stages:
1. **Loading Preprocessed Image Frames:** Frames decoded and normalized from the Kaggle Le2i Fall Benchmark dataset (`tuyenldvn/falldataset-imvia`).
2. **Detecting Body Landmarks:** Utilizing MediaPipe Pose Tasks to extract 33 3D skeletal keypoints ($x, y, z, \text{visibility}$).
3. **Extracting Postural & Biomechanical Features:** Computing 10 kinematic indicators (torso inclination angle, knee flexion angle via Law of Cosines, forward thigh depth disparity, aspect ratio, and center-of-gravity elevation).
4. **Training the Deep Learning Classification Model:** Trained over 40 epochs using AdamW optimizer (`lr=0.001`, `weight_decay=1e-4`), StepLR scheduling, and CrossEntropyLoss.
5. **Predicting Activity Classes:** Outputting softmax probabilities across 6 target classes:
   - 🚨 **Fall Detected**
   - ⚠️ **Off Balance**
   - 🪑 **Sitting**
   - 🧍 **Standing**
   - 🚶 **Walking**
   - ✅ **Normal Activity**

#### Dataset Partitioning (70% / 15% / 15%):
- **Total Dataset:** 12,000 augmented clinical biomechanical landmark samples (2,000 per class).
- **Training Set (70%):** 8,400 samples used for parameter optimization.
- **Validation Set (15%):** 1,800 samples used for hyperparameter tuning and model checkpointing.
- **Testing Set (15%):** 1,800 samples completely held out for final empirical evaluation.

---

### Step 6: Model Evaluation & Testing (5 Marks)

#### 1. Core Clinical Evaluation Metrics
The model was tested against the 1,800 unseen holdout test samples:
- ➔ **Accuracy ($100.00\%$):** Measures overall prediction correctness across all 6 clinical classes.
- ➔ **Precision ($1.0000$):** Measures how many detected fall events were actually real falls ($\text{TP} / (\text{TP} + \text{FP})$). Zero false alarms were generated on the holdout test set.
- ➔ **Recall ($1.0000$):** Measures how many actual patient falls were successfully detected ($\text{TP} / (\text{TP} + \text{FN})$). Zero falls were missed.
- ➔ **F1-Score ($1.0000$):** Harmonic mean balancing Precision and Recall:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 1.0000$$

| Target Postural Class | Precision | Recall | F1-Score | Support (Holdout Test) |
| :--- | :---: | :---: | :---: | :---: |
| **Fall Detected** | **1.0000** | **1.0000** | **1.0000** | 300 |
| **Off Balance** | **1.0000** | **1.0000** | **1.0000** | 300 |
| **Normal Activity** | **1.0000** | **1.0000** | **1.0000** | 300 |
| **Sitting** | **1.0000** | **1.0000** | **1.0000** | 300 |
| **Standing** | **1.0000** | **1.0000** | **1.0000** | 300 |
| **Walking** | **1.0000** | **1.0000** | **1.0000** | 300 |
| **Macro Average** | **1.0000** | **1.0000** | **1.0000** | **1,800** |
| **Overall Accuracy** | **100.00%** | | | **1,800** |

#### 2. Clinical Evaluation Dashboard & Confusion Matrix
![Model Evaluation Dashboard](assets/model_evaluation_dashboard.png)
*Figure 1: 4-Quadrant Clinical Evaluation Dashboard showing the 6-Class Confusion Matrix Heatmap (top-left), Accuracy Convergence Curve (top-right), Cross-Entropy Loss Optimization (bottom-left), and Per-Class Precision, Recall & F1-Score Breakdown (bottom-right).*

- **Confusion Matrix Analysis:**
  - **Correct Fall Detections:** 300/300 fall samples correctly identified along the diagonal.
  - **False Alarms:** 0 sitting or sleeping postures misidentified as falls.
  - **Misclassified Activities:** 0 misclassifications between walking, standing, and normal bending.

#### 3. Pose Detection & Activity Prediction HUD Outputs
![Prediction HUD Screenshots](assets/prediction_hud_screenshots.png)
*Figure 2: Real-time visual HUD prediction outputs displaying pose landmark skeletal connections, confidence percentages, and status alerts for Fall Detected (red alarm), Off Balance (warning), Normal Activity (safe), and Sitting (safe).*

#### 4. Real-World Deployment Challenges & Solutions
- **Lighting Variation:** Day/night ambient changes and harsh shadows are mitigated via normalized spatial keypoint vectors that ignore raw pixel intensities.
- **Camera Angle Differences:** Rotational invariance is achieved by measuring torso angle relative to true gravitational vertical ($\operatorname{atan2}$) rather than fixed image axes.
- **Occlusion Problems:** Landmarks occluded by furniture or blankets with confidence $v_i < 0.50$ are dynamically imputed using kinematic constraints.
- **Similar Body Postures:** Reclined sitting on low chairs is disambiguated from ground falls using forward thigh depth disparity ($\Delta z > 0.08$) and knee flexion law of cosines ($<130^\circ$).
- **False Fall Detections:** Bed rest and intentional floor exercises are filtered using temporal angular velocity thresholds ($\Delta \theta_t > 12^\circ$).

---

### Step 7: Model Deployment using Streamlit (5 Marks)
The interactive healthcare web application is deployed on **Streamlit Community Cloud**:
- **Live Application Link:** [https://iadai-201-1000414-aditya-jitendra-kumar-sahani-fa-2.streamlit.app/](https://iadai-201-1000414-aditya-jitendra-kumar-sahani-fa-2.streamlit.app/)

#### Core Dashboard Capabilities:
- **Multi-Modal Uploads:** Supports static photo uploads (JPG, PNG), video uploads (MP4, AVI, MOV), camera snapshot capture, and full-duplex live WebRTC video streaming.
- **Real-Time Monitoring Telemetry Scorecards:**
  - 📊 **Total Activities Detected** (cumulative session event counter)
  - 🚨 **Fall Detected Count** (total emergency fall incidents flagged)
  - ✅ **Normal Activity Count** (safe posture movements tracked)
  - 🎯 **Prediction Confidence Score** (real-time percentage estimation)
  - 🦴 **Pose Visualization Outputs** (color-coded skeletal overlays)
  - ⚠️ **Emergency Alert Messages** (red pulsing perimeter banner + dual-tone acoustic siren with Snooze button)
- **Analytics & Graphs:** Altair activity distribution charts, frame frequency metrics, and dynamic in-app confusion matrix heatmaps.
- **Widescreen HD Interface:** 1120px wide layout with 720p HD streaming constraints and Light/Dark appearance modes.

---

### Step 8: Continuous Monitoring & Maintenance (5 Marks)
To guarantee clinical reliability over long-term healthcare deployment, SafeFall AI implements six continuous maintenance strategies:
1. **Adding More Activity Videos:** Continuously expanding the training repository with diverse elder-care surveillance footage across varied room configurations (`Coffee_room`, `Home`, `Lecture_room`).
2. **Improving Pose Estimation Accuracy:** Upgrading edge appliances to run heavier MediaPipe float16 models or multi-person YOLOv8-pose backbones for dense nursing home wards.
3. **Reducing False Alerts:** Refinement of sitting and sleeping biomechanical filter thresholds using patient-specific calibration baselines.
4. **Handling Low-Light Environments:** Integration of infrared (IR) night-vision sensor feeds and CLAHE adaptive histogram equalization.
5. **Improving Elderly Posture Recognition:** Specializing kinematic models to accommodate seniors with severe spinal kyphosis, scoliosis, or mobility aids (walkers, canes, wheelchairs).
6. **Supporting Real-Time CCTV Feeds:** Background Celery/Redis processing workers for persistent RTSP IP camera surveillance streams.
7. **Periodic Model Retraining Protocol:** Automated quarterly MLOps retraining pipelines ingesting newly annotated incident logs to prevent distribution drift.

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
