# 🛡️ SafeFall AI: Real-Time Human Posture & Fall Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![YOLOv8 Pose](https://img.shields.io/badge/YOLOv8-Pose%20Estimation-green)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SafeFall AI** is an intelligent, real-time computer vision system engineered for activity monitoring and fall detection across assisted living centers, healthcare environments, and home spaces. It leverages **YOLOv8 Pose Estimation** to extract 17 anatomical keypoints, computes biomechanical kinematic descriptors, and performs temporal sequence classification via deep bidirectional neural networks and deterministic posture logic.

---

## 🚀 Key Architectural Features

### 🔐 1. Salted Cryptographic Authentication
- **PBKDF2-HMAC-SHA256**: Passwords are never stored in plaintext; salted with 16 cryptographically secure random bytes and derived over 250,000 iterations.
- **User Management**: Supports multi-account registration, login validation, session persistence, and seamless 1-click guest access.
- **Session State**: Full user session management with automatic logout and control console profile badges.

### 🎥 2. Real-Time Vision & Locomotion Telemetry
- **Streamlit-WebRTC Streaming**: Ultra-low latency 15 FPS video pipeline running in background worker threads with `video_receiver_size=2` to eliminate queue delays.
- **Non-Blocking `@st.fragment` Architecture**: Adopts periodic 1-second telemetry refreshing to eliminate blocking while-loops and prevent browser WebSocket DOM thrashing.
- **Subject Tracking**: Spatial IoU and centroid displacement tracker preserving identity continuity through occlusions.
- **Dynamic On-Frame HUD Overlay**: Real-time posture pill badge, confidence indicator, and colored skeletal joints drawn directly onto video frames in <0.5 ms.
- **Low-Light Contrast Enhancement**: Integrated power-law gamma transformation and Contrast Limited Adaptive Histogram Equalization (CLAHE).
- **Ambient Ring Lighting**: Interactive screen fill-light ring with configurable Warm, Cool, and White tones.

### 📐 3. Biomechanical Kinematics & Deep Learning Engine
- **6 Clinical Posture Classes**: Complete classification across **SITTING**, **STANDING**, **WALKING**, **OFF_BALANCE**, **NORMAL_ACTIVITY**, and **FALL**.
- **PyTorch Temporal BiLSTM Network**: 45-epoch model trained on 15,336 frames from the Le2i dataset (89.55% accuracy, 95.44% precision on unseen test splits).
- **Dual-Engine Redundancy**: Seamless blending of DeepNet probabilities (60%) and deterministic kinematic posture rules (40%).
- **Anatomical Angle Vectors**:
  - 3-point knee joint flexion and hip-to-knee vertical compression.
  - Torso inclination angle relative to gravitational vertical.
  - Dynamic gait oscillation derived from inter-ankle stride separation.
  - Ground-level aspect ratio and high-speed vertical collapse trajectory.
- **Desk Mode**: Specialized upper-body kinematic heuristics when camera framing occludes lower extremities.

### 🚨 4. Emergency SOS & Rapid Hospital Response
- **Single-Touch Circular SOS Button**: Immediate 1-click manual emergency dispatch and automated fall detection links.
- **Google Maps Emergency Hospital Locator**: Direct routing to nearby Level-1 trauma centers, accredited ERs, and geriatric units.
- **Speed-Dial Directory**: One-touch direct dialing to EMS 911, primary geriatricians, on-duty caregivers, and family proxies.
- **Escalating 3-Stage Acoustic Siren**: Synthesizes progressive alarms (Stage 1: 620 Hz chime, Stage 2: 880 Hz siren, Stage 3: 1350 Hz code red).
- **Hysteresis Temporal Filter**: Exponential Moving Average (EMA) and multi-frame streak gating preventing false alarms.
- **Automated Incident Logging**: Confirmed fall incidents automatically write timestamped full-resolution snapshots to `outputs/falls/` and record structured logs in `data/incident_logs.csv`.

---

## 🏗️ System Architecture

```
Camera / Uploaded Video / Static Photo
                 │
                 ▼
   [ Low-Light Preprocessing (CLAHE + Gamma LUT) ]
                 │
                 ▼
      [ YOLOv8 Human Pose Model ]
                 │ (17 Keypoints: x, y, confidence)
                 ▼
   [ Primary Subject Visual Tracker (IoU + Centroid) ]
                 │
                 ▼
   [ Biomechanical Kinematics & Temporal Feature Buffer ]
     - Bounding Box Aspect Ratio (W/H)
     - Torso Inclination Angle (degrees from vertical)
     - Knee & Hip Flexion Angles
     - Inter-Ankle Stride Separation
     - Vertical Descent Velocity
                 │
        ┌────────┴────────────────────┐
        ▼                             ▼
[ 4-Class BiLSTM DeepNet ]   [ Posture Kinematics Engine ]
        └────────┬────────────────────┘
                 │
                 ▼
   [ Temporal Decision Filter (EMA + Hysteresis Alert Gate) ]
                 │
     ┌───────────┴────────────────────────┐
     ▼                                    ▼
Nominal Posture                     FALL DETECTED
(Sitting / Standing / Walking)            │
     │                                    ├── 🚨 Acoustic Siren Alert
     │                                    ├── 📸 Save Snapshot to outputs/falls/
     │                                    └── 📝 Append to Incident CSV Log
     ▼
[ Interactive Streamlit Command Console Dashboard ]
```

---

## 📁 Repository Structure

```
.
├── app.py                      # Main entrypoint and UI orchestration
├── yolov8n-pose.pt             # YOLOv8 nano pose model weights
├── requirements.txt            # Python dependencies
├── packages.txt                # Linux OS package dependencies
├── run_app.bat                 # 1-click Windows launch script
├── push_to_github.bat          # 1-click GitHub synchronization script
├── core/                       # Core analytical and machine learning modules
│   ├── security.py             # Salted PBKDF2 authentication vault
│   ├── kinematics.py           # Biomechanical angle and posture heuristics
│   ├── body_tracker.py         # Spatial IoU & centroid person tracker
│   ├── temporal_filter.py      # Exponential smoothing & hysteresis gate
│   └── vision_pipeline.py      # YOLOv8 pipeline & WebRTC streaming worker
├── ui/                         # Presentation and design layer
│   ├── styles.py               # Glassmorphic CSS, themes, and animations
│   ├── components.py           # SVG emblems, cards, charts, and siren
│   ├── auth_view.py            # Login, registration, and guest cards
│   └── dashboard_view.py       # Live monitor, photo, video, SOS, overview views
├── model/                      # Trained AI models & detectors
│   ├── safefall_nn_model.pth   # PyTorch BiLSTM neural classifier
│   ├── feature_scaler.joblib   # Robust feature standardizer
│   └── pose_detector.py        # Pose estimation and feature extraction
├── outputs/                    # Output artifacts
│   ├── users.json              # Salted user credentials
│   └── falls/                  # Incident snapshots directory
└── tests/                      # Automated test suites
    ├── test_all_6_classes.py   # 6-class activity verification
    └── test_sos_alarm.py       # Emergency SOS and siren verification
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA.git
cd "ML & DL FA 2"
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Application
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
