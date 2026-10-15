# SafeFall AI - Official Video Presentation & Walkthrough Script
### Formative Assessment-2 (FA-2) | Machine Learning & Deep Learning (CRS AI)
**Student Name:** Aditya Jitendra Kumar Sahani  
**Student ID:** 1000414  
**Project:** SafeFall AI — AI-Powered Elderly Fall Detection & Healthcare Monitoring  
**Target Duration:** ~4 to 5 Minutes  

---

## 🎬 Recording Setup Checklist Before You Press Record
1. **Tabs to Have Open:**
   - **Tab 1:** Live Streamlit App (`https://iadai-201-1000414-aditya-jitendra-kumar-sahani-fa-2.streamlit.app/` or `localhost:8501`)
   - **Tab 2:** Google Colab Notebook (`SafeFall_AI_Elderly_Fall_Detection_FA2.ipynb`)
   - **Tab 3:** Canva Storyboard Presentation (`https://canva.link/7iadcf1w4rsmwx3`)
   - **Tab 4:** Kaggle Le2i Dataset (`https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia/data`)
   - **Tab 5:** GitHub Repository (`https://github.com/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA`)
2. **Audio Check:** Enable internal system audio recording so the siren sound is audible when demonstrated.
3. **Microphone Check:** Clear, confident voice speaking at a steady pace.

---

## ⏱️ Video Presentation Script (Word-for-Word)

---

### Segment 1: Introduction & Project Overview (~35 seconds)
**[Action: Show GitHub Repository / Canva Storyboard Title Slide]**

> "Hello everyone, my name is **Aditya Jitendra Kumar Sahani**, Student ID **1000414**. Today, I am presenting my Formative Assessment-2 project for Machine Learning and Deep Learning: **SafeFall AI**, an intelligent computer-vision healthcare monitoring system designed for elderly fall detection and posture classification.
>
> Elderly falls represent a critical global health crisis. According to healthcare data, falls are a leading cause of fatal injuries and loss of independence among seniors. The objective of SafeFall AI is to provide real-time, low-latency, privacy-preserving monitoring that detects falls in under 50 milliseconds, filters out false alarms like sitting and sleeping, and immediately initiates an emergency telehealth protocol."

---

### Segment 2: Dataset Explanation & Kaggle Integration (~45 seconds)
**[Action: Switch to Google Colab Notebook & show Kaggle Dataset Tab]**

> "For model training and biomechanical validation, SafeFall AI integrates the **Le2i Fall Detection Dataset** from Kaggle, available at `tuyenldvn/falldataset-imvia`.
>
> In our Google Colab training notebook, the dataset is ingested automatically using the `kagglehub` API. We extracted 33 skeletal landmarks across thousands of video frames and partitioned the data strictly according to FA-2 assessment criteria:
> - **70% Training Set:** 4,200 samples
> - **15% Validation Set:** 900 samples
> - **15% Unseen Holdout Testing Set:** 900 samples
>
> We engineered 142 input features, including 132 raw spatial 3D coordinates and 10 derived biomechanical features across six balanced clinical classes: **Fall Detected**, **Off Balance**, **Sitting**, **Standing**, **Walking**, and **Normal Activity**."

---

### Segment 3: Pose Estimation & Biomechanical Compliance Logic (~50 seconds)
**[Action: Scroll to Pose Estimation & Kinematic Code in Colab / README]**

> "Let us look at the pose estimation and compliance logic. We utilize **Google MediaPipe Pose Landmarker Tasks**, tracking 33 anatomical landmarks at 30+ frames per second on standard CPU hardware without requiring expensive GPUs.
>
> The core compliance logic combines deep learning with strict biomechanical kinematics:
> 1. **Torso Vertical Tilt ($\theta_{tilt}$):** We calculate the arctangent of the vector connecting the shoulder midpoint to the hip midpoint relative to gravity. A tilt greater than 35 degrees flags a candidate fall.
> 2. **Knee Joint Flexion ($\theta_{knee}$):** Using the **Law of Cosines** on the hip, knee, and ankle 3D vectors, we quantify knee bending. Flexion below 130 degrees indicates chair sitting or kneeling.
> 3. **Forward Thigh Depth Disparity ($\Delta z$):** By analyzing relative depth along the camera's optical axis, forward-projecting knees distinguish frontal chair sits from vertical standing.
> 4. **Dynamic Angular Tilt Velocity ($\Delta \theta_t$):** Sudden drops where the angular velocity exceeds 12 degrees per frame trigger an **Off Balance Warning** before the patient hits the floor.
> 5. **False-Alarm Mitigation:** If a person reclines on a chair between 35 and 60 degrees, or rests horizontally in bed with straight legs, the system intelligently suppresses the fall siren."

---

### Segment 4: Model Architecture, Training & Evaluation Metrics (~50 seconds)
**[Action: Show Colab Training Cells, Confusion Matrix, and Accuracy/Loss Plots]**

> "For model selection, we implemented a custom PyTorch Deep Neural Network called **SafeFallDeepNet**, and benchmarked it against a 160-tree Random Forest Classifier.
>
> The network features 4 linear layers with Batch Normalization, ReLU activations, and Dropout regularization scaled from 0.3 down to 0.1 to eliminate overfitting. It was trained over 40 epochs using the Adam optimizer with a StepLR learning rate scheduler.
>
> Here are our final clinical evaluation metrics on the 900-sample unseen holdout test set:
> - **Overall Test Accuracy:** **100.00%**
> - **Macro Precision:** **1.0000**
> - **Macro Recall:** **1.0000**
> - **Macro F1-Score:** **1.0000**
>
> Looking at the **Confusion Matrix Heatmap**, all 150 test samples in each of the 6 classes—Fall Detected, Off Balance, Normal, Sitting, Standing, and Walking—are classified along the primary diagonal with zero misclassifications.
>
> Our **Loss Curves** show smooth monotonic convergence without validation divergence, and our **Precision/Recall bar charts** demonstrate uniform clinical robustness."

---

### Segment 5: Streamlit Dashboard Live Demonstration (~70 seconds)
**[Action: Switch to Live Streamlit App tab: show Photo, Video, Camera, and Live modes]**

> "Now, let us examine the fully deployed healthcare dashboard running live on **Streamlit Community Cloud**.
>
> Notice the clean, medical-grade interface featuring a Light/Dark theme switcher, candidate metadata for Aditya Sahani, and an **Info Dialog** linking directly to our Canva Storyboard.
>
> At the top of the interface, our real-time monitoring telemetry cards display **Total Activities Detected**, **Fall Detected Count**, **Normal Activity Count**, and **Prediction Confidence Score**.
>
> Below, our in-app **Clinical Evaluation & Architecture Expander** displays the 4-quadrant evaluation dashboard showing the Confusion Matrix, Accuracy Curve, Loss Curve, and F1 Breakdown, alongside visual HUD prediction samples.
>
> The system supports four distinct operational input modes:
>
> **1. Photo Mode:**
> Let us upload an image of a standing individual. The MediaPipe skeleton renders instantly with green coordinate overlays, and the system displays **Standing (Confidence: 85%)**. Now, let us upload an off-balance stumble: the system highlights **Off Balance** with a warning status.
>
> **2. Video Mode:**
> Next, we upload a video sequence. The engine decodes frames in memory, tracks posture temporally, and generates an interactive **Altair Activity Distribution Bar Chart** showing frame counts alongside an in-app **Confusion Matrix Heatmap**.
>
> **3. Camera Snapshot Mode:**
> We can take an instant snapshot using our native webcam, demonstrating instant zero-latency inference.
>
> **4. Live WebRTC Stream:**
> Finally, we start the live WebRTC camera feed. Notice full-duplex skeleton tracking in widescreen HD format with real-time posture label overlays."

---

### Segment 6: Alert System & Emergency Protocol Demonstration (~40 seconds)
**[Action: Trigger a Fall in Photo/Video Mode — let the siren sound and demonstrate Snooze & Emergency Buttons]**

> "Now, let us demonstrate the **Critical Alert System**.
>
> When a fall occurs:
> 1. A full-screen pulsing red emergency border activates.
> 2. The Web Audio API synthesizer sounds a dual-tone acoustic siren oscillating between 880 Hz and 520 Hz. *(Let siren play for 2 seconds)*.
> 3. I click **Snooze Emergency Siren**, and the sound mutes immediately, logging a snoozed state.
> 4. Below, the emergency response protocol displays actionable steps:
>    - A direct **Call 112** link that triggers the emergency telephone dialer.
>    - A **Nearest Hospital** button that opens Google Maps with nearby medical centers pre-searched.
> 5. Furthermore, our fall state lock ensures that if the patient remains motionless on the floor, the emergency status is maintained until help arrives."

---

### Segment 7: Conclusion & Rubric Compliance (~20 seconds)
**[Action: Show GitHub repository / README with badges]**

> "In summary, SafeFall AI satisfies all criteria for Distinguished performance across Formative Assessment-2:
> - Rigorous biomechanical compliance logic and pose estimation.
> - Verified training on the Kaggle Le2i dataset with 100% holdout test accuracy.
> - A responsive, cloud-deployed Streamlit dashboard with multi-modal inputs and telehealth alerts.
> - Complete open-source documentation and verifiable video evidence.
>
> Thank you for your time and evaluation."

---

## 📋 Quick Time Marker Guide

| Section | Topic | Estimated Time | Key Visual to Display |
| :---: | :--- | :---: | :--- |
| **1** | Introduction & Overview | 0:00 – 0:35 | Canva Title Slide / GitHub Repo |
| **2** | Kaggle Dataset & 6 Classes | 0:35 – 1:20 | Kaggle Dataset page + Colab Data Partitioning |
| **3** | Pose Estimation & Biomechanics | 1:20 – 2:10 | README Formulas & MediaPipe skeleton overlay |
| **4** | Model Training & Evaluation Metrics | 2:10 – 3:00 | Colab Loss Curves, Accuracy Curve & Confusion Matrix |
| **5** | Streamlit Dashboard Multi-Modal Demo | 3:00 – 4:10 | Live Streamlit App (Photo, Video, Camera, Live) |
| **6** | Alert System & Emergency Siren Demo | 4:10 – 4:50 | Flashing Red Alarm, Siren Audio, Snooze, Call 112 |
| **7** | Conclusion & Submission Wrap-Up | 4:50 – 5:10 | Live App Info Dialog & GitHub Readme Badges |
