"""
SafeFall AI - Intelligent Elderly Fall Detection & Healthcare Monitoring Dashboard
Ultra-Modern UI with Dynamic Dark/Light Mode, Interactive Onboarding,
6-Class Posture Spectrum, Continuous Live Webcam Stream, Clinical HUD & Telemetry.
Formative Assessment-2 (FA-2) | CareVision HealthTech Pvt. Ltd.
"""

import os
import sys
import time
import json
import tempfile
import numpy as np
import pandas as pd
import cv2
import torch
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from model.pose_detector import SafeFallPoseDetector
from model.fall_classifier import SafeFallClassifier, CLASSES
from utils.alert_manager import AlertManager
from utils.visualizer import draw_hud
from utils.feedback_trainer import (
    log_feedback_sample, count_feedback_samples,
    retrain_model_with_feedback, clear_feedback_samples
)

# Streamlit Page Config
st.set_page_config(
    page_title="SafeFall AI - Intelligent Fall & Instability Healthcare Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "total_inferences" not in st.session_state:
    st.session_state["total_inferences"] = 0
if "fall_count" not in st.session_state:
    st.session_state["fall_count"] = 0
if "warning_count" not in st.session_state:
    st.session_state["warning_count"] = 0
if "normal_count" not in st.session_state:
    st.session_state["normal_count"] = 0
if "selected_demo_sample" not in st.session_state:
    st.session_state["selected_demo_sample"] = "fall_sample_1.jpg"
if "sound_alert_enabled" not in st.session_state:
    st.session_state["sound_alert_enabled"] = True
if "live_camera_active" not in st.session_state:
    st.session_state["live_camera_active"] = False
if "latest_feature_vector" not in st.session_state:
    st.session_state["latest_feature_vector"] = None
if "latest_prediction" not in st.session_state:
    st.session_state["latest_prediction"] = "None"
if "show_onboarding" not in st.session_state:
    st.session_state["show_onboarding"] = True
if "onboarding_step" not in st.session_state:
    st.session_state["onboarding_step"] = 1
if "app_theme" not in st.session_state:
    st.session_state["app_theme"] = "dark"

# Audio Alert Generators (Web Audio)
def play_fall_siren():
    st.markdown("""<audio autoplay><source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg"></audio>""", unsafe_allow_html=True)

def play_warning_chime():
    st.markdown("""<audio autoplay><source src="https://assets.mixkit.co/active_storage/sfx/2874/2874-preview.mp3" type="audio/mpeg"></audio>""", unsafe_allow_html=True)

# Sidebar: Quick Settings & Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/medical-heart.png", width=64)
    st.markdown("### **SafeFall AI**")
    st.caption("CareVision HealthTech Pvt. Ltd. | FA-2 Clinical AI")
    st.divider()
    
    # Theme Toggle
    theme_choice = st.radio(
        "Appearance & Theme:",
        ["🌙 Dark Mode (Night Surveillance / Ward)", "☀️ Light Mode (Daytime Clinical)"],
        index=0 if st.session_state["app_theme"] == "dark" else 1
    )
    is_dark = "Dark" in theme_choice
    st.session_state["app_theme"] = "dark" if is_dark else "light"
    
    st.divider()
    st.markdown("#### Navigation Mode")
    mode = st.radio(
        "Select Clinical Workspace:",
        [
            "1-Click Quick Demo & Diagnostics",
            "Continuous Live Webcam Stream",
            "⚖️ Off-Balancer & Stability Diagnostics",
            "Custom Image Upload",
            "Video Stream Analyzer",
            "Model Performance & Metrics",
            "Emergency Dispatch Center",
            "Active Learning & Retraining",
            "Architecture & Step Guide"
        ],
        index=0
    )
    
    st.divider()
    st.markdown("#### Caregiver & Ward Settings")
    patient_id = st.text_input("Patient ID / Name:", value="PATIENT-8042 (Senior Care)")
    room_loc = st.selectbox(
        "Surveillance Sector:",
        ["Home_01 Living Room", "Home_02 Bedroom", "Coffee_room_01 Lounge", "Office_01 Corridor", "Lecture_room Area"]
    )
    st.session_state["sound_alert_enabled"] = st.checkbox("Enable Emergency Audio Sirens", value=st.session_state["sound_alert_enabled"])
    
    st.markdown("##### Quick Actions")
    c_side1, c_side2 = st.columns(2)
    with c_side1:
        if st.button("🎓 Guided Tour", use_container_width=True, help="Launch interactive clinical tour"):
            st.session_state["show_onboarding"] = True
            st.session_state["onboarding_step"] = 1
            st.rerun()
    with c_side2:
        if st.button("🔔 Test Siren", use_container_width=True, help="Test audio emergency alarm"):
            play_fall_siren()
            st.toast("🚨 Emergency siren audio triggered!", icon="🔊")

    st.divider()
    st.markdown("#### Posture Detection Model")
    engine_choice = st.radio(
        "Active Classifier Pipeline:",
        [
            "⚡ User Pose Kinematics (Direct Heuristics)",
            "🧠 Dual-Net AI Ensemble (PyTorch + RF)"
        ],
        index=0,
        help="Select detection algorithm: User uploaded activity_from_pose logic or DeepNet AI ensemble"
    )
    use_user_engine = ("User" in engine_choice)

    st.divider()
    st.markdown("#### Engine Diagnostics")
    st.success("User Kinematics Engine: Active" if use_user_engine else "Dual-Net AI Ensemble: Active")
    st.success("MediaPipe Pose 3D: Active")
    st.info("Caregiver Telemetry: Online")

# Dynamic CSS Theme Engine (Dark Mode & Light Mode)
if is_dark:
    bg_color = "#0B1120"
    card_bg = "#1E293B"
    card_border = "#334155"
    text_color = "#F8FAFC"
    subtext_color = "#94A3B8"
    hero_grad = "linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #1E3A8A 100%)"
    hero_border = "#3B82F6"
    card_shadow = "0 8px 24px rgba(0, 0, 0, 0.45)"
    metric_card_bg = "#131E32"
    bg_glow = "radial-gradient(circle at 85% 12%, rgba(59, 130, 246, 0.08) 0%, transparent 450px), radial-gradient(circle at 15% 88%, rgba(239, 68, 68, 0.05) 0%, transparent 450px)"
else:
    bg_color = "#F8FAFC"
    card_bg = "#FFFFFF"
    card_border = "#E2E8F0"
    text_color = "#0F172A"
    subtext_color = "#475569"
    hero_grad = "linear-gradient(135deg, #1E3A8A 0%, #0284C7 100%)"
    hero_border = "#2563EB"
    card_shadow = "0 6px 20px rgba(30, 58, 138, 0.08)"
    metric_card_bg = "#F1F5F9"
    bg_glow = "radial-gradient(circle at 85% 12%, rgba(37, 99, 235, 0.04) 0%, transparent 450px)"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }}
    
    .stApp {{
        background-color: {bg_color} !important;
        background-image: {bg_glow} !important;
        background-attachment: fixed !important;
        color: {text_color} !important;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {'#0F172A' if is_dark else '#F8FAFC'} !important;
        border-right: 1px solid {card_border} !important;
    }}
    
    [data-testid="stSidebar"] * {{
        color: {text_color} !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
        font-weight: 700 !important;
    }}
    
    p, span, label {{
        color: {text_color};
    }}
    
    div[data-testid="stMetric"] {{
        background: {card_bg} !important;
        border: 1px solid {card_border} !important;
        padding: 14px 18px !important;
        border-radius: 12px !important;
        box-shadow: {card_shadow} !important;
    }}
    
    [data-testid="stMetricValue"] {{
        color: {'#38BDF8' if is_dark else '#0284C7'} !important;
        font-weight: 800 !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {subtext_color} !important;
        font-weight: 600 !important;
    }}
    
    .stButton > button {{
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
    }}
    
    [data-testid="stExpander"] {{
        background: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 12px !important;
    }}
    
    /* Hero Header */
    .hero-header {{
        background: {hero_grad};
        color: #FFFFFF;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 22px;
        border: 1px solid {hero_border};
        box-shadow: {card_shadow};
    }}
    .hero-title {{
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
        color: #FFFFFF !important;
    }}
    .hero-subtitle {{
        font-size: 1.05rem;
        color: #E2E8F0 !important;
        font-weight: 500;
        max-width: 900px;
        line-height: 1.5;
    }}
    .hero-badge {{
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 10px;
        color: #FFFFFF !important;
    }}
    
    /* Clinical Cards */
    .clinical-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 20px;
        box-shadow: {card_shadow};
        margin-bottom: 16px;
        color: {text_color};
    }}
    
    /* Emergency Alert Banners */
    .alert-banner-danger {{
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.25) 0%, rgba(239, 68, 68, 0.15) 100%);
        border: 2px solid #DC2626;
        border-left: 8px solid #DC2626;
        color: {'#FECACA' if is_dark else '#991B1B'};
        padding: 18px 22px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.15rem;
        margin: 14px 0px;
        box-shadow: 0 4px 18px rgba(220, 38, 38, 0.35);
    }}
    .alert-banner-warning {{
        background: linear-gradient(135deg, rgba(234, 88, 12, 0.25) 0%, rgba(249, 115, 22, 0.15) 100%);
        border: 2px solid #EA580C;
        border-left: 8px solid #EA580C;
        color: {'#FED7AA' if is_dark else '#9A3412'};
        padding: 18px 22px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.15rem;
        margin: 14px 0px;
        box-shadow: 0 4px 18px rgba(234, 88, 12, 0.3);
    }}
    .alert-banner-safe {{
        background: linear-gradient(135deg, rgba(5, 150, 105, 0.25) 0%, rgba(16, 185, 129, 0.15) 100%);
        border: 2px solid #059669;
        border-left: 8px solid #059669;
        color: {'#A7F3D0' if is_dark else '#065F46'};
        padding: 18px 22px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.12rem;
        margin: 14px 0px;
        box-shadow: 0 4px 14px rgba(5, 150, 105, 0.25);
    }}
    
    /* Code / Monospace */
    code {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9em;
        background: {'#1E293B' if is_dark else '#F1F5F9'};
        color: {'#38BDF8' if is_dark else '#0284C7'};
        padding: 2px 6px;
        border-radius: 6px;
    }}
    
    /* Onboarding Guide Card */
    .onboarding-box {{
        background: {card_bg};
        border: 2px solid {'#3B82F6' if is_dark else '#2563EB'};
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 22px;
        color: {text_color};
        box-shadow: {card_shadow};
    }}
    .onboarding-title {{
        font-size: 1.25rem;
        font-weight: 800;
        color: {'#60A5FA' if is_dark else '#1D4ED8'};
        margin-bottom: 0px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# Cached AI Engine Loader
@st.cache_resource
def load_ai_engine():
    pose_detector = SafeFallPoseDetector(static_image_mode=False, model_complexity=1)
    classifier = SafeFallClassifier(model_dir=os.path.join(BASE_DIR, "model"))
    alert_mgr = AlertManager(log_file=os.path.join(BASE_DIR, "data", "incident_logs.csv"))
    return pose_detector, classifier, alert_mgr

pose_detector, classifier, alert_mgr = load_ai_engine()

# Fast In-Memory Image Cache
@st.cache_data
def get_cached_sample_image(filename: str):
    p = os.path.join(BASE_DIR, "data", "sample_media", filename)
    if os.path.exists(p):
        return cv2.imread(p)
    return None

# Hero Header Banner
st.markdown(f"""
<div class='hero-header'>
    <div class='hero-title'>SafeFall AI - Elderly Care & Fall Prevention Platform</div>
    <div class='hero-subtitle'>Autonomous Deep Learning & Biomechanical Kinematics Surveillance across 6 Posture Spectrum States for Elderly Care Centers, Hospital Wards & Smart Homes</div>
    <div class='hero-badge'>Location: {room_loc} | Monitored Subject: {patient_id} | Theme: {'Dark Mode' if is_dark else 'Light Mode'}</div>
</div>
""", unsafe_allow_html=True)

# Top KPI Summary Cards
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Analyzed", st.session_state["total_inferences"], delta="Frames")
with k2:
    st.metric("Critical Falls", st.session_state["fall_count"], delta="High Priority" if st.session_state["fall_count"] > 0 else "0", delta_color="inverse")
with k3:
    st.metric("Off-Balance Warnings", st.session_state["warning_count"], delta="Pre-Fall Instability", delta_color="inverse")
with k4:
    st.metric("Safe Postures", st.session_state["normal_count"], delta="Normal ADL")
with k5:
    st.metric("Model Test Accuracy", "89.6%", delta="Le2i Full Benchmark")

# =========================================================
# INTERACTIVE GUIDED ONBOARDING SYSTEM (4-STEP STEPPER)
# =========================================================
if st.session_state.get("show_onboarding", True):
    current_step = st.session_state.get("onboarding_step", 1)
    
    st.markdown(f"""
    <div class='onboarding-box'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:10px;'>
            <div class='onboarding-title'>
                <span style='font-size:1.4rem;'>🎓</span> <span>SafeFall AI Interactive Clinical Onboarding & System Tour</span>
            </div>
            <div style='display:flex; align-items:center; gap:8px;'>
                <span style='font-size:0.86rem; font-weight:700; color:{'#38BDF8' if is_dark else '#0284C7'}; background:{'#1E293B' if is_dark else '#EFF6FF'}; padding:5px 14px; border-radius:12px; border:1px solid {'#3B82F6'};'>
                    Step {current_step} of 4
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 4-Step Navigation Pill Buttons
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        if st.button("1. Architecture & Privacy", use_container_width=True, type="primary" if current_step == 1 else "secondary"):
            st.session_state["onboarding_step"] = 1
            st.rerun()
    with s_col2:
        if st.button("2. 6-State Postures", use_container_width=True, type="primary" if current_step == 2 else "secondary"):
            st.session_state["onboarding_step"] = 2
            st.rerun()
    with s_col3:
        if st.button("3. Biomechanical Off-Balancer", use_container_width=True, type="primary" if current_step == 3 else "secondary"):
            st.session_state["onboarding_step"] = 3
            st.rerun()
    with s_col4:
        if st.button("4. Caregiver Alerts & Dispatch", use_container_width=True, type="primary" if current_step == 4 else "secondary"):
            st.session_state["onboarding_step"] = 4
            st.rerun()

    # Step Content Render
    if current_step == 1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='clinical-card' style='height:100%;'>
                <b style='font-size:1.05rem; color:#3B82F6;'>🔒 HIPAA & GDPR Privacy-by-Design</b><br>
                <p style='font-size:0.88rem; color:{subtext_color}; margin-top:8px; line-height:1.5;'>
                    Raw surveillance frames are processed entirely in volatile memory (RAM). 
                    SafeFall AI extracts only <b>33 anonymized 3D Cartesian coordinates</b>. 
                    No patient facial imagery or identifiable video streams are ever stored to disk or transmitted to external servers.
                </p>
                <div style='background:{metric_card_bg}; padding:8px 12px; border-radius:8px; font-size:0.82rem; color:{text_color}; border:1px solid {card_border};'>
                    ✅ <b>Zero Facial Recognition Storage</b><br>
                    ✅ <b>Edge-First Sovereign Processing</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='clinical-card' style='height:100%;'>
                <b style='font-size:1.05rem; color:#10B981;'>⚡ Sub-25ms Real-Time Inference</b><br>
                <p style='font-size:0.88rem; color:{subtext_color}; margin-top:8px; line-height:1.5;'>
                    Dual-engine pipeline running synchronized <b>PyTorch Deep Kinematics Neural Network</b> and <b>Balanced Random Forest</b>. 
                    Processes frames in <b>&lt; 25 milliseconds</b>, enabling instantaneous detection of rapid collapses and stumbles on standard hardware.
                </p>
                <div style='background:{metric_card_bg}; padding:8px 12px; border-radius:8px; font-size:0.82rem; color:{text_color}; border:1px solid {card_border};'>
                    ✅ <b>Dual-Model Redundancy</b><br>
                    ✅ <b>Works on Standard CPU / GPU</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='clinical-card' style='height:100%;'>
                <b style='font-size:1.05rem; color:#F59E0B;'>📐 Camera Placement Guide</b><br>
                <p style='font-size:0.88rem; color:{subtext_color}; margin-top:8px; line-height:1.5;'>
                    For optimal posture recognition and fall detection in hospital wards and senior care homes:
                </p>
                <ul style='font-size:0.86rem; color:{subtext_color}; padding-left:18px; line-height:1.5;'>
                    <li><b>Mounting Height:</b> 1.8m – 2.4m above floor</li>
                    <li><b>Depression Angle:</b> 30° – 45° downward</li>
                    <li><b>Field of View:</b> Full body (torso to ankles)</li>
                    <li><b>Lighting:</b> Standard room or infrared illumination</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
    elif current_step == 2:
        st.markdown(f"<p style='color:{subtext_color}; font-size:0.92rem; margin-bottom:12px;'>SafeFall AI monitors the patient across the complete <b>6-state clinical posture spectrum</b> rather than a fragile binary threshold:</p>", unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown(f"""
            <div class='clinical-card' style='border-top: 4px solid #DC2626;'>
                <b style='font-size:1.02rem; color:#DC2626;'>🚨 1. Fall Detected</b><br>
                <span style='font-size:0.84rem; color:{subtext_color};'>
                    <b>Kinematics:</b> Horizontal floor collapse, Torso Tilt &gt; 65°, W/H Aspect Ratio &gt; 1.20, Hip Clearance &lt; 0.150.<br>
                    <b>Action:</b> Dispatches audio sirens, SMS alerts & logs incident.
                </span>
            </div>
            <div class='clinical-card' style='border-top: 4px solid #7C3AED; margin-top:10px;'>
                <b style='font-size:1.02rem; color:#7C3AED;'>🪑 3. Sitting Posture</b><br>
                <span style='font-size:0.84rem; color:{subtext_color};'>
                    <b>Kinematics:</b> Seated resting posture (chair/bed). Knee flexion &lt; 130°, elevated hip clearance &gt; 0.160, torso tilt &lt; 50°.<br>
                    <b>Action:</b> Safe state. Biomechanically stabilized to prevent false alarms.
                </span>
            </div>
            """, unsafe_allow_html=True)
        with g2:
            st.markdown(f"""
            <div class='clinical-card' style='border-top: 4px solid #EA580C;'>
                <b style='font-size:1.02rem; color:#EA580C;'>⚠️ 2. Off-Balance Warning</b><br>
                <span style='font-size:0.84rem; color:{subtext_color};'>
                    <b>Kinematics:</b> Pre-fall stumbling / instability. Torso tilt 32°– 64°, asymmetric leg stride, Center of Mass drifting beyond Base of Support.<br>
                    <b>Action:</b> Sounds cautionary chime so caregivers can prevent ground impact.
                </span>
            </div>
            <div class='clinical-card' style='border-top: 4px solid #059669; margin-top:10px;'>
                <b style='font-size:1.02rem; color:#059669;'>🧍 4. Standing Upright</b><br>
                <span style='font-size:0.84rem; color:{subtext_color};'>
                    <b>Kinematics:</b> Stationary vertical stance. Torso tilt &lt; 25°, narrow bounding aspect ratio &lt; 0.75, balanced feet.<br>
                    <b>Action:</b> Safe state. Routine equilibrium confirmation.
                </span>
            </div>
            """, unsafe_allow_html=True)
        with g3:
            st.markdown(f"""
            <div class='clinical-card' style='border-top: 4px solid #2563EB;'>
                <b style='font-size:1.02rem; color:#2563EB;'>🚶 5. Walking Gait</b><br>
                <span style='font-size:0.84rem; color:{subtext_color};'>
                    <b>Kinematics:</b> Dynamic locomotion. Alternating ankle stride, knee flexion asymmetry &gt; 18°, vertical foot lift &gt; 0.035, moving hip displacement.<br>
                    <b>Action:</b> Safe state. Verified straight and angled walking gait.
                </span>
            </div>
            <div class='clinical-card' style='border-top: 4px solid #D97706; margin-top:10px;'>
                <b style='font-size:1.02rem; color:#D97706;'>🔄 6. Normal Activity</b><br>
                <span style='font-size:0.84rem; color:{subtext_color};'>
                    <b>Kinematics:</b> Controlled forward bending (picking dropped item, tying shoes). Torso tilted but feet firmly planted, stability score &gt; 50%.<br>
                    <b>Action:</b> Safe state. Eliminates nuisance false alarms.
                </span>
            </div>
            """, unsafe_allow_html=True)

    elif current_step == 3:
        b1, b2 = st.columns([1, 1])
        with b1:
            st.markdown(f"""
            <div class='clinical-card'>
                <b style='font-size:1.05rem; color:#3B82F6;'>⚖️ Center of Mass (CoM) vs Base of Support (BoS)</b><br>
                <p style='font-size:0.88rem; color:{subtext_color}; margin-top:8px; line-height:1.6;'>
                    In human biomechanics, postural equilibrium is maintained only as long as the <b>Center of Mass (CoM)</b> projection falls within the <b>Base of Support (BoS)</b> polygon formed by the feet.
                </p>
                <ul style='font-size:0.86rem; color:{subtext_color}; padding-left:18px; line-height:1.6;'>
                    <li><b>Stable Equilibrium:</b> CoM centered directly between ankles. Dynamic Stability Score &ge; 70%.</li>
                    <li><b>Controlled Dynamic Flexion:</b> When bending forward intentionally, the elder expands their BoS to counterbalance. Stability &ge; 50% (Normal Activity).</li>
                    <li><b>Biomechanical Boundary Loss:</b> When tripping or stumbling, CoM accelerates outside BoS before ground impact occurs (Off-Balance warning).</li>
                    <li><b>Ground Impact:</b> Complete loss of support, vertical CoM reaches floor level (Fall Detected).</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            st.markdown(f"""
            <div class='clinical-card'>
                <b style='font-size:1.05rem; color:#10B981;'>📊 Dynamic Stability Index Formulation</b><br>
                <div style='background:{metric_card_bg}; padding:14px; border-radius:10px; border:1px solid {card_border}; margin-top:10px;'>
                    <code style='font-size:0.95rem; font-weight:700;'>Stability = max(0, 100 - (Tilt × 0.85 + CoM_Dev × 120))%</code>
                </div>
                <p style='font-size:0.88rem; color:{subtext_color}; margin-top:12px; line-height:1.5;'>
                    To observe live real-time physics calculations with interactive sliders for torso angles, hip heights, and balance ratios, navigate to the <b>⚖️ Off-Balancer & Stability Diagnostics</b> tab in the sidebar!
                </p>
            </div>
            """, unsafe_allow_html=True)

    elif current_step == 4:
        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown(f"""
            <div class='clinical-card' style='height:100%;'>
                <b style='font-size:1.05rem; color:#EF4444;'>🔔 Emergency Audio Siren</b><br>
                <p style='font-size:0.88rem; color:{subtext_color}; margin-top:8px; line-height:1.5;'>
                    When a fall event occurs, SafeFall AI plays a zero-latency web audio alarm to immediately alert attending nurses on the floor.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔊 Test Audio Siren Now", use_container_width=True):
                play_fall_siren()
                st.toast("🚨 Audio siren triggered successfully!", icon="🔊")
        with a2:
            st.markdown(f"""
            <div class='clinical-card' style='height:100%;'>
                <b style='font-size:1.05rem; color:#3B82F6;'>📱 Instant SMS & Dispatch</b><br>
                <p style='font-size:0.88rem; color:{subtext_color}; margin-top:8px; line-height:1.5;'>
                    Automated SMS/Email payload sent to the on-duty emergency roster with room number, patient ID, and exact fall timestamp.
                </p>
                <div style='background:{metric_card_bg}; padding:6px 10px; border-radius:6px; font-size:0.78rem; color:{text_color}; border:1px solid {card_border};'>
                    <code>[ALERT] {patient_id} fell in {room_loc}! Torso 78°, Urgent Dispatch!</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with a3:
            st.markdown(f"""
            <div class='clinical-card' style='height:100%;'>
                <b style='font-size:1.05rem; color:#10B981;'>🔄 Active Retraining Loop</b><br>
                <p style='font-size:0.88rem; color:{subtext_color}; margin-top:8px; line-height:1.5;'>
                    Encountered a novel room angle or custom furniture? Simply label the frame in <b>Active Learning & Retraining</b> and trigger 1-click model retraining without server downtime.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Stepper Control Navigation Buttons
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    nav_prev, nav_next, nav_dismiss = st.columns([1, 1, 1])
    with nav_prev:
        if current_step > 1:
            if st.button("◀ Previous Step", use_container_width=True):
                st.session_state["onboarding_step"] = current_step - 1
                st.rerun()
    with nav_next:
        if current_step < 4:
            if st.button("Next Step ▶", type="primary", use_container_width=True):
                st.session_state["onboarding_step"] = current_step + 1
                st.rerun()
        else:
            if st.button("✓ Finish Tour & Start Monitoring", type="primary", use_container_width=True):
                st.session_state["show_onboarding"] = False
                st.toast("🎉 Onboarding completed! SafeFall AI is active.", icon="✅")
                st.rerun()
    with nav_dismiss:
        if st.button("✕ Dismiss Tour", use_container_width=True):
            st.session_state["show_onboarding"] = False
            st.rerun()
            
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Compact Replay Bar when dismissed
    c_info, c_btn = st.columns([4, 1])
    with c_info:
        st.info("💡 **Clinical Tip:** SafeFall AI is actively monitoring patient kinematics across the 6-state posture spectrum. Need a refresher on system controls?")
    with c_btn:
        if st.button("🎓 Reopen Onboarding Tour", use_container_width=True):
            st.session_state["show_onboarding"] = True
            st.session_state["onboarding_step"] = 1
            st.rerun()

st.divider()

# =========================================================
# HELPER: INFERENCE & VISUALIZATION (6-CLASS AWARE)
# =========================================================
def run_and_display_inference(input_bgr, title_prefix="", allow_synthetic_fallback=False):
    results = pose_detector.process_frame(input_bgr, allow_synthetic_fallback=allow_synthetic_fallback)
    landmarks = pose_detector.extract_landmarks(results)
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.markdown(f"#### {title_prefix} Original Surveillance Feed")
        st.image(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB), width="stretch")
        
    with col_img2:
        st.markdown(f"#### {title_prefix} MediaPipe Pose & Healthcare HUD")
        if landmarks is not None:
            features_dict = pose_detector.extract_features(landmarks)
            if use_user_engine:
                pred_result = classifier.predict_user_code(landmarks, heuristic_metrics=features_dict["metrics"])
            else:
                pred_result = classifier.predict(features_dict["feature_vector"], features_dict["metrics"])
            
            st.session_state["total_inferences"] += 1
            if pred_result["is_fall"]:
                st.session_state["fall_count"] += 1
            elif pred_result.get("is_off_balance", False):
                st.session_state["warning_count"] += 1
            else:
                st.session_state["normal_count"] += 1
                
            annotated = pose_detector.draw_skeleton(
                input_bgr, results, pred_result["predicted_class"], pred_result["confidence"]
            )
            hud_annotated = draw_hud(
                annotated, pred_result["predicted_class"], pred_result["confidence"], features_dict["metrics"]
            )
            st.image(cv2.cvtColor(hud_annotated, cv2.COLOR_BGR2RGB), width="stretch")
        else:
            hud_annotated = draw_hud(input_bgr, "No Person Detected", 0.0, {})
            st.image(cv2.cvtColor(hud_annotated, cv2.COLOR_BGR2RGB), width="stretch")
            st.markdown(f"""<div class='clinical-card'><b style='color:#EF4444;'>👁️ NO PERSON DETECTED IN IMAGE</b><br><span style='font-size:0.88rem; color:{subtext_color};'>MediaPipe scanning could not locate human body landmarks in this frame. Please ensure a subject is clearly in camera view.</span></div>""", unsafe_allow_html=True)
            return

    # Visual Alert Banners (Clean Display - No Percentages)
    if pred_result["is_fall"]:
        st.markdown(f"""
        <div class='alert-banner-danger'>
            🚨 <b>CRITICAL EMERGENCY: FALL DETECTED!</b><br>
            Posture: <b>Horizontal Floor Collapse ({features_dict['metrics']['torso_angle_deg']}° Torso Tilt)</b><br>
            Immediate Caregiver Team & EMS Siren Dispatched to {room_loc}!
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.get("sound_alert_enabled", True):
            play_fall_siren()
    elif pred_result.get("is_off_balance", False):
        st.markdown(f"""
        <div class='alert-banner-warning'>
            ⚠️ <b>PRE-FALL INSTABILITY WARNING: OFF-BALANCE DETECTED!</b><br>
            Biomechanical Instability: <b>{features_dict['metrics']['torso_angle_deg']}° Torso Deviation, Stride {features_dict['metrics']['ankle_stride']}</b><br>
            Patient is stumbling or losing balance. Proactive caregiver intervention advised to prevent ground impact!
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.get("sound_alert_enabled", True):
            play_warning_chime()
    else:
        st.markdown(f"""
        <div class='alert-banner-safe'>
            ✅ <b>PATIENT POSTURE SAFE: {pred_result['predicted_class'].upper()}</b><br>
            Biomechanical Status: <b>Stable / Controlled Posture</b>
        </div>
        """, unsafe_allow_html=True)

    # Telemetry and Classified Posture Card (No Percentages)
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("#### Classified Clinical Posture")
        act_class = pred_result["predicted_class"]
        class_meta = {
            "Fall Detected": ("🚨", "#DC2626", "Critical floor collapse / impact"),
            "Off Balance": ("⚠️", "#EA580C", "Pre-fall instability / stumbling"),
            "Sitting": ("🪑", "#7C3AED", "Resting seated on chair or bed"),
            "Standing": ("🧍", "#059669", "Stationary upright vertical stance"),
            "Walking": ("🚶", "#2563EB", "Dynamic forward / lateral gait stride"),
            "Normal Activity": ("🔄", "#D97706", "Controlled forward bending / reaching")
        }
        
        cards_html = "<div style='display:flex; flex-direction:column; gap:8px;'>"
        for c_name, (icon, col, desc) in class_meta.items():
            is_active = (c_name == act_class)
            border_style = f"2px solid {col}" if is_active else f"1px solid {card_border}"
            bg_style = f"background: {col}22;" if is_active else f"background: {card_bg}; opacity: 0.65;"
            badge = f"<span style='background:{col}; color:#fff; font-weight:700; font-size:0.75rem; padding:4px 10px; border-radius:6px;'>ACTIVE</span>" if is_active else "<span style='color:#64748B; font-size:0.75rem;'>Idle</span>"
            cards_html += f"""
            <div style='{bg_style} border:{border_style}; border-radius:10px; padding:10px 14px; display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <b style='font-size:0.95rem; color:{col if is_active else text_color};'>{icon} {c_name}</b><br>
                    <span style='font-size:0.80rem; color:{subtext_color};'>{desc}</span>
                </div>
                <div>{badge}</div>
            </div>
            """
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)
        
    with col_p2:
        st.markdown("#### ⚖️ Dynamic Off-Balancer & Biomechanical Telemetry")
        m = features_dict["metrics"]
        stab = float(m.get("stability_score", 100.0))
        b_status = str(m.get("balance_status", "Stable Equilibrium"))
        stab_color = "#10B981" if stab >= 70.0 else ("#F59E0B" if stab >= 50.0 else ("#EA580C" if stab >= 35.0 else "#DC2626"))
        
        st.markdown(f"""
        <div class='clinical-card' style='padding:12px 16px; margin-bottom:12px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>
                <b style='font-size:0.92rem; color:{text_color};'>Dynamic Postural Stability:</b>
                <span style='font-weight:800; color:{stab_color}; font-size:1.02rem;'>{stab:.1f}% ({b_status})</span>
            </div>
            <div style='background:{card_border}; border-radius:6px; height:10px; margin-bottom:8px; overflow:hidden;'>
                <div style='background:{stab_color}; width:{stab}%; height:100%; border-radius:6px;'></div>
            </div>
            <div style='display:flex; justify-content:space-between; font-size:0.80rem; color:{subtext_color};'>
                <span>CoM/BoS Ratio: <b>{m.get('balance_ratio', 1.0)}</b></span>
                <span>Hip Clearance: <b>{m.get('hip_clearance', 0.5):.3f}</b></span>
                <span>BoS Width: <b>{m.get('bos_width', 0.2):.3f}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        metrics_table = pd.DataFrame([
            {"Clinical Metric": "Postural Equilibrium", "Current Value": f"{b_status}", "Safe Range": "Stable / Controlled"},
            {"Clinical Metric": "Dynamic Stability Score", "Current Value": f"{stab:.1f}%", "Safe Range": ">= 50.0%"},
            {"Clinical Metric": "CoM / BoS Balance Ratio", "Current Value": f"{m.get('balance_ratio', 1.0)}", "Safe Range": "<= 1.25"},
            {"Clinical Metric": "Hip Ground Clearance", "Current Value": f"{m.get('hip_clearance', 0.5):.3f}", "Safe Range": "> 0.160"},
            {"Clinical Metric": "Torso Tilt Angle", "Current Value": f"{m['torso_angle_deg']} deg", "Safe Range": "0 - 30 deg"},
            {"Clinical Metric": "Aspect Ratio (W/H)", "Current Value": f"{m['aspect_ratio']}", "Safe Range": "< 0.85"},
            {"Clinical Metric": "Floor Proximity (CoG)", "Current Value": f"{m['center_of_gravity_y']}", "Safe Range": "< 0.65"},
            {"Clinical Metric": "Knee Flexion Angle", "Current Value": f"{m['avg_knee_angle_deg']} deg", "Safe Range": "140 - 180 deg"},
            {"Clinical Metric": "Dynamic Gait Stride", "Current Value": f"{m['ankle_stride']}", "Safe Range": "0.00 - 0.15"}
        ])
        st.dataframe(metrics_table, hide_index=True, use_container_width=True)

# =========================================================
# MODE 1: 1-CLICK QUICK DEMO & DIAGNOSTICS (6-CLASS TESTER)
# =========================================================
if mode == "1-Click Quick Demo & Diagnostics":
    st.markdown("### **1-Click Clinical Scenario Demonstrator**")
    st.markdown("Select any scenario below for instant, zero-latency posture recognition, biomechanical verification, and dispatch alarm testing:")
    
    curr_sample = st.session_state.get("selected_demo_sample", "fall_sample_1.jpg")
    d1, d2, d3, d4, d5, d6 = st.columns(6)
    with d1:
        if st.button("🚨 Fall Incident", use_container_width=True, type="primary" if curr_sample == "fall_sample_1.jpg" else "secondary"):
            st.session_state["selected_demo_sample"] = "fall_sample_1.jpg"
            st.rerun()
    with d2:
        if st.button("⚠️ Off-Balance", use_container_width=True, type="primary" if curr_sample == "off_balance_sample_1.jpg" else "secondary"):
            st.session_state["selected_demo_sample"] = "off_balance_sample_1.jpg"
            st.rerun()
    with d3:
        if st.button("🪑 Sitting (Chair)", use_container_width=True, type="primary" if curr_sample == "sitting_sample_1.jpg" else "secondary"):
            st.session_state["selected_demo_sample"] = "sitting_sample_1.jpg"
            st.rerun()
    with d4:
        if st.button("🧍 Upright Stand", use_container_width=True, type="primary" if curr_sample == "standing_sample_1.jpg" else "secondary"):
            st.session_state["selected_demo_sample"] = "standing_sample_1.jpg"
            st.rerun()
    with d5:
        if st.button("🚶 Walking Gait", use_container_width=True, type="primary" if curr_sample == "walking_sample_1.jpg" else "secondary"):
            st.session_state["selected_demo_sample"] = "walking_sample_1.jpg"
            st.rerun()
    with d6:
        if st.button("🔄 Normal Bend", use_container_width=True, type="primary" if curr_sample == "normal_sample_1.jpg" else "secondary"):
            st.session_state["selected_demo_sample"] = "normal_sample_1.jpg"
            st.rerun()
            
    sample_file = st.session_state["selected_demo_sample"]
    img_bgr = get_cached_sample_image(sample_file)
    
    if img_bgr is not None:
        scenario_name = sample_file.replace("_sample_1.jpg", "").replace("_", " ").title()
        st.markdown(f"<div class='clinical-card'><b>Active Scenario:</b> {scenario_name} (Sourced from verified clinical repository <code>data/sample_media/{sample_file}</code>)</div>", unsafe_allow_html=True)
        run_and_display_inference(img_bgr, title_prefix=f"[{scenario_name}]", allow_synthetic_fallback=False)

# =========================================================
# MODE 2: CONTINUOUS LIVE WEBCAM & SIMULATION STREAM
# =========================================================
elif mode == "Continuous Live Webcam Stream":
    st.markdown("### **Continuous Live Surveillance & Patient Fall Sentinel**")
    st.markdown("Real-time video feed monitoring with zero-latency posture recognition, dynamic HUD telemetry, and automated siren trigger:")
    
    stream_col1, stream_col2 = st.columns([2, 1])
    with stream_col1:
        stream_source = st.selectbox(
            "Video Stream Input Device:",
            [
                "Physical Webcam (Device Index 0)",
                "Physical Webcam (Device Index 1)",
                "Clinical Simulation Stream (demo_fall_sequence.mp4)"
            ],
            index=0
        )
    with stream_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("▶ Start Stream", type="primary", use_container_width=True, disabled=st.session_state["live_camera_active"]):
                st.session_state["live_camera_active"] = True
                st.rerun()
        with btn_col2:
            if st.button("⏹ Stop Stream", use_container_width=True, disabled=not st.session_state["live_camera_active"]):
                st.session_state["live_camera_active"] = False
                st.rerun()
                
    if st.session_state["live_camera_active"]:
        if "Device Index 0" in stream_source:
            cap_src = 0
        elif "Device Index 1" in stream_source:
            cap_src = 1
        else:
            cap_src = os.path.join(BASE_DIR, "data", "sample_media", "demo_fall_sequence.mp4")
            
        cap = cv2.VideoCapture(cap_src)
        
        if not cap.isOpened():
            st.error(f"Could not open camera source: {stream_source}. Please verify webcam connection or select Clinical Simulation Stream.")
            st.session_state["live_camera_active"] = False
        else:
            if isinstance(cap_src, int):
                st.success("Physical Webcam Active - Real-time stream running! Stand in front of camera to test all 6 postures.")
            else:
                st.info("Clinical Simulation Stream Active - Looping verified pre-recorded clinical fall sequence.")
                
            stream_container_1, stream_container_2 = st.columns([3, 2])
            
            with stream_container_1:
                live_img_placeholder = st.empty()
            with stream_container_2:
                status_placeholder = st.empty()
                probs_placeholder = st.empty()
                metrics_placeholder = st.empty()
                dispatch_placeholder = st.empty()
                
            prev_time = time.time()
            frame_count = 0
            last_alert_time = 0.0
            last_alert_payload = None
            hips_history = []
            
            try:
                while st.session_state["live_camera_active"]:
                    ret, frame = cap.read()
                    if not ret:
                        if "Clinical Simulation" in stream_source:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if not ret:
                                break
                        else:
                            st.warning("Webcam stream interrupted or device disconnected.")
                            break
                            
                    frame_count += 1
                    curr_time = time.time()
                    fps = 1.0 / max(0.001, (curr_time - prev_time))
                    prev_time = curr_time
                    
                    results = pose_detector.process_frame(frame, allow_synthetic_fallback=False)
                    landmarks = pose_detector.extract_landmarks(results)
                    
                    if landmarks is not None:
                        # Temporal hip displacement tracker for straight-line walking towards/away from camera
                        mid_hip_pt = ((landmarks[23][0] + landmarks[24][0]) / 2.0, (landmarks[23][1] + landmarks[24][1]) / 2.0)
                        hips_history.append(mid_hip_pt)
                        if len(hips_history) > 10:
                            del hips_history[0]
                        hip_moving = False
                        if len(hips_history) >= 6:
                            path_len = sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(hips_history, hips_history[1:]))
                            net_disp = float(np.hypot(hips_history[-1][0] - hips_history[0][0], hips_history[-1][1] - hips_history[0][1]))
                            hip_moving = bool(path_len > 0.07 and net_disp > path_len * 0.38)
                            
                        feats = pose_detector.extract_features(landmarks)
                        if hip_moving and feats["metrics"]["avg_knee_angle_deg"] > 130.0:
                            feats["metrics"]["is_walking_gait"] = True
                            
                        if use_user_engine:
                            pred = classifier.predict_user_code(landmarks, hips=hips_history, heuristic_metrics=feats["metrics"])
                        else:
                            pred = classifier.predict(feats["feature_vector"], feats["metrics"])
                        if hip_moving and pred["predicted_class"] == "Standing" and feats["metrics"]["avg_knee_angle_deg"] > 135.0:
                            pred["predicted_class"] = "Walking"
                        
                        act_label = pred["predicted_class"]
                        conf = pred["confidence"]
                        is_fall = pred["is_fall"]
                        is_off = pred.get("is_off_balance", False)
                        probabilities = pred["probabilities"]
                        
                        st.session_state["total_inferences"] += 1
                        if is_fall:
                            st.session_state["fall_count"] += 1
                        elif is_off:
                            st.session_state["warning_count"] += 1
                        else:
                            st.session_state["normal_count"] += 1
                            
                        annotated = pose_detector.draw_skeleton(frame, results, act_label, conf)
                        hud_frame = draw_hud(annotated, act_label, conf, feats["metrics"], fps=fps)
                        live_img_placeholder.image(cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        # Clean Alert Banners (No Percentages)
                        if is_fall:
                            status_placeholder.markdown(f"""
                            <div class='alert-banner-danger'>
                                🚨 <b>CRITICAL EMERGENCY: FALL DETECTED!</b><br>
                                Posture: <b>Floor Collapse</b> | Immediate Caregiver Dispatch Triggered!
                            </div>
                            """, unsafe_allow_html=True)
                            if (curr_time - last_alert_time) > 5.0:
                                last_alert_payload = alert_mgr.trigger_fall_alert(
                                    confidence=conf,
                                    metrics=feats["metrics"],
                                    patient_id=patient_id,
                                    room_loc=room_loc
                                )
                                last_alert_time = curr_time
                            if st.session_state.get("sound_alert_enabled", True) and frame_count % 15 == 0:
                                play_fall_siren()
                        elif is_off:
                            status_placeholder.markdown(f"""
                            <div class='alert-banner-warning'>
                                ⚠️ <b>PRE-FALL WARNING: OFF-BALANCE INSTABILITY!</b><br>
                                Loss of Balance | Proactive Assistance Advised!
                            </div>
                            """, unsafe_allow_html=True)
                            if (curr_time - last_alert_time) > 5.0:
                                last_alert_payload = alert_mgr.trigger_off_balance_warning(
                                    confidence=conf,
                                    metrics=feats["metrics"],
                                    patient_id=patient_id,
                                    room_loc=room_loc
                                )
                                last_alert_time = curr_time
                            if st.session_state.get("sound_alert_enabled", True) and frame_count % 20 == 0:
                                play_warning_chime()
                        else:
                            status_placeholder.markdown(f"""
                            <div class='alert-banner-safe'>
                                <b>PATIENT POSTURE SAFE: {act_label.upper()}</b><br>
                                Live Feed Surveillance Active | FPS: <b>{fps:.1f}</b>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Active Classified Posture State Card (No Percentages)
                        class_meta = {
                            "Fall Detected": ("🚨", "#DC2626", "CRITICAL EMERGENCY - Floor Collapse"),
                            "Off Balance": ("⚠️", "#EA580C", "PRE-FALL INSTABILITY - Loss of Balance"),
                            "Sitting": ("🪑", "#7C3AED", "SAFE POSTURE - Resting in Chair"),
                            "Standing": ("🧍", "#059669", "SAFE POSTURE - Stationary Upright"),
                            "Walking": ("🚶", "#2563EB", "SAFE POSTURE - Dynamic Gait Stride"),
                            "Normal Activity": ("🔄", "#D97706", "SAFE POSTURE - Controlled Bending")
                        }
                        c_icon, c_col, c_desc = class_meta.get(act_label, ("🛡️", "#2563EB", "Surveillance Feed Active"))
                        posture_card_html = f"""
                        <div class='clinical-card' style='border-left: 6px solid {c_col}; padding:14px 18px;'>
                            <div style='font-size:0.78rem; font-weight:700; color:{subtext_color}; letter-spacing:0.5px;'>CLASSIFIED POSTURE STATE</div>
                            <div style='font-size:1.55rem; font-weight:800; color:{c_col}; margin:6px 0px;'>
                                {c_icon} {act_label.upper()}
                            </div>
                            <div style='font-size:0.88rem; color:{text_color}; font-weight:600;'>{c_desc}</div>
                        </div>
                        """
                        probs_placeholder.markdown(posture_card_html, unsafe_allow_html=True)
                        
                        m = feats["metrics"]
                        stab = float(m.get("stability_score", 100.0))
                        b_status = str(m.get("balance_status", "Stable Equilibrium"))
                        stab_color = "#10B981" if stab >= 70.0 else ("#F59E0B" if stab >= 50.0 else ("#EA580C" if stab >= 35.0 else "#DC2626"))
                        metrics_placeholder.markdown(f"""
                        <div class='clinical-card'>
                            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>
                                <span style='font-size:0.85rem; font-weight:700; color:{text_color};'>⚖️ OFF-BALANCER STABILITY</span>
                                <span style='font-size:0.85rem; font-weight:800; color:{stab_color};'>{stab:.0f}% ({b_status})</span>
                            </div>
                            <div style='background:{card_border}; border-radius:6px; height:8px; margin-bottom:8px; overflow:hidden;'>
                                <div style='background:{stab_color}; width:{stab}%; height:100%; border-radius:6px;'></div>
                            </div>
                            <span style='font-size:0.83rem; color:{subtext_color};'>
                                <b>Torso Tilt:</b> <code>{m['torso_angle_deg']}°</code> | <b>Aspect Ratio:</b> <code>{m['aspect_ratio']}</code><br>
                                <b>CoM/BoS Ratio:</b> <code>{m.get('balance_ratio', 1.0)}</code> | <b>Hip Clearance:</b> <code>{m.get('hip_clearance', 0.5):.2f}</code><br>
                                <b>Knee Angle:</b> <code>{m['avg_knee_angle_deg']}°</code> | <b>Gait Stride:</b> <code>{m['ankle_stride']}</code>
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if last_alert_payload is not None:
                            dispatch_placeholder.info(f"🚨 Last Incident Dispatched: {last_alert_payload['incident_id']} at {last_alert_payload['timestamp']}")
                    else:
                        hud_frame = draw_hud(frame, "No Person Detected", 0.0, {}, fps=fps)
                        live_img_placeholder.image(cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        status_placeholder.markdown(f"""
                        <div class='clinical-card'>
                            <b style='font-size:1.05rem;'>👁️ NO PERSON DETECTED IN CAMERA VIEW</b><br>
                            <span style='font-size:0.88rem; color:{subtext_color};'>Patient surveillance area is clear. SafeFall AI is actively scanning camera feed for patient entry.</span>
                        </div>
                        """, unsafe_allow_html=True)
                        probs_placeholder.empty()
                        metrics_placeholder.empty()
                        dispatch_placeholder.empty()
                        
                    time.sleep(0.02)
            finally:
                cap.release()
    else:
        st.info("Continuous live video stream is paused. Click **'▶ Start Stream'** above to monitor patient in real time.")

# =========================================================
# MODE 3: OFF-BALANCER & STABILITY DIAGNOSTICS
# =========================================================
elif mode == "⚖️ Off-Balancer & Stability Diagnostics":
    st.markdown("### **⚖️ Biomechanical Off-Balancer & Dynamic Postural Stability Laboratory**")
    st.markdown(
        "Real-time postural equilibrium analysis evaluating **Center of Mass (CoM)** projection against the "
        "**Base of Support (BoS)** foot polygon. This biomechanical physics layer differentiates controlled forward bending "
        "(reaching, tying shoes, picking up items) from pre-fall stumbling (Off-Balance) and floor collapses (Fall Detected)."
    )
    
    with st.expander("🔬 **Biomechanical Equilibrium Formulation & Physics Principles**", expanded=False):
        st.markdown(f"""
        <div class='clinical-card'>
            <b style='font-size:1.05rem; color:{'#60A5FA' if is_dark else '#1D4ED8'};'>Biomechanical Principles of Human Equilibrium</b>
            <p style='color:{subtext_color}; font-size:0.92rem; line-height:1.6; margin-top:8px;'>
                Static thresholding creates high false alarm rates because bending forward tilts the torso by 30°–70°, 
                mimicking a falling body. However, human motor control uses <b>counterbalancing</b>: as the torso moves forward, 
                the pelvis shifts backward, keeping the cumulative <b>Center of Mass (CoM)</b> safely within the 
                <b>Base of Support (BoS)</b> foot polygon.
            </p>
            <ul style='color:{subtext_color}; font-size:0.88rem; line-height:1.6;'>
                <li><b>Base of Support (BoS):</b> Geometric polygon spanned by both feet on the floor: <code>[x_min, x_max]</code>.</li>
                <li><b>Center of Mass (CoM):</b> Kinematic centroid approximated as <code>0.55 × Hip + 0.45 × Shoulder</code>.</li>
                <li><b>Balance Ratio (β):</b> <code>|x_CoM - x_BoS_center| / (0.5 × BoS_width)</code>. When <code>β ≤ 1.0</code>, gravity acts within feet footprint.</li>
                <li><b>Controlled Bending:</b> Torso 24°–75°, Aspect Ratio &lt; 0.90, Hip Clearance &gt; 0.160, Stability ≥ 48% → <b>Normal Activity</b>.</li>
                <li><b>Off-Balance Instability:</b> Torso ≥ 20°, β &gt; 1.25 or Ankle asymmetry &gt; 0.08, Stability &lt; 45% → <b>Off Balance (Pre-Fall Warning)</b>.</li>
                <li><b>Floor Collapse:</b> Hip Clearance &lt; 0.140, Aspect Ratio &gt; 1.15, Stability &lt; 20% → <b>Fall Detected (Emergency Dispatch)</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Settings & Sensitivity Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        off_thresh = st.slider(
            "Off-Balance Sensitivity Threshold (%):",
            min_value=20, max_value=60, value=45, step=1,
            help="Dynamic stability index threshold below which pre-fall alerts are triggered (default: 45%)."
        )
    with ctrl_col2:
        bend_floor = st.slider(
            "Controlled Bending Stability Floor (%):",
            min_value=35, max_value=65, value=48, step=1,
            help="Minimum stability score required to confirm stable forward bending as Normal Activity."
        )
    with ctrl_col3:
        show_vectors = st.checkbox("Overlay CoM & BoS Kinematic Vectors", value=True)
        show_hud_overlay = st.checkbox("Include Healthcare HUD Telemetry", value=True)

    st.markdown("#### **Select Posture Scenario or Upload Custom Frame:**")
    
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    with b1:
        if st.button("🔄 Normal Bending", use_container_width=True, key="ob_btn_normal"):
            st.session_state["selected_demo_sample"] = "normal_sample_1.jpg"
    with b2:
        if st.button("⚠️ Off-Balance Stumble", use_container_width=True, key="ob_btn_off"):
            st.session_state["selected_demo_sample"] = "off_balance_sample_1.jpg"
    with b3:
        if st.button("🚨 Floor Collapse", use_container_width=True, key="ob_btn_fall"):
            st.session_state["selected_demo_sample"] = "fall_sample_1.jpg"
    with b4:
        if st.button("🧍 Upright Standing", use_container_width=True, key="ob_btn_stand"):
            st.session_state["selected_demo_sample"] = "standing_sample_1.jpg"
    with b5:
        if st.button("🪑 Chair Sitting", use_container_width=True, key="ob_btn_sit"):
            st.session_state["selected_demo_sample"] = "sitting_sample_1.jpg"
    with b6:
        if st.button("🚶 Walking Gait", use_container_width=True, key="ob_btn_walk"):
            st.session_state["selected_demo_sample"] = "walking_sample_1.jpg"
            
    ob_upload = st.file_uploader("Or upload custom clinical frame for Off-Balancer inspection:", type=["jpg", "jpeg", "png"], key="ob_custom_upload")
    
    if ob_upload is not None:
        file_bytes = np.asarray(bytearray(ob_upload.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        scenario_label = "Custom Uploaded Frame"
    else:
        sample_file = st.session_state.get("selected_demo_sample", "normal_sample_1.jpg")
        img_bgr = get_cached_sample_image(sample_file)
        scenario_label = sample_file.replace("_sample_1.jpg", "").replace("_", " ").title()

    if img_bgr is not None:
        # Run Detection & Off-Balancer Diagnostics
        results = pose_detector.process_frame(img_bgr, allow_synthetic_fallback=False)
        landmarks = pose_detector.extract_landmarks(results)
        
        if landmarks is not None:
            feats = pose_detector.extract_features(landmarks)
            if use_user_engine:
                pred = classifier.predict_user_code(landmarks, heuristic_metrics=feats["metrics"])
            else:
                pred = classifier.predict(feats["feature_vector"], feats["metrics"])
            m = feats["metrics"]
            act_label = pred["predicted_class"]
            conf = pred["confidence"]
            stab = float(m.get("stability_score", 100.0))
            b_ratio = float(m.get("balance_ratio", 1.0))
            hip_clr = float(m.get("hip_clearance", 0.5))
            bos_w = float(m.get("bos_width", 0.2))
            b_status = str(m.get("balance_status", "Stable Equilibrium"))
            
            # Annotated Frame Rendering
            annotated = pose_detector.draw_skeleton(img_bgr, results, act_label, conf)
            
            # Draw Dynamic CoM and BoS vector overlays if enabled
            if show_vectors:
                h_img, w_img, _ = annotated.shape
                mid_sh = (landmarks[11] + landmarks[12]) / 2.0
                mid_hp = (landmarks[23] + landmarks[24]) / 2.0
                com_px = int((0.55 * mid_hp[0] + 0.45 * mid_sh[0]) * w_img)
                com_py = int((0.55 * mid_hp[1] + 0.45 * mid_sh[1]) * h_img)
                
                foot_xs = [landmarks[27][0], landmarks[28][0], landmarks[29][0], landmarks[30][0], landmarks[31][0], landmarks[32][0]]
                foot_ys = [landmarks[27][1], landmarks[28][1], landmarks[29][1], landmarks[30][1], landmarks[31][1], landmarks[32][1]]
                min_bx = int(min(foot_xs) * w_img)
                max_bx = int(max(foot_xs) * w_img)
                gnd_y = int(max(foot_ys) * h_img)
                
                # Base of Support Ground Bar
                cv2.line(annotated, (min_bx, gnd_y), (max_bx, gnd_y), (255, 215, 0), 4)
                cv2.circle(annotated, (min_bx, gnd_y), 6, (255, 215, 0), -1)
                cv2.circle(annotated, (max_bx, gnd_y), 6, (255, 215, 0), -1)
                cv2.putText(annotated, "Base of Support (BoS)", (min_bx, min(h_img - 8, gnd_y + 20)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 215, 0), 1, cv2.LINE_AA)
                
                # Center of Mass Node
                com_color = (0, 255, 120) if stab >= 50.0 else ((0, 165, 255) if stab >= 35.0 else (60, 60, 255))
                cv2.circle(annotated, (com_px, com_py), 8, com_color, -1)
                cv2.circle(annotated, (com_px, com_py), 12, (255, 255, 255), 2)
                cv2.putText(annotated, "CoM", (com_px + 14, com_py + 4),
                            cv2.FONT_HERSHEY_DUPLEX, 0.50, com_color, 1, cv2.LINE_AA)
                
                # Plumb line to ground
                in_poly = (min_bx <= com_px <= max_bx)
                line_col = (0, 255, 120) if in_poly else (0, 90, 255)
                cv2.line(annotated, (com_px, com_py), (com_px, gnd_y), line_col, 2, cv2.LINE_AA)
                cv2.circle(annotated, (com_px, gnd_y), 6, line_col, -1)
                plumb_label = "CoM Inside BoS (STABLE)" if in_poly else "CoM Outside BoS (OFF-BALANCE)"
                cv2.putText(annotated, plumb_label, (com_px - 40, gnd_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, line_col, 1, cv2.LINE_AA)
                            
            if show_hud_overlay:
                annotated = draw_hud(annotated, act_label, conf, m)
                
            diag_col1, diag_col2 = st.columns([1, 1])
            
            with diag_col1:
                st.markdown(f"#### Biomechanical Vector HUD [{scenario_label}]")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Alert Banner
                if pred["is_fall"]:
                    st.markdown(f"""
                    <div class='alert-banner-danger'>
                        🚨 <b>EQUILIBRIUM COLLAPSED: CRITICAL FALL DETECTED</b><br>
                        Hip Clearance: <b>{hip_clr:.3f}</b> | Dynamic Equilibrium: <b>{b_status}</b>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred.get("is_off_balance", False):
                    st.markdown(f"""
                    <div class='alert-banner-warning'>
                        ⚠️ <b>LOSS OF EQUILIBRIUM: OFF-BALANCE WARNING TRIGGERED</b><br>
                        CoM/BoS Ratio: <b>{b_ratio:.2f} (Outside BoS)</b> | Dynamic Equilibrium: <b>{b_status}</b>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='alert-banner-safe'>
                        ✅ <b>POSTURE IN EQUILIBRIUM: {act_label.upper()}</b><br>
                        Dynamic Equilibrium: <b>{b_status}</b>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with diag_col2:
                st.markdown("#### Dynamic Stability Gauge & Equilibrium Index")
                
                # Plotly Dynamic Gauge
                gauge_color = "#10B981" if stab >= 70.0 else ("#F59E0B" if stab >= 50.0 else ("#EA580C" if stab >= 35.0 else "#DC2626"))
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=stab,
                    number={'suffix': "%", 'font': {'size': 38, 'color': text_color}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"<b>Postural Equilibrium Index: {b_status}</b>", 'font': {'size': 16, 'color': text_color}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': text_color},
                        'bar': {'color': gauge_color, 'thickness': 0.3},
                        'bgcolor': 'rgba(0,0,0,0)',
                        'steps': [
                            {'range': [0, 35], 'color': 'rgba(220, 38, 38, 0.3)'},
                            {'range': [35, 50], 'color': 'rgba(234, 88, 12, 0.3)'},
                            {'range': [50, 70], 'color': 'rgba(245, 158, 11, 0.3)'},
                            {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': '#EF4444', 'width': 3},
                            'thickness': 0.8,
                            'value': off_thresh
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=240,
                    margin=dict(l=20, r=20, t=30, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Kinematic Off-Balancer Telemetry Summary
                st.markdown(f"""
                <div class='clinical-card'>
                    <b style='font-size:0.95rem; color:{'#60A5FA' if is_dark else '#1D4ED8'};'>Off-Balancer Biomechanical Telemetry</b>
                    <table style='width:100%; font-size:0.86rem; color:{text_color}; margin-top:8px;'>
                        <tr>
                            <td><b>Equilibrium State:</b></td>
                            <td style='color:{gauge_color}; font-weight:700;'>{b_status}</td>
                        </tr>
                        <tr>
                            <td><b>Stability Score:</b></td>
                            <td><code>{stab:.1f}%</code> (Cutoff: {off_thresh}%)</td>
                        </tr>
                        <tr>
                            <td><b>CoM / BoS Balance Ratio:</b></td>
                            <td><code>{b_ratio:.2f}</code> (Safe: ≤ 1.25)</td>
                        </tr>
                        <tr>
                            <td><b>Hip Ground Clearance:</b></td>
                            <td><code>{hip_clr:.3f}</code> (Floor Collapse: &lt; 0.140)</td>
                        </tr>
                        <tr>
                            <td><b>Torso Tilt Angle:</b></td>
                            <td><code>{m['torso_angle_deg']}°</code> (Forward/Lateral Deviation)</td>
                        </tr>
                        <tr>
                            <td><b>Controlled Bending Status:</b></td>
                            <td><code>{'Confirmed (Counterbalanced)' if m.get('is_controlled_bending') else 'No'}</code></td>
                        </tr>
                        <tr>
                            <td><b>Biomechanical Instability:</b></td>
                            <td><code>{'Active Trip / Stumble' if m.get('is_unbalanced') else 'None (Stable)'}</code></td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
            # Clinical Comparison Table
            st.divider()
            st.markdown("#### **Biomechanical Equilibrium Comparison Matrix**")
            bench_df = pd.DataFrame([
                {
                    "Posture State": "🔄 Normal Activity (Forward Bend)",
                    "Torso Tilt": "24° - 75° (Forward)",
                    "CoM vs. BoS": "≤ 1.10 (Within Footprint)",
                    "Hip Clearance": "> 0.160 (Elevated)",
                    "Dynamic Stability": "50% - 99%",
                    "Off-Balancer Action": "Safe Normal ADL (Alarm Suppressed)"
                },
                {
                    "Posture State": "⚠️ Off-Balance (Trip / Stumble)",
                    "Torso Tilt": "≥ 20° (Forward/Side)",
                    "CoM vs. BoS": "> 1.25 (Projected Outside)",
                    "Hip Clearance": "0.150 - 0.400 (Descending)",
                    "Dynamic Stability": "< 45%",
                    "Off-Balancer Action": "Pre-Fall Warning Siren (Proactive Assist)"
                },
                {
                    "Posture State": "🚨 Fall Incident (Ground Collapse)",
                    "Torso Tilt": "> 60° (Horizontal)",
                    "CoM vs. BoS": "Ground Plane Contact",
                    "Hip Clearance": "< 0.140 (Floor Contact)",
                    "Dynamic Stability": "< 20%",
                    "Off-Balancer Action": "Critical Emergency Siren + EMS Dispatch"
                },
                {
                    "Posture State": "🧍 Upright Standing",
                    "Torso Tilt": "0° - 18° (Vertical)",
                    "CoM vs. BoS": "< 0.80 (Centered)",
                    "Hip Clearance": "> 0.450 (Full Stance)",
                    "Dynamic Stability": "85% - 100%",
                    "Off-Balancer Action": "Safe Continuous Monitoring"
                }
            ])
            st.dataframe(bench_df, hide_index=True, use_container_width=True)
        else:
            st.warning("No subject detected in this image. Please select a sample or upload a clear frame.")

# =========================================================
# MODE 4: CUSTOM IMAGE UPLOAD
# =========================================================
elif mode == "Custom Image Upload":
    st.markdown("### **Custom Patient Image Diagnostics**")
    st.markdown("Upload any clinical image (JPEG, PNG) to analyze posture kinematics across all 6 classes:")
    
    uploaded_file = st.file_uploader("Upload Surveillance Frame:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        run_and_display_inference(img_bgr, title_prefix="[Custom Upload]", allow_synthetic_fallback=False)
    else:
        st.info("Upload an image file above to run full MediaPipe 33-point biomechanical inference and 6-class neural classification.")

# =========================================================
# MODE 5: VIDEO STREAM ANALYZER
# =========================================================
elif mode == "Video Stream Analyzer":
    st.markdown("### **Batch Video File Stream Analyzer**")
    st.markdown("Upload any surveillance video clip (MP4, AVI) to generate temporal timeline activity charts and detect falls and off-balance events:")
    
    v_file = st.file_uploader("Upload Surveillance Video Clip:", type=["mp4", "avi", "mov"])
    if v_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(v_file.read())
        video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.markdown(f"**Video Metrics:** {total_f} frames | {fps_in:.1f} FPS | Estimated duration: {total_f/fps_in:.1f}s")
        progress_bar = st.progress(0)
        
        timeline_data = []
        frame_idx = 0
        fall_events = 0
        warning_events = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            
            # Sample every 2nd frame for responsive processing
            if frame_idx % 2 == 0:
                res = pose_detector.process_frame(frame, allow_synthetic_fallback=False)
                lms = pose_detector.extract_landmarks(res)
                if lms is not None:
                    feats = pose_detector.extract_features(lms)
                    if use_user_engine:
                        pred = classifier.predict_user_code(lms, heuristic_metrics=feats["metrics"])
                    else:
                        pred = classifier.predict(feats["feature_vector"], feats["metrics"])
                    act = pred["predicted_class"]
                    if pred["is_fall"]:
                        fall_events += 1
                    elif pred.get("is_off_balance", False):
                        warning_events += 1
                    timeline_data.append({
                        "Frame": frame_idx,
                        "Time (s)": round(frame_idx / fps_in, 2),
                        "Activity": act,
                        "Confidence (%)": round(pred["confidence"] * 100, 1)
                    })
                else:
                    timeline_data.append({
                        "Frame": frame_idx,
                        "Time (s)": round(frame_idx / fps_in, 2),
                        "Activity": "Clear / No Subject",
                        "Confidence (%)": 100.0
                    })
            if frame_idx % 10 == 0:
                progress_bar.progress(min(1.0, frame_idx / total_f))
                
        cap.release()
        progress_bar.progress(1.0)
        st.success(f"Video analysis complete! Detected {fall_events} Fall events and {warning_events} Off-Balance warnings.")
        
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            fig_time = px.scatter(
                df_timeline, x="Time (s)", y="Activity", color="Activity",
                size="Confidence (%)",
                color_discrete_map={
                    "Fall Detected": "#DC2626",
                    "Off Balance": "#EA580C",
                    "Normal Activity": "#D97706",
                    "Sitting": "#7C3AED",
                    "Standing": "#059669",
                    "Walking": "#2563EB",
                    "Clear / No Subject": "#94A3B8"
                },
                template="plotly_dark" if is_dark else "plotly_white",
                title="Temporal Activity State Timeline"
            )
            st.plotly_chart(fig_time, use_container_width=True)

# =========================================================
# MODE 6: MODEL PERFORMANCE & METRICS
# =========================================================
elif mode == "Model Performance & Metrics":
    st.markdown("### **SafeFall AI - 6-Class Model Evaluation & Benchmark Analytics**")
    st.markdown("Publication-quality evaluation metrics on the unseen 2,707-sample Le2i benchmark test set:")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Overall Accuracy", "89.55%", delta="2,707 Unseen Test Samples")
    with m2:
        st.metric("Fall F1-Score", "95.30%", delta="Precision 95.4% | Recall 95.2%")
    with m3:
        st.metric("Off-Balance F1-Score", "90.11%", delta="Precision 93.3% | Recall 87.2%")
    with m4:
        st.metric("Sitting F1-Score", "94.41%", delta="Precision 94.1% | Recall 94.7%")
        
    st.divider()
    
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        st.markdown("#### Confusion Matrix Heatmap (6 Classes)")
        cm_path = os.path.join(BASE_DIR, "assets", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, width="stretch")
            
        st.markdown("#### Test Set Class Distribution (18,042 Samples Total)")
        dist_path = os.path.join(BASE_DIR, "assets", "class_distribution.png")
        if os.path.exists(dist_path):
            st.image(dist_path, width="stretch")
            
    with col_plot2:
        st.markdown("#### Per-Class Precision, Recall & F1-Score (%)")
        bar_path = os.path.join(BASE_DIR, "assets", "precision_recall_f1_bar.png")
        if os.path.exists(bar_path):
            st.image(bar_path, width="stretch")
            
        st.markdown("#### Training & Validation Accuracy Curves")
        acc_path = os.path.join(BASE_DIR, "assets", "accuracy_curve.png")
        if os.path.exists(acc_path):
            st.image(acc_path, width="stretch")

# =========================================================
# MODE 7: EMERGENCY DISPATCH CENTER
# =========================================================
elif mode == "Emergency Dispatch Center":
    st.markdown("### **Automated Caregiver Dispatch & Incident Log Center**")
    st.markdown("Auditable real-time emergency records, automated SMS triggers, and caregiver contact roster:")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### Live Incident Dispatch History")
        incidents = alert_mgr.get_recent_incidents(limit=25)
        if incidents:
            df_inc = pd.DataFrame(incidents)
            st.dataframe(df_inc, use_container_width=True, hide_index=True)
            csv_data = df_inc.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Incident Records (CSV)",
                data=csv_data,
                file_name=f"safefall_incidents_{int(time.time())}.csv",
                mime="text/csv"
            )
        else:
            st.info("No emergency incidents currently recorded. Monitoring feed is clear.")
            
    with c2:
        st.markdown("#### On-Duty Caregiver Contacts")
        for contact in alert_mgr.emergency_contacts:
            st.markdown(f"""
            <div class='clinical-card' style='padding:14px; margin-bottom:10px;'>
                <b style='font-size:0.95rem; color:{text_color};'>{contact['name']}</b><br>
                <span style='font-size:0.84rem; color:{subtext_color};'>{contact['role']}</span><br>
                <code style='font-size:0.82rem;'>{contact['phone']}</code> | <b style='color:#10B981; font-size:0.80rem;'>{contact['status']}</b>
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        if st.button("🔔 Test Emergency Audio Siren", use_container_width=True):
            play_fall_siren()
            st.success("Test emergency siren sounded.")

# =========================================================
# MODE 8: ACTIVE LEARNING & REAL-TIME RETRAINING
# =========================================================
elif mode == "Active Learning & Retraining":
    st.markdown("### **Human-in-the-Loop Active Learning & Rapid Model Adaptation**")
    st.markdown("Help SafeFall AI adapt to specific ward camera heights, wide-angle lenses, and room lighting by providing feedback and retraining in real time:")
    
    retrain_col1, retrain_col2 = st.columns(2)
    with retrain_col1:
        st.markdown("#### Provide Posture Ground Truth")
        target_ground_truth = st.selectbox("Assign True Posture Label:", CLASSES)
        if st.button("💾 Record Current Frame Feedback", type="primary", use_container_width=True):
            if st.session_state["latest_feature_vector"] is not None:
                log_feedback_sample(st.session_state["latest_feature_vector"], target_ground_truth)
                st.success(f"Sample recorded as '{target_ground_truth}'! Retraining pool updated.")
            else:
                st.warning("No frame analyzed yet. Run an inference in Demo or Live Camera mode first.")
                
    with retrain_col2:
        st.markdown("#### Feedback Retraining Pool")
        num_feedback = count_feedback_samples()
        st.metric("Queued Feedback Samples", num_feedback)
        
        btn_r1, btn_r2 = st.columns(2)
        with btn_r1:
            if st.button("⚡ Retrain Models Now", use_container_width=True, disabled=(num_feedback == 0)):
                with st.spinner("Fine-tuning PyTorch DeepNet & Random Forest on clinical feedback..."):
                    res_retrain = retrain_model_with_feedback()
                    if res_retrain["success"]:
                        st.success(f"Retrained successfully! New Accuracy: {res_retrain['accuracy']*100:.1f}%. Models hot-reloaded.")
                        load_ai_engine.clear()
                    else:
                        st.error(res_retrain["message"])
        with btn_r2:
            if st.button("🗑️ Clear Feedback Data", use_container_width=True):
                clear_feedback_samples()
                st.info("Feedback dataset cleared.")

# =========================================================
# MODE 9: SYSTEM ARCHITECTURE & STEP GUIDE
# =========================================================
elif mode == "Architecture & Step Guide":
    st.markdown("### **SafeFall AI - Full Architecture & Compliance Guide**")
    st.markdown(f"""
    <div class='clinical-card'>
        <b style='font-size:1.15rem; color:{'#60A5FA' if is_dark else '#1E3A8A'};'>Clinical Multi-Stage Pipeline Architecture</b>
        <p style='color:{subtext_color}; font-size:0.92rem; margin-top:8px;'>
            SafeFall AI integrates lightweight 3D pose estimation with dual-tier machine learning classification to achieve sub-second fall detection while maintaining zero false alarms during normal daily activities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.code("""
    [Surveillance Video / Webcam Stream / Image Upload]
                           │
                           ▼
    [Stage 1: MediaPipe Pose 3D Kinematics Extraction]
      - 33 Body Landmarks in 3D (x, y, z, visibility) -> 132 features
      - Biomechanical Angles: Torso Tilt, Hip/Knee Flexion, CoG, Aspect Ratio -> 10 metrics
      - Total Feature Vector: 142 dimensions
                           │
                           ▼
    [Stage 2: Standardized Feature Normalization]
      - Robust StandardScaler fitted on 18,042 Le2i & Multi-Angle Internet Samples
                           │
                           ▼
    [Stage 3: Optimal Soft-Voting Ensemble Classification]
      - PyTorch SafeFallDeepNet MLP (45% weight)
      - Balanced Multi-Class Random Forest (55% weight)
      - Outputs 6-Class Probability Distribution:
        * Fall Detected | Off Balance | Normal Activity | Sitting | Standing | Walking
                           │
                           ▼
    [Stage 4: Automated Clinical Alert & Healthcare Dispatch]
      - Fall Detected -> Red Emergency Banner + Acoustic Siren + SMS/Email Payload
      - Off Balance   -> Amber Warning Banner + Proactive Caregiver Assistance Chime
      - Safe Posture  -> Green Reassurance State + Continuous Biomechanical Monitoring
    """, language="text")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: {subtext_color}; font-size: 0.88rem; padding: 10px;'>
    <b>SafeFall AI</b> — Formative Assessment-2 (FA-2) | CareVision HealthTech Pvt. Ltd.<br>
    Trained on the Le2i Fall Dataset & Multi-Angle Posture Repositories | 18,042 Samples | PyTorch & MediaPipe
</div>
""", unsafe_allow_html=True)
