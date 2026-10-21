"""
app.py
======
SafeFall AI - Intelligent Real-Time Activity and Human Fall Detection System.
Healthcare Computer Vision Product for Elderly Safety Sentinel Monitoring.
Combines YOLOv8 Pose Landmark Estimation, Biomechanical Kinematics, and Deep BiLSTM Networks.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="SafeFall AI - Elderly Safety Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from core.security import SecureCredentialVault
from core.vision_pipeline import SafeFallPipelineCoordinator
from ui.auth_view import render_authentication_gate
from ui.dashboard_view import render_dashboard
from ui.styles import PALETTES, inject_theme_styles

ROOT_DIR = Path(__file__).resolve().parent
USERS_FILE = ROOT_DIR / "outputs" / "users.json"
FALLS_DIR = ROOT_DIR / "outputs" / "falls"

# Initialize credential vault
CREDENTIAL_VAULT = SecureCredentialVault(USERS_FILE)


@st.cache_resource(show_spinner="Initializing YOLOv8 pose detector and activity models...")
def initialize_pipeline_coordinator() -> SafeFallPipelineCoordinator:
    """Instantiate and cache the central vision and kinematics coordinator."""
    return SafeFallPipelineCoordinator(ROOT_DIR)


def initialize_session_defaults() -> None:
    """Ensure all clinical settings and navigation states exist in session state."""
    defaults = {
        "nav_page": "Live Monitor",
        "setting_fall_thr": 0.60,
        "setting_need": 2,
        "setting_alpha": 0.65,
        "setting_stride": 2,
        "setting_imgsz": 192,
        "setting_desk_mode": False,

        "setting_max_frames": 900,
        "setting_alarm_enabled": True,
        "setting_alarm_volume": 0.8,
        "setting_enhance": False,
        "setting_gamma": 1.6,
        "setting_fill_light": False,
        "setting_fill_tone": "Warm",
        "setting_fill_intensity": 55,
        "setting_force_legacy": False,
        "selected_palette": "Healthcare Sage",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if "active_user" not in st.session_state and not st.session_state.get("_signed_out", False):
        st.session_state["active_user"] = {
            "name": "Aditya Sahani",
            "email": "1000414@safefall.ai",
            "id": "1000414",
            "guest": False
        }


def main() -> None:
    initialize_session_defaults()

    # -------------------------------------------------------------
    # STEP 1: AUTHENTICATION GATEWAY
    # -------------------------------------------------------------
    if "active_user" not in st.session_state:
        render_authentication_gate(CREDENTIAL_VAULT)
        st.stop()

    # -------------------------------------------------------------
    # STEP 2: LOAD INFERENCE ENGINE
    # -------------------------------------------------------------
    try:
        coordinator = initialize_pipeline_coordinator()
    except Exception as err:
        st.error(f"Error loading SafeFall core pipeline: {err}")
        st.info("Ensure 'yolov8n-pose.pt' is present in the workspace directory.")
        st.stop()

    current_user = st.session_state["active_user"]

    # -------------------------------------------------------------
    # STEP 3: CLEAN, SPACIOUS HEALTHCARE SIDEBAR NAVIGATION
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">'
            '<h2>SafeFall AI</h2>'
            '<p>Elderly Safety &amp; Fall Detection Sentinel</p>'
            '</div>',
            unsafe_allow_html=True
        )

        # Operator Profile Card
        initials = "".join(part[0] for part in current_user["name"].split()[:2]).upper() or "AS"
        account_type = f"Lead Investigator &bull; ID: {current_user.get('id', '1000414')}"

        st.markdown(
            f'<div class="sidebar-profile">'
            f'<div class="sidebar-avatar">{initials}</div>'
            f'<div class="sidebar-profile-info">'
            f'<div class="nm">{current_user["name"]}</div>'
            f'<div class="st"><span class="status-dot"></span>{account_type}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        current_page = st.session_state.get("nav_page", "Live Monitor")

        # Interactive Sidebar Navigation Bar
        sidebar_pages = [
            "📹 Live Camera",
            "📷 Photo Analysis",
            "🎥 Video Analysis",
            "📸 Camera Snapshot",
            "🚨 Emergency SOS",
            "📊 System Overview",
            "📈 Model Insights",
            "📁 Dataset",
            "📜 Incident History",
            "⚙️ Settings"
        ]
        sidebar_mapping = {
            "📹 Live Camera": "Live Monitor",
            "📷 Photo Analysis": "Photo",
            "🎥 Video Analysis": "Video",
            "📸 Camera Snapshot": "Snapshot",
            "🚨 Emergency SOS": "Emergency SOS",
            "📊 System Overview": "Overview",
            "📈 Model Insights": "Model Insights",
            "📁 Dataset": "Dataset",
            "📜 Incident History": "History",
            "⚙️ Settings": "Settings"
        }
        reverse_sidebar_mapping = {v: k for k, v in sidebar_mapping.items()}
        current_sidebar_label = reverse_sidebar_mapping.get(current_page, "📹 Live Camera")
        sidebar_idx = sidebar_pages.index(current_sidebar_label) if current_sidebar_label in sidebar_pages else 0

        st.markdown(
            '<div style="font-size:0.72rem; font-weight:700; color:var(--text-tertiary); text-transform:uppercase; letter-spacing:0.06em; margin:14px 0 6px 0">Navigation Bar</div>',
            unsafe_allow_html=True
        )
        chosen_sidebar_label = st.radio(
            "Sidebar Navigation",
            sidebar_pages,
            index=sidebar_idx,
            label_visibility="collapsed",
            key="sidebar_navigation_radio"
        )
        if sidebar_mapping.get(chosen_sidebar_label) != current_page:
            st.session_state["nav_page"] = sidebar_mapping[chosen_sidebar_label]
            st.rerun()

        # System Health & Hardware Info Chips
        st.markdown(
            f'<div style="margin-top:16px; padding-top:14px; border-top:1px solid var(--border-subtle)">'
            f'<div style="font-size:0.72rem; font-weight:700; color:var(--text-tertiary); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px">System Telemetry</div>'
            f'<div style="display:flex; flex-direction:column; gap:6px">'
            f'<span class="badge" style="width:100%; justify-content:flex-start"><span class="status-dot"></span>Device: {str(coordinator.device).upper()}</span>'
            f'<span class="badge" style="width:100%; justify-content:flex-start">{"DeepNet + RF Ensemble (FA-2)" if getattr(coordinator, "is_trained_ensemble", False) else ("BiLSTM Neural Model" if coordinator.is_trained_4class else "Rule Kinematics Engine")}</span>'
            f'<span class="badge" style="width:100%; justify-content:flex-start">YOLOv8 Pose &bull; 30 FPS</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Quick Documentation & Presentation Links
        st.markdown(
            f'<div style="margin-top:16px; padding-top:12px; border-top:1px solid var(--border-subtle); display:flex; flex-direction:column; gap:8px">'
            f'<a href="https://iadai-201-1000414-aditya-jitendra-kumar-sahani-fa-2.streamlit.app/" target="_blank" style="text-decoration:none; font-size:0.80rem; font-weight:600; color:var(--accent); display:flex; align-items:center; gap:6px">🚀 Live Streamlit App ↗</a>'
            f'<a href="https://canva.link/7iadcf1w4rsmwx3" target="_blank" style="text-decoration:none; font-size:0.80rem; font-weight:600; color:var(--accent); display:flex; align-items:center; gap:6px">🎨 Canva Storyboard ↗</a>'
            f'<a href="https://colab.research.google.com/github/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA/blob/main/SafeFall_AI_Elderly_Fall_Detection_FA2.ipynb" target="_blank" style="text-decoration:none; font-size:0.80rem; font-weight:600; color:var(--accent); display:flex; align-items:center; gap:6px">📓 Google Colab Notebook ↗</a>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.write("")
        if st.button("Sign Out", key="btn_logout", use_container_width=True):
            st.session_state["_signed_out"] = True
            st.session_state.pop("active_user", None)
            st.session_state["auth_mode"] = "signin"
            st.rerun()

    # -------------------------------------------------------------
    # STEP 4: THEME INJECTION & AMBIENT ILLUMINATION
    # -------------------------------------------------------------
    primary_accent = "#5E8B7A"  # Soft sage
    secondary_accent = "#8EA8C3"  # Muted lavender
    inject_theme_styles(primary_accent, secondary_accent, theme_mode="light")

    if st.session_state.get("setting_fill_light", False):
        tone_colors = {
            "Warm": "255,225,180",
            "Cool": "200,230,255",
            "White": "255,255,255"
        }
        rgb_tone = tone_colors.get(st.session_state.get("setting_fill_tone", "Warm"), "255,225,180")
        alpha_level = st.session_state.get("setting_fill_intensity", 55) / 100.0
        st.markdown(
            f'<div class="fill-light" style="'
            f'position: fixed; inset: 0; pointer-events: none; z-index: 999999;'
            f'border: 24px solid rgba({rgb_tone}, {alpha_level:.2f});'
            f'box-shadow: inset 0 0 100px rgba({rgb_tone}, {alpha_level * 0.70:.2f});'
            f'"></div>',
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------
    # STEP 5: RENDER MAIN DASHBOARD
    # -------------------------------------------------------------
    runtime_options: Dict[str, Any] = {
        "fall_thr": st.session_state.get("setting_fall_thr", 0.60),
        "need": st.session_state.get("setting_need", 2),
        "alpha": st.session_state.get("setting_alpha", 0.65),
        "stride": st.session_state.get("setting_stride", 2),
        "imgsz": st.session_state.get("setting_imgsz", 192),
        "desk_mode": st.session_state.get("setting_desk_mode", False),
        "max_frames": st.session_state.get("setting_max_frames", 900),
        "enhance": st.session_state.get("setting_enhance", False),
        "gamma": st.session_state.get("setting_gamma", 1.6),
        "alarm_enabled": st.session_state.get("setting_alarm_enabled", True),
        "alarm_volume": st.session_state.get("setting_alarm_volume", 0.8),
        "force_legacy": st.session_state.get("setting_force_legacy", False),
        "use_model": coordinator.uses_neural_model(st.session_state.get("setting_force_legacy", False)),
        "theme_mode": "light"
    }

    render_dashboard(coordinator, FALLS_DIR, runtime_options)


if __name__ == "__main__":
    main()
