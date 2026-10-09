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
        "nav_page": "Overview",
        "setting_fall_thr": 0.60,
        "setting_need": 2,
        "setting_alpha": 0.65,
        "setting_stride": 1,
        "setting_imgsz": 256,
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
            "name": "Clinical Evaluator",
            "email": "demo@safefall.ai",
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
        initials = "".join(part[0] for part in current_user["name"].split()[:2]).upper() or "U"
        account_type = "Guest Evaluator" if current_user.get("guest") else "Clinical Staff"

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

        # Primary Navigation
        nav_options = [
            "📊 Overview",
            "📹 Live Monitor",
            "🔬 Media Analysis",
            "🚨 Emergency SOS",
            "📈 Model Insights",
            "🗃️ Dataset",
            "🕒 History",
            "⚙️ Settings"
        ]

        # Determine current selection index
        current_page = st.session_state.get("nav_page", "Overview")
        current_index = 0
        for i, opt in enumerate(nav_options):
            if current_page in opt:
                current_index = i
                break

        selected_nav = st.radio(
            "Navigation Menu",
            nav_options,
            index=current_index,
            label_visibility="collapsed",
            key="sidebar_navigation_radio"
        )

        # Clean selected page name without emoji
        page_name_clean = selected_nav.split(" ", 1)[-1]
        if page_name_clean != st.session_state.get("nav_page"):
            st.session_state["nav_page"] = page_name_clean
            st.rerun()

        st.write("")

        # System Health & Hardware Info Chips
        st.markdown(
            f'<div style="margin-top:20px; padding-top:14px; border-top:1px solid var(--border-subtle)">'
            f'<div style="font-size:0.72rem; font-weight:700; color:var(--text-tertiary); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px">System Telemetry</div>'
            f'<div style="display:flex; flex-direction:column; gap:6px">'
            f'<span class="badge" style="width:100%; justify-content:flex-start"><span class="status-dot"></span>Device: {str(coordinator.device).upper()}</span>'
            f'<span class="badge" style="width:100%; justify-content:flex-start">{"BiLSTM Neural Model" if coordinator.is_trained_4class else "Rule Kinematics Engine"}</span>'
            f'<span class="badge" style="width:100%; justify-content:flex-start">YOLOv8 Pose &bull; 30 FPS</span>'
            f'</div>'
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
        "need": st.session_state.get("setting_need", 4),
        "alpha": st.session_state.get("setting_alpha", 0.35),
        "stride": st.session_state.get("setting_stride", 1),
        "imgsz": st.session_state.get("setting_imgsz", 480),
        "desk_mode": st.session_state.get("setting_desk_mode", True),
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
