"""
ui/auth_view.py
===============
Clean, professional, and accessible authentication portal for SafeFall AI.
Follows the modern healthcare product aesthetic with warm neutral backgrounds,
soft sage accents, salted PBKDF2 password security, quick demo auto-fill, and guest onboarding.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from core.security import SecureCredentialVault
from ui.styles import PALETTES, inject_theme_styles


def switch_auth_mode(mode: str) -> None:
    """Toggle between signin and signup views."""
    st.session_state["auth_mode"] = mode


def authenticate_as_guest() -> None:
    """Establish unauthenticated guest session."""
    st.session_state["_signed_out"] = False
    st.session_state["active_user"] = {
        "name": "Clinical Evaluator (Guest)",
        "email": "guest@safefall.local",
        "guest": True
    }
    st.rerun()


def login_as_demo() -> None:
    """Instant 1-click authenticated demo session."""
    st.session_state["_signed_out"] = False
    st.session_state["active_user"] = {
        "name": "Clinical Evaluator (Demo)",
        "email": "demo@safefall.ai",
        "guest": False
    }
    st.rerun()


def render_authentication_gate(vault: SecureCredentialVault) -> None:
    """Render clean, centered healthcare authentication portal."""
    inject_theme_styles(accent_hex="#5E8B7A", accent2_hex="#8EA8C3", theme_mode="light", include_sidebar_suppression=True)

    current_mode = st.session_state.get("auth_mode", "signin")

    # Centered portal container
    _, center_col, _ = st.columns([1, 1.4, 1])

    with center_col:
        st.markdown(
            '<div style="text-align:center; margin:2.5rem 0 1.5rem">'
            '<div style="display:inline-flex; align-items:center; gap:8px; padding:6px 14px; border-radius:9999px; background:#F2F3EC; border:1px solid #E5E7DF; font-size:0.78rem; font-weight:600; color:#5E8B7A; margin-bottom:12px">'
            '<span class="status-dot"></span> FA-2 Machine Learning & Deep Learning Project'
            '</div>'
            '<h1 style="font-size:2.2rem; font-weight:800; letter-spacing:-0.03em; color:#1F2933; margin-bottom:6px">SafeFall AI</h1>'
            '<p style="font-size:0.95rem; color:#4B5563; font-weight:500">AI-Powered Elderly Safety & Human Fall Detection Sentinel</p>'
            '</div>',
            unsafe_allow_html=True
        )

        if current_mode == "signup":
            with st.form("signup_portal", clear_on_submit=False):
                st.markdown("### Create Clinical Account")
                st.caption("Register an operator profile for elderly surveillance telemetry.")
                name_input = st.text_input("Full Name", placeholder="Dr. Alex Morgan")
                email_input = st.text_input("Email Address", placeholder="operator@hospital.org")
                pw_input = st.text_input("Password", type="password", placeholder="Minimum 8 characters")
                pw2_input = st.text_input("Confirm Password", type="password")
                submit_action = st.form_submit_button("Register Account", use_container_width=True)

            if submit_action:
                success, message, profile = vault.register_account(
                    name_input,
                    email_input,
                    pw_input,
                    pw2_input
                )
                if success:
                    st.session_state["_signed_out"] = False
                    st.session_state["active_user"] = profile
                    st.rerun()
                else:
                    st.error(message)

            st.write("")
            st.button(
                "Already have an account? Sign In",
                key="btn_to_signin",
                on_click=switch_auth_mode,
                args=("signin",),
                use_container_width=True
            )

        else:
            default_email = st.session_state.pop("demo_email", "")
            default_pass = st.session_state.pop("demo_pass", "")

            with st.form("signin_portal", clear_on_submit=False):
                st.markdown("### Operator Sign In")
                st.caption("Access real-time vision telemetry and incident records.")
                email_input = st.text_input("Email Address", value=default_email, placeholder="demo@safefall.ai")
                pw_input = st.text_input("Password", value=default_pass, type="password")
                submit_action = st.form_submit_button("Sign In to SafeFall AI", use_container_width=True)

            if submit_action:
                success, message, profile = vault.authenticate_credentials(
                    email_input,
                    pw_input
                )
                if success:
                    st.session_state["_signed_out"] = False
                    st.session_state["active_user"] = profile
                    st.rerun()
                else:
                    st.error(message)

            st.write("")
            col_demo, col_guest = st.columns(2)
            with col_demo:
                st.button(
                    "⚡ 1-Click Demo Sign-In",
                    key="btn_autofill_demo",
                    use_container_width=True,
                    on_click=login_as_demo
                )
            with col_guest:
                st.button(
                    "👤 Continue as Guest",
                    key="btn_guest_access",
                    use_container_width=True,
                    on_click=authenticate_as_guest
                )

            st.write("")
            st.button(
                "➕ Register New Clinical Account",
                key="btn_to_signup",
                on_click=switch_auth_mode,
                args=("signup",),
                use_container_width=True
            )

        st.markdown(
            '<div style="text-align:center; color:#7C8894; font-size:0.78rem; margin-top:20px">'
            'Secured with salted PBKDF2-HMAC-SHA256 &bull; Local offline evaluation session'
            '</div>',
            unsafe_allow_html=True
        )
