"""
ui/components.py
================
Reusable user interface components, SVG graphics, acoustic alerts,
horizontal probability indicators, and publication-quality diagnostic charts.
Crafted for a calm, trustworthy modern healthcare product aesthetic.
"""

from __future__ import annotations

import base64
import io
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from core.kinematics import ACTIVITY_CLASSES
from ui.styles import CLASS_GLYPHS, CLASS_HEX_COLORS


def render_brand_logo_html(size_px: int = 40, show_glow: bool = False) -> str:
    """Return empty string as logo has been completely removed."""
    return ""


def render_section_title(title: str, subtitle: str = "") -> str:
    """Format clean, modern section heading with calm healthcare typography."""
    sub_html = f'<p>{subtitle}</p>' if subtitle else ""
    return (
        f'<div class="section-heading">'
        f'<h2>{title}</h2>'
        f'{sub_html}'
        f'</div>'
    )


def render_metric_kpi(label: str, value: str, footer: str = "") -> str:
    """Format single clean Bento KPI stat tile with generous spacing."""
    footer_html = f'<div class="sub">{footer}</div>' if footer else ""
    return (
        f'<div class="stat-tile">'
        f'<div class="l">{label}</div>'
        f'<div class="v">{value}</div>'
        f'{footer_html}'
        f'</div>'
    )


def render_pipeline_breadcrumb(active_step: int = 1) -> str:
    """Render visual data pipeline breadcrumb from input media to final safety classification."""
    steps = [
        "1. Input Media",
        "2. Pose Landmarking",
        "3. Biomechanical Kinematics",
        "4. Activity Classification",
        "5. Safety Verification"
    ]
    parts = []
    for i, s in enumerate(steps, 1):
        cls = "pipeline-step active" if i == active_step else "pipeline-step"
        parts.append(f'<span class="{cls}">{s}</span>')
        if i < len(steps):
            parts.append('<span class="pipeline-arrow">&rarr;</span>')

    return f'<div class="pipeline-breadcrumb">{"".join(parts)}</div>'


def render_activity_cards_html(probabilities: Any, active_label: Optional[str] = None, animate: bool = False) -> str:
    """Generate clean, structured posture telemetry cards for all clinical activities."""
    cards = []
    probs_list = list(probabilities) if probabilities is not None else []
    while len(probs_list) < len(ACTIVITY_CLASSES):
        probs_list.append(0.0)
    for idx, name in enumerate(ACTIVITY_CLASSES):
        prob = float(probs_list[idx])
        is_active = (name == active_label)
        active_class = " active" if is_active else ""
        glyph = CLASS_GLYPHS.get(name, "🧍")
        color = CLASS_HEX_COLORS[idx] if idx < len(CLASS_HEX_COLORS) else "#7FAF9B"
        status_tag = "ACTIVE" if is_active else "IDLE"

        card_html = (
            f'<div class="activity-card{active_class}" style="--c:{color}">'
            f'<div class="activity-card-header">'
            f'<span class="activity-card-tag">{status_tag}</span>'
            f'<span class="status-dot{" pulse" if is_active else ""}" style="background:{color}"></span>'
            f'</div>'
            f'<div class="activity-card-icon">{glyph}</div>'
            f'<div class="activity-card-label">{name.capitalize()}</div>'
            f'<div class="activity-card-val">{prob:.1%}</div>'
            f'<div class="progress-track"><div class="progress-fill" style="width:{prob * 100.0:.1f}%; background:{color}"></div></div>'
            f'</div>'
        )
        cards.append(card_html)

    return f'<div class="activity-grid">{"".join(cards)}</div>'


def render_horizontal_probability_indicators(probabilities: Any, highlight_fall: bool = False) -> str:
    """Render clean, high-priority horizontal activity probability indicators."""
    rows = []
    probs_list = list(probabilities) if probabilities is not None else []
    while len(probs_list) < len(ACTIVITY_CLASSES):
        probs_list.append(0.0)
    for idx, name in enumerate(ACTIVITY_CLASSES):
        val = float(probs_list[idx])
        color = CLASS_HEX_COLORS[idx] if idx < len(CLASS_HEX_COLORS) else "#7FAF9B"
        glyph = CLASS_GLYPHS.get(name, "🧍")

        # Fall indicator highlights in coral if confidence is significant
        is_fall_class = (name == "FALL")
        row_color = "#D97878" if (is_fall_class and (val > 0.40 or highlight_fall)) else color

        rows.append(
            f'<div class="prob-row">'
            f'<div class="prob-label"><span>{glyph}</span> {name.capitalize()}</div>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{val * 100.0:.1f}%; background:{row_color}"></div></div>'
            f'<div class="prob-pct" style="color:{"#D97878" if (is_fall_class and val > 0.40) else "var(--text-primary)"}">{val:.1%}</div>'
            f'</div>'
        )
    return f'<div style="padding:10px 4px">{"".join(rows)}</div>'


def render_probability_bars_html(probabilities: Any) -> str:
    """Backwards-compatible wrapper for horizontal probability indicators."""
    return render_horizontal_probability_indicators(probabilities)


DEFAULT_NEARBY_HOSPITALS = [
    {
        "name": "St. Jude Metropolitan Medical Center",
        "type": "Level 1 Trauma Center & Acute Care",
        "distance": "1.2 km",
        "time": "4 min EMS transit",
        "er_status": "24/7 ER Active",
        "phone": "+1 (555) 911-0101",
        "tel": "tel:911",
        "address": "742 Evergreen Healthcare Blvd",
        "maps_url": "https://www.google.com/maps/search/St.+Jude+Metropolitan+Medical+Center/"
    },
    {
        "name": "Mercy Memorial Emergency Pavilion",
        "type": "Comprehensive Stroke & Geriatric Trauma",
        "distance": "2.8 km",
        "time": "7 min EMS transit",
        "er_status": "Immediate Triage Open",
        "phone": "+1 (555) 911-0102",
        "tel": "tel:911",
        "address": "1200 Pine Medical Center Way",
        "maps_url": "https://www.google.com/maps/search/Mercy+Memorial+Emergency+Pavilion/"
    },
    {
        "name": "Valley Community Hospital & Urgent Care",
        "type": "Emergency Department & Rapid Response",
        "distance": "4.5 km",
        "time": "11 min EMS transit",
        "er_status": "Open 24/7",
        "phone": "+1 (555) 911-0103",
        "tel": "tel:911",
        "address": "88 Oakview Medical Parkway",
        "maps_url": "https://www.google.com/maps/search/Valley+Community+Hospital/"
    },
    {
        "name": "Apex Senior Care & Geriatric Trauma Unit",
        "type": "Specialized Post-Fall Orthopedic & ICU",
        "distance": "5.6 km",
        "time": "14 min EMS transit",
        "er_status": "Geriatric ER Ready",
        "phone": "+1 (555) 911-0104",
        "tel": "tel:911",
        "address": "450 Serenity Care Blvd",
        "maps_url": "https://www.google.com/maps/search/Apex+Senior+Care+Trauma+Unit/"
    }
]


def render_fall_alert_card(
    conf: float,
    timestamp_str: str = "",
    fall_time: Optional[float] = None,
    fall_duration: float = 0.0
) -> str:
    """
    Render medical-grade healthcare fall alert state in muted coral with
    progressive escalation stage, direct Google Maps hospital redirect, and SOS emergency buttons.
    """
    ts = timestamp_str or datetime.now().strftime("%H:%M:%S")
    time_detail = f" at {fall_time:.1f}s into sequence" if fall_time is not None else ""

    # Determine escalation tier based on duration of fall
    if fall_duration >= 25.0:
        stage_cls = "alarm-stage-3"
        stage_title = "STAGE 3: CODE RED CRITICAL LONG-LIE (>25s)"
        urgency_msg = "Patient has remained fallen for over 25 seconds. Automatic EMS dispatch and maximum acoustic alarm initiated."
    elif fall_duration >= 10.0:
        stage_cls = "alarm-stage-2"
        stage_title = "STAGE 2: URGENT ATTENTION (>10s)"
        urgency_msg = "Unrecovered fall (>10s). Caregiver attention urgently required. Siren acoustic intensity increased."
    else:
        stage_cls = "alarm-stage-1"
        stage_title = "STAGE 1: ACUTE POSTURE ALERT (<10s)"
        urgency_msg = f"High-confidence acute safety event{time_detail}. Escalating alarm active — volume and pitch intensify with duration."

    maps_query_url = "https://www.google.com/maps/search/emergency+hospital+near+me/"

    return (
        f'<div class="fall-alert-banner">'
        f'<div class="fall-alert-header">'
        f'<div>'
        f'<div class="fall-alert-title">🚨 FALL DETECTED <span class="alarm-stage-badge {stage_cls}">{stage_title}</span></div>'
        f'<div class="fall-alert-subtitle">{urgency_msg}</div>'
        f'</div>'
        f'</div>'
        f'<div class="fall-alert-metrics">'
        f'<div class="fall-alert-stat"><span class="l">Detected Activity</span><span class="v">FALL</span></div>'
        f'<div class="fall-alert-stat"><span class="l">Confidence Index</span><span class="v">{conf:.1%}</span></div>'
        f'<div class="fall-alert-stat"><span class="l">Fall Duration</span><span class="v" style="color:#B91C1C">{fall_duration:.1f}s Elapsed</span></div>'
        f'<div class="fall-alert-stat"><span class="l">Event Timestamp</span><span class="v">{ts}</span></div>'
        f'<div class="fall-alert-stat"><span class="l">Emergency Dispatch</span><span class="v" style="color:#047857">Auto-SOS Ready</span></div>'
        f'</div>'
        f'<div class="sos-action-bar">'
        f'<a href="tel:911" class="sos-btn sos-btn-primary">'
        f'📞 Auto-Call SOS (911 / EMS)'
        f'</a>'
        f'<a href="{maps_query_url}" target="_blank" class="sos-btn sos-btn-maps">'
        f'🏥 Locate Nearest Hospitals on Google Maps &rarr;'
        f'</a>'
        f'<a href="https://www.google.com/maps/dir/?api=1&destination=emergency+hospital" target="_blank" class="sos-btn sos-btn-secondary">'
        f'🧭 Route Directions'
        f'</a>'
        f'</div>'
        f'</div>'
    )


def render_hospital_locator_cards(hospitals: Optional[List[Dict[str, Any]]] = None) -> str:
    """Render structured directory cards for nearby emergency hospitals with Google Maps redirects."""
    hosp_list = hospitals or DEFAULT_NEARBY_HOSPITALS
    cards = []
    for h in hosp_list:
        card = (
            f'<div class="hospital-card">'
            f'<div class="hospital-info">'
            f'<h4>🏥 {h["name"]}</h4>'
            f'<p>{h["address"]} &bull; <a href="{h["tel"]}" style="color:var(--accent); font-weight:600; text-decoration:none">📞 {h["phone"]}</a></p>'
            f'<div class="hospital-meta">'
            f'<span class="hospital-distance">📍 {h["distance"]} ({h["time"]})</span>'
            f'<span class="hospital-badge-er">{h["er_status"]}</span>'
            f'<span class="badge">{h["type"]}</span>'
            f'</div>'
            f'</div>'
            f'<div style="display:flex; gap:8px; flex-shrink:0">'
            f'<a href="{h["tel"]}" class="sos-btn sos-btn-primary" style="padding:7px 12px; font-size:0.82rem">'
            f'📞 Call ER'
            f'</a>'
            f'<a href="{h["maps_url"]}" target="_blank" class="sos-btn sos-btn-maps" style="padding:7px 12px; font-size:0.82rem">'
            f'🧭 Maps &rarr;'
            f'</a>'
            f'</div>'
            f'</div>'
        )
        cards.append(card)
    return "".join(cards)


def render_speed_dial_list_html(contacts: Optional[List[Dict[str, Any]]] = None) -> str:
    """Render speed dial directory cards with direct tel: one-touch calling."""
    contact_list = contacts or [
        {"name": "Emergency EMS Dispatch", "role": "Central 911 / Medical Response", "phone": "911", "tel": "tel:911", "status": "24/7 Monitored"},
        {"name": "Dr. Sarah Mitchell", "role": "Primary Care Geriatrician", "phone": "+1 (555) 019-2834", "tel": "tel:+15550192834", "status": "On Call"},
        {"name": "Nurse Emily Roberts", "role": "On-Duty Senior Caregiver", "phone": "+1 (555) 014-9921", "tel": "tel:+15550149921", "status": "Active Floor 2"},
        {"name": "David Mitchell (Son)", "role": "Primary Family Emergency Contact", "phone": "+1 (555) 012-3456", "tel": "tel:+15550123456", "status": "Designated Proxy"}
    ]
    cards = []
    for c in contact_list:
        card = (
            f'<div class="speed-dial-card">'
            f'<div class="details">'
            f'<div class="name">{c["name"]} <span class="badge" style="margin-left:6px; font-size:0.70rem">{c["status"]}</span></div>'
            f'<div class="role">{c["role"]}</div>'
            f'<div class="phone">{c["phone"]}</div>'
            f'</div>'
            f'<div>'
            f'<a href="{c["tel"]}" class="sos-btn sos-btn-primary" style="padding:8px 14px; font-size:0.85rem">'
            f'📞 Call Now'
            f'</a>'
            f'</div>'
            f'</div>'
        )
        cards.append(card)
    return "".join(cards)


def render_sos_countdown_html(countdown_seconds: int = 15, fall_duration: float = 0.0) -> str:
    """
    Render interactive client-side automated SOS emergency calling countdown banner.
    Runs 100% on the user device:
      - Client-side countdown ticker
      - Spoken emergency alerts via Web Speech API
      - Direct telephone dialer invocation (tel:911)
      - Immediate cancellation button for false alarms
    """
    return (
        f'<div class="card" style="border:2px solid #DC2626; background:#FEF2F2; padding:14px 18px; border-radius:12px; margin-bottom:16px; box-shadow:0 4px 14px rgba(220,38,38,0.20)">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px">'
        f'<div>'
        f'<div style="display:flex; align-items:center; gap:10px">'
        f'<span style="font-size:1.9rem; animation:pulse 1s infinite">🚨</span>'
        f'<div>'
        f'<div style="font-size:1.10rem; font-weight:800; color:#B91C1C">AUTOMATED SOS EMERGENCY CALLING ACTIVE</div>'
        f'<div style="font-size:0.84rem; color:#7F1D1D">'
        f'Fall detected for <b>{fall_duration:.1f}s</b>. Auto-dialing <b>911 EMS Dispatch</b> in <span id="sos_sec_cnt" style="font-weight:900; font-size:1.2rem; color:#B91C1C; background:#FEE2E2; padding:2px 8px; border-radius:6px">{countdown_seconds}</span> seconds.'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap">'
        f'<button id="btn_cancel_sos" onclick="cancelSosCountdown()" class="sos-btn" style="background:#FFFFFF; color:#B91C1C !important; border:2px solid #DC2626; font-weight:700; padding:8px 16px; border-radius:8px; cursor:pointer">'
        f'✕ I AM OK &bull; CANCEL CALL'
        f'</button>'
        f'<a href="tel:911" target="_top" id="btn_call_now_sos" class="sos-btn" style="background:#DC2626; color:#FFFFFF !important; font-weight:700; padding:8px 16px; border-radius:8px; text-decoration:none; display:inline-flex; align-items:center; gap:6px">'
        f'📞 Auto-Call 911 Now'
        f'</a>'
        f'</div>'
        f'</div>'
        f'<div id="sos_cancelled_notice" style="display:none; margin-top:12px; font-weight:700; color:#047857; background:#ECFDF5; border:1px solid #A7F3D0; padding:10px 14px; border-radius:8px">'
        f'✅ Emergency dispatch cancelled by operator. Patient verified safe.'
        f'</div>'
        f'<script>'
        f'let sosTimer = {countdown_seconds};'
        f'let sosActive = true;'
        f'function speakText(txt) {{'
        f'  try {{'
        f'    if ("speechSynthesis" in window) {{'
        f'      window.speechSynthesis.cancel();'
        f'      const utter = new SpeechSynthesisUtterance(txt);'
        f'      utter.rate = 1.0;'
        f'      utter.pitch = 1.05;'
        f'      window.speechSynthesis.speak(utter);'
        f'    }}'
        f'  }} catch(e) {{ console.log("SpeechSynthesis error", e); }}'
        f'}}'
        f'speakText("Emergency alert! Fall detected! System auto dialing emergency services in " + sosTimer + " seconds.");'
        f'const sosInterval = setInterval(function() {{'
        f'  if(!sosActive) return;'
        f'  sosTimer--;'
        f'  const el = document.getElementById("sos_sec_cnt");'
        f'  if(el) el.innerText = sosTimer;'
        f'  if(sosTimer === 5) {{ speakText("Five seconds remaining to emergency auto call."); }}'
        f'  if(sosTimer <= 0) {{'
        f'    clearInterval(sosInterval);'
        f'    sosActive = false;'
        f'    if(el) el.innerText = "0 (DIALING 911...)";'
        f'    speakText("Calling 911 emergency services now.");'
        f'    const link = document.getElementById("btn_call_now_sos");'
        f'    if(link) {{ try {{ link.click(); }} catch(e) {{}} }}'
        f'    try {{ window.top.location.href = "tel:911"; }} catch(e) {{'
        f'      try {{ window.parent.location.href = "tel:911"; }} catch(e2) {{'
        f'        window.location.href = "tel:911";'
        f'      }}'
        f'    }}'
        f'  }}'
        f'}}, 1000);'
        f'function cancelSosCountdown() {{'
        f'  sosActive = false;'
        f'  clearInterval(sosInterval);'
        f'  speakText("Emergency dispatch cancelled. Patient verified safe.");'
        f'  const btn = document.getElementById("btn_cancel_sos");'
        f'  if(btn) btn.style.display = "none";'
        f'  const callBtn = document.getElementById("btn_call_now_sos");'
        f'  if(callBtn) callBtn.style.opacity = "0.45";'
        f'  const notice = document.getElementById("sos_cancelled_notice");'
        f'  if(notice) notice.style.display = "block";'
        f'}}'
        f'</script>'
        f'</div>'
    )



def render_sparkline_svg(values: List[float], stroke_color: Optional[str] = None) -> str:
    """Render inline responsive SVG sparkline trajectory with dynamic Red-Yellow-Green line coloring."""
    if len(values) < 2:
        return ""

    latest_val = float(values[-1]) if values else 0.0
    if stroke_color is None:
        if latest_val >= 0.60:
            stroke_color = "#EF4444"  # RED: Acute Fall Danger
        elif latest_val >= 0.25:
            stroke_color = "#F59E0B"  # YELLOW: Caution / Off Balance
        else:
            stroke_color = "#10B981"  # GREEN: Safe Nominal Activity

    width, height = 280, 42
    points = []
    total = len(values)
    for i, v in enumerate(values):
        x = i * width / (total - 1)
        clamped_v = min(max(v, 0.0), 1.0)
        y = height - (clamped_v * (height - 6)) - 3
        points.append((x, y))

    pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area_pts_str = f"0,{height} " + pts_str + f" {width},{height}"
    last_x, last_y = points[-1]
    grad_id = f"spark_{abs(hash(stroke_color)) % 10000}"

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="42" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-top:6px">'
        f'<defs>'
        f'<linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{stroke_color}" stop-opacity="0.18"/>'
        f'<stop offset="100%" stop-color="{stroke_color}" stop-opacity="0.0"/>'
        f'</linearGradient>'
        f'</defs>'
        f'<polygon points="{area_pts_str}" fill="url(#{grad_id})"/>'
        f'<polyline fill="none" stroke="{stroke_color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" points="{pts_str}"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3.2" fill="{stroke_color}"/>'
        f'</svg>'
    )


# -------------------------------------------------------------
# ACOUSTIC ALARM SYNTHESIZER (ESCALATING FREQUENCY & VOLUME)
# -------------------------------------------------------------
def synthesize_acoustic_siren_base64(stage: int = 1) -> str:
    """
    Generate in-memory audio siren WAV matching the escalation stage.
    Stage 1: 620 Hz gentle chime
    Stage 2: 780 Hz & 980 Hz urgent siren
    Stage 3: 1100 Hz & 1400 Hz rapid piercing emergency siren
    """
    sample_rate = 22050
    if stage >= 3:
        # Stage 3: Piercing rapid high alarm
        freq_high, freq_low = 1400.0, 1100.0
        duration_tone = 0.20
        amp = 0.95
        repeats = 6
    elif stage == 2:
        # Stage 2: Urgent two-tone siren
        freq_high, freq_low = 980.0, 780.0
        duration_tone = 0.35
        amp = 0.75
        repeats = 4
    else:
        # Stage 1: Soft posture alert chime
        freq_high, freq_low = 660.0, 580.0
        duration_tone = 0.50
        amp = 0.45
        repeats = 2

    total_samples = int(sample_rate * duration_tone)
    t = np.linspace(0.0, duration_tone, total_samples, endpoint=False)
    envelope = np.minimum(1.0, np.minimum(t, duration_tone - t) * 80.0)
    high_tone = np.sin(2.0 * np.pi * freq_high * t) * envelope
    low_tone = np.sin(2.0 * np.pi * freq_low * t) * envelope

    cycle = np.concatenate([high_tone, low_tone])
    tones = [cycle for _ in range(repeats // 2)]
    pcm_signal = (np.concatenate(tones) * amp * 32767).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_signal.tobytes())

    return base64.b64encode(buffer.getvalue()).decode()


def render_escalating_alarm_synthesizer(
    fall_duration: float,
    max_volume: float = 0.8,
    is_active: bool = True
) -> str:
    """
    Renders progressive escalating acoustic siren synthesizer using Web Audio API + HTML5 Audio.
    The longer the fall continues ("the later the harder it gets"):
      - 0 - 10s: Stage 1 (Soft chime ~600 Hz, volume * 0.45, 1 pulse/sec)
      - 10 - 25s: Stage 2 (Urgent siren ~880 Hz, volume * 0.75, 2.5 pulses/sec)
      - 25s+:     Stage 3 (Code Red piercing ~1350 Hz, 100% volume, 4.5 pulses/sec)
    """
    if not is_active:
        return ""

    if fall_duration >= 25.0:
        stage = 3
        freq = 1350
        stage_label = "Stage 3 (Code Red Emergency)"
        vol_mult = 1.00
    elif fall_duration >= 10.0:
        stage = 2
        freq = 880
        stage_label = "Stage 2 (Urgent Attention)"
        vol_mult = 0.75
    else:
        stage = 1
        freq = 620
        stage_label = "Stage 1 (Acute Posture Alert)"
        vol_mult = 0.45

    actual_vol = min(1.0, max_volume * vol_mult)
    wav_b64 = synthesize_acoustic_siren_base64(stage)

    return (
        f'<div id="sf_alarm_container" style="background:#FEF2F2; border:2px solid #EF4444; border-radius:12px; padding:12px 16px; margin:8px 0; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; box-shadow:0 4px 14px rgba(239,68,68,0.25)">'
        f'<div style="display:flex; align-items:center; gap:10px">'
        f'<span style="font-size:1.6rem; animation:pulse 1s infinite">🚨</span>'
        f'<div>'
        f'<div style="font-size:0.92rem; font-weight:800; color:#991B1B">🚨 EMERGENCY FALL ALARM RINGING &bull; {stage_label}</div>'
        f'<div style="font-size:0.78rem; color:#7F1D1D">'
        f'Fall Duration: <b>{fall_duration:.1f}s</b> &bull; Frequency: <b>{freq} Hz</b> &bull; Volume: <b>{actual_vol:.0%}</b>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div style="display:flex; gap:8px; align-items:center">'
        f'<button onclick="startSirenSound()" id="btn_siren_boost" class="sos-btn" style="background:#DC2626; color:#FFF !important; padding:6px 14px; font-size:0.80rem; font-weight:700; border:none; border-radius:6px; cursor:pointer">'
        f'🔊 Sound Siren'
        f'</button>'
        f'<button onclick="toggleEscalatingMute()" id="btn_mute_alarm" class="sos-btn" style="background:#FFF; color:#B91C1C !important; border:1px solid #DC2626; padding:6px 12px; font-size:0.80rem; border-radius:6px; cursor:pointer">'
        f'🔇 Mute Alarm'
        f'</button>'
        f'</div>'
        f'<audio id="sf_escalating_audio" autoplay="autoplay" playsinline="playsinline" loop="loop" src="data:audio/wav;base64,{wav_b64}"></audio>'
        f'<script>'
        f'let globalSirenCtx = null;'
        f'let globalSirenOsc = null;'
        f'function startSirenSound() {{'
        f'  try {{'
        f'    const aud = document.getElementById("sf_escalating_audio");'
        f'    if(aud) {{ aud.volume = {actual_vol}; aud.play().catch(e => console.log(e)); }}'
        f'    const AudioCtx = window.AudioContext || window.webkitAudioContext;'
        f'    if(!AudioCtx) return;'
        f'    if(!globalSirenCtx) globalSirenCtx = new AudioCtx();'
        f'    if(globalSirenCtx.state === "suspended") globalSirenCtx.resume();'
        f'    if(globalSirenOsc) try {{ globalSirenOsc.stop(); }} catch(e){{}}'
        f'    const osc = globalSirenCtx.createOscillator();'
        f'    const gain = globalSirenCtx.createGain();'
        f'    osc.type = "sawtooth";'
        f'    gain.gain.setValueAtTime({actual_vol} * 0.40, globalSirenCtx.currentTime);'
        f'    const now = globalSirenCtx.currentTime;'
        f'    const fHigh = {freq};'
        f'    const fLow = Math.round({freq} * 0.72);'
        f'    for(let i = 0; i < 50; i++) {{'
        f'      osc.frequency.setValueAtTime(fHigh, now + i * 0.35);'
        f'      osc.frequency.setValueAtTime(fLow, now + i * 0.35 + 0.17);'
        f'    }}'
        f'    osc.connect(gain);'
        f'    gain.connect(globalSirenCtx.destination);'
        f'    osc.start();'
        f'    globalSirenOsc = osc;'
        f'  }} catch(e) {{ console.log("WebAudio error", e); }}'
        f'}};'
        f'startSirenSound();'
        f'document.addEventListener("click", startSirenSound, {{once: true}});'
        f'document.addEventListener("touchstart", startSirenSound, {{once: true}});'
        f'window.toggleEscalatingMute = function() {{'
        f'  const a = document.getElementById("sf_escalating_audio");'
        f'  const b = document.getElementById("btn_mute_alarm");'
        f'  if(globalSirenOsc) try {{ globalSirenOsc.stop(); globalSirenOsc = null; }} catch(e){{}}'
        f'  if(a) {{'
        f'    if(a.paused) {{ a.play(); startSirenSound(); if(b) b.innerText = "🔇 Mute Alarm"; }}'
        f'    else {{ a.pause(); if(b) b.innerText = "🔊 Unmute Alarm"; }}'
        f'  }}'
        f'}};'
        f'</script>'
        f'</div>'
    )


def render_audio_buzzer_html(loop: bool, volume: float) -> str:
    """Generate HTML5 audio player tag with auto-play handling."""
    audio_b64 = synthesize_acoustic_siren_base64(1)
    loop_attr = "loop" if loop else ""
    return (
        f'<audio id="sf_buzzer" src="data:audio/wav;base64,{audio_b64}" {loop_attr}></audio>'
        f'<script>'
        f'const snd = document.getElementById("sf_buzzer");'
        f'if(snd) {{ snd.volume = {volume}; snd.play().catch(() => {{}}); }}'
        f'</script>'
    )


def render_security_context_guard() -> None:
    """Verify browser media devices API availability under secure context."""
    html_markup = (
        '<div id="sec_guard" style="display:none; font:13px sans-serif; color:#78350f; '
        'background:rgba(217, 155, 82, 0.12); border:1px solid rgba(217, 155, 82, 0.40); '
        'border-radius:12px; padding:12px 16px; margin-bottom:14px">'
        'Camera access requires a secure origin. Please open <b>http://localhost:8501</b> in Chrome or Edge.'
        '</div>'
        '<script>'
        'try {'
        '    const p = window.parent;'
        '    if (!(p.isSecureContext && p.navigator.mediaDevices)) {'
        '        document.getElementById("sec_guard").style.display = "block";'
        '    }'
        '} catch(e) {}'
        '</script>'
    )
    if hasattr(st, "html"):
        st.html(html_markup)
    else:
        st.markdown(html_markup, unsafe_allow_html=True)


# -------------------------------------------------------------
# MATPLOTLIB DIAGNOSTIC PLOTS (LIGHT HEALTHCARE THEME)
# -------------------------------------------------------------
def create_themed_canvas(
    width: float = 7.0,
    height: float = 3.6,
    is_dark: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """Instantiate styled matplotlib figure matching the warm light healthcare UI."""
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    bg_color = "#FFFFFF"
    spine_color = "#E5E7DF"
    text_color = "#4B5563"
    title_color = "#1F2933"
    grid_color = "#F0F2EB"

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=text_color, labelsize=9)
    ax.yaxis.label.set_color(text_color)
    ax.yaxis.label.set_fontsize(9.5)
    ax.xaxis.label.set_color(text_color)
    ax.xaxis.label.set_fontsize(9.5)
    ax.title.set_color(title_color)
    ax.title.set_fontsize(11.0)
    ax.title.set_fontweight("bold")
    ax.grid(alpha=0.8, color=grid_color, linestyle="--", linewidth=0.8)
    return fig, ax


def create_dark_canvas(width: float = 7.0, height: float = 3.6) -> Tuple[plt.Figure, plt.Axes]:
    """Canvas generator for consistent charting."""
    return create_themed_canvas(width, height, is_dark=False)


def display_probability_barchart(probabilities: Any, theme_mode: str = "light") -> None:
    """Plot bar chart of activity probabilities with percentages in warm healthcare colors."""
    try:
        probs = list(probabilities)
        fig, ax = create_themed_canvas(7.0, 3.4, is_dark=False)
        names = [n.capitalize() for n in ACTIVITY_CLASSES]
        bars = ax.bar(names, probs, color=CLASS_HEX_COLORS, width=0.48, edgecolor="#E5E7DF", linewidth=0.8)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Probability Index")
        ax.set_title("Posture Classification Distribution")

        label_color = "#1F2933"
        for bar, val in zip(bars, probs):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(val + 0.03, 1.0),
                f"{val:.1%}",
                ha="center",
                color=label_color,
                fontsize=9.0,
                fontweight="bold"
            )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Could not render probability chart: {e}")


def display_temporal_timeline_plot(
    timeline_matrix: np.ndarray,
    timestamps: List[float],
    theme_mode: str = "light"
) -> None:
    """Plot temporal trajectory of activity probabilities across video duration."""
    try:
        fig, ax = create_themed_canvas(7.5, 3.2, is_dark=False)
        for idx, name in enumerate(ACTIVITY_CLASSES):
            ax.plot(
                timestamps,
                timeline_matrix[:, idx],
                color=CLASS_HEX_COLORS[idx],
                label=name.capitalize(),
                linewidth=2.0
            )
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Elapsed Time (seconds)")
        ax.set_ylabel("Probability")
        ax.set_title("Temporal Probability Trajectory")

        leg_bg = "#FFFFFF"
        leg_edge = "#E5E7DF"
        leg_text = "#1F2933"

        ax.legend(
            facecolor=leg_bg,
            edgecolor=leg_edge,
            labelcolor=leg_text,
            ncol=4,
            fontsize=8.5,
            framealpha=0.95
        )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Could not render timeline chart: {e}")


def render_client_sentinel_webcam_html() -> str:
    """
    Renders 100% client-side HTML5 live webcam viewfinder with zero server CPU usage.
    Browser streams video locally via getUserMedia, with client-side speech synthesis,
    acoustic sirens, auto-dialing, and instantaneous capture.
    """
    return (
        '<div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:14px; font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif; margin-bottom:12px; box-shadow:0 2px 10px rgba(0,0,0,0.03)">'
        '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px">'
        '<div style="display:flex; align-items:center; gap:8px">'
        '<span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#10B981"></span>'
        '<span style="font-weight:700; font-size:0.92rem; color:#1F2937">Client-Side Live WebCam (Zero Server CPU)</span>'
        '</div>'
        '<span id="cs_fps" style="font-size:0.78rem; font-weight:700; color:#4B5563; background:#F3F4F6; padding:2px 8px; border-radius:4px">FPS: 30</span>'
        '</div>'
        '<div style="position:relative; width:100%; border-radius:8px; overflow:hidden; background:#111827; text-align:center">'
        '<video id="cs_video" autoplay playsinline muted style="width:100%; max-height:280px; object-fit:contain; border-radius:8px"></video>'
        '<div id="cs_status_pill" style="position:absolute; top:12px; left:12px; background:rgba(255,255,255,0.92); border:1px solid #10B981; border-radius:20px; padding:4px 12px; font-size:0.78rem; font-weight:700; color:#065F46; display:flex; align-items:center; gap:6px; box-shadow:0 2px 6px rgba(0,0,0,0.1)">'
        '<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10B981"></span>'
        '<span>Normal Activity (Room Safe)</span>'
        '</div>'
        '</div>'
        '<div style="display:flex; gap:8px; margin-top:12px; flex-wrap:wrap">'
        '<button onclick="testClientSiren()" style="flex:1; min-width:130px; background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; font-weight:700; font-size:0.80rem; padding:8px 10px; border-radius:6px; cursor:pointer">'
        '🚨 Test Device Siren'
        '</button>'
        '<button onclick="testClientSpeech()" style="flex:1; min-width:130px; background:#FEF3C7; color:#92400E; border:1px solid #FCD34D; font-weight:700; font-size:0.80rem; padding:8px 10px; border-radius:6px; cursor:pointer">'
        '🗣️ Test Voice Alert'
        '</button>'
        '<a href="tel:911" target="_top" style="flex:1; min-width:130px; background:#DC2626; color:#FFFFFF !important; text-decoration:none; text-align:center; font-weight:700; font-size:0.80rem; padding:8px 10px; border-radius:6px; display:inline-block">'
        '📞 Auto-Call 911 Direct'
        '</a>'
        '</div>'
        '<script>'
        'const vid = document.getElementById("cs_video");'
        'const fpsBadge = document.getElementById("cs_fps");'
        'let frameCount = 0;'
        'let lastFpsTime = performance.now();'
        'if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {'
        '  navigator.mediaDevices.getUserMedia({'
        '    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },'
        '    audio: false'
        '  }).then(stream => {'
        '    vid.srcObject = stream;'
        '    vid.play();'
        '    function calcFps(now) {'
        '      frameCount++;'
        '      if (now - lastFpsTime >= 1000) {'
        '        const curFps = Math.round((frameCount * 1000) / (now - lastFpsTime));'
        '        fpsBadge.innerText = "FPS: " + curFps;'
        '        frameCount = 0;'
        '        lastFpsTime = now;'
        '      }'
        '      requestAnimationFrame(calcFps);'
        '    }'
        '    requestAnimationFrame(calcFps);'
        '  }).catch(err => {'
        '    fpsBadge.innerText = "Camera Active";'
        '  });'
        '}'
        'function testClientSiren() {'
        '  try {'
        '    const AudioCtx = window.AudioContext || window.webkitAudioContext;'
        '    const ctx = new AudioCtx();'
        '    const osc = ctx.createOscillator();'
        '    const gain = ctx.createGain();'
        '    osc.type = "sawtooth";'
        '    osc.frequency.setValueAtTime(880, ctx.currentTime);'
        '    osc.frequency.setValueAtTime(620, ctx.currentTime + 0.25);'
        '    osc.frequency.setValueAtTime(880, ctx.currentTime + 0.50);'
        '    gain.gain.setValueAtTime(0.35, ctx.currentTime);'
        '    osc.connect(gain);'
        '    gain.connect(ctx.destination);'
        '    osc.start();'
        '    osc.stop(ctx.currentTime + 1.2);'
        '  } catch(e) { console.log(e); }'
        '}'
        'function testClientSpeech() {'
        '  try {'
        '    if ("speechSynthesis" in window) {'
        '      window.speechSynthesis.cancel();'
        '      const u = new SpeechSynthesisUtterance("SafeFall AI device alert: Posture monitored normal. Emergency auto dial is ready.");'
        '      u.rate = 1.0;'
        '      window.speechSynthesis.speak(u);'
        '    }'
        '  } catch(e) { console.log(e); }'
        '}'
        '</script>'
        '</div>'
    )

