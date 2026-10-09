"""
ui/styles.py
============
Clean, modern, and accessible design system for SafeFall AI.
Premium Healthcare + AI Computer Vision Product aesthetic.
Warm off-white ivory canvas, deep charcoal typography, soft sage accents,
muted coral alert states, and generous spacing.
"""

from __future__ import annotations

from typing import Dict, Tuple

import streamlit as st

PALETTES: Dict[str, Tuple[str, str]] = {
    "Healthcare Sage": ("#5E8B7A", "#8EA8C3"),
    "Clinical Teal": ("#4E878C", "#65B891"),
    "Muted Lavender": ("#7B8FA1", "#A7BBC7"),
    "Warm Neutral": ("#6B705C", "#A5A58D"),
    "Calm Azure": ("#3B82A6", "#8EA8C3"),
}

# 6 Clinical Activity Classes: Sitting, Standing, Walking, Off Balance, Normal Activity, Fall
# Standard Red - Yellow - Green Color System:
# Green: SITTING (#40916C), STANDING (#52B788), WALKING (#2D6A4F), NORMAL_ACTIVITY (#10B981)
# Yellow: OFF_BALANCE (#F59E0B)
# Red: FALL (#EF4444)
CLASS_HEX_COLORS = ["#40916C", "#52B788", "#2D6A4F", "#F59E0B", "#10B981", "#EF4444"]
CLASS_GLYPHS = {
    "SITTING": "🪑",
    "STANDING": "🧍",
    "WALKING": "🚶",
    "OFF_BALANCE": "⚠️",
    "NORMAL_ACTIVITY": "🧘",
    "FALL": "🚨"
}


def hex_to_rgb_string(hex_color: str) -> str:
    """Convert hex string (e.g. #5E8B7A) to comma-separated RGB (e.g. 94,139,122)."""
    c = hex_color.lstrip("#")
    return f"{int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)}"


def generate_stylesheet(accent_hex: str = "#5E8B7A", accent2_hex: str = "#8EA8C3", theme_mode: str = "light") -> str:
    """Generate clean, accessible, modern healthcare product stylesheet for SafeFall AI."""
    rgb_primary = hex_to_rgb_string(accent_hex)
    rgb_secondary = hex_to_rgb_string(accent2_hex)

    # Core Healthcare Palette (Warm light neutral default)
    theme_vars = f"""
--accent: {accent_hex};
--accent-hover: #4D7466;
--accent2: {accent2_hex};
--ar: {rgb_primary};
--br: {rgb_secondary};
--bg-canvas: #F7F7F3;
--bg-surface: #FFFFFF;
--bg-elevated: #F2F3EC;
--bg-subtle: #FAFBF8;
--border-subtle: #E5E7DF;
--border-focus: #C8CCC0;
--border-accent: rgba(var(--ar), 0.45);
--text-primary: #1F2933;
--text-secondary: #4B5563;
--text-tertiary: #7C8894;
--shadow-sm: 0 1px 3px rgba(31, 41, 51, 0.04);
--shadow-md: 0 4px 14px rgba(31, 41, 51, 0.06), 0 1px 3px rgba(31, 41, 51, 0.03);
--shadow-lg: 0 10px 25px rgba(31, 41, 51, 0.08);
--input-bg: #FFFFFF;
--input-border: #D2D6CD;
--sidebar-bg: #FBFBFA;
--sidebar-border: #E8EAE2;
--status-green: #489975;
--status-green-bg: rgba(72, 153, 117, 0.10);
--status-amber: #D99B52;
--status-amber-bg: rgba(217, 155, 82, 0.10);
--status-red: #D97878;
--status-red-bg: rgba(217, 120, 120, 0.10);
"""

    return f"""<style>
:root {{
{theme_vars}
}}

html, body, [class*="css"] {{
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
-webkit-font-smoothing: antialiased;
color: var(--text-primary);
}}

.stApp {{
background-color: var(--bg-canvas);
background-image: radial-gradient(1200px 600px at 50% 0%, rgba(var(--ar), 0.035), transparent 70%);
color: var(--text-primary);
}}

[data-testid="stAppViewContainer"], [data-testid="stMain"] {{
background: transparent;
}}

.block-container {{
max-width: 1360px;
padding-top: 1.2rem;
padding-bottom: 3.5rem;
position: relative;
z-index: 1;
}}

header[data-testid="stHeader"] {{
background: transparent;
}}

#MainMenu, footer {{
visibility: hidden;
}}

h1, h2, h3, h4 {{
font-family: "Plus Jakarta Sans", sans-serif;
font-weight: 700;
letter-spacing: -0.02em;
color: var(--text-primary) !important;
margin: 0;
}}

p, span, label {{
color: inherit;
}}

::selection {{
background: rgba(var(--ar), 0.20);
color: var(--text-primary);
}}

/* ========================================================= */
/* TOP NAVIGATION / HEALTHCARE PRODUCT HEADER                */
/* ========================================================= */
.top-header {{
display: flex;
justify-content: space-between;
align-items: center;
gap: 16px;
flex-wrap: wrap;
padding: 16px 24px;
border-radius: 16px;
margin-bottom: 20px;
background: var(--bg-surface);
border: 1px solid var(--border-subtle);
box-shadow: var(--shadow-sm);
}}

.top-header-left {{
display: flex;
align-items: center;
gap: 14px;
}}

.top-header-brand {{
display: flex;
flex-direction: column;
}}

.top-header-brand h1 {{
font-size: 1.45rem;
font-weight: 700;
line-height: 1.2;
color: var(--text-primary) !important;
letter-spacing: -0.025em;
}}

.top-header-brand p {{
font-size: 0.84rem;
color: var(--text-secondary);
margin: 2px 0 0;
font-weight: 500;
}}

.top-header-right {{
display: flex;
align-items: center;
gap: 10px;
flex-wrap: wrap;
}}

/* Badges & Pills */
.badge {{
display: inline-flex;
align-items: center;
gap: 6px;
padding: 6px 14px;
border-radius: 9999px;
background: var(--bg-elevated);
border: 1px solid var(--border-subtle);
font-size: 0.80rem;
font-weight: 600;
color: var(--text-secondary);
}}

.badge.active {{
color: var(--text-primary);
border-color: var(--border-accent);
background: #FFFFFF;
box-shadow: var(--shadow-sm);
}}

.status-dot {{
width: 8px;
height: 8px;
border-radius: 50%;
display: inline-block;
background: var(--status-green);
}}

.status-dot.amber {{
background: var(--status-amber);
}}

.status-dot.red {{
background: var(--status-red);
}}

.status-dot.pulse {{
animation: status-ping 2.0s cubic-bezier(0, 0, 0.2, 1) infinite;
}}

@keyframes status-ping {{
75%, 100% {{ transform: scale(1.7); opacity: 0; }}
}}

/* ========================================================= */
/* CARDS & HEALTHCARE SURFACES                               */
/* ========================================================= */
.card {{
background: var(--bg-surface);
border: 1px solid var(--border-subtle);
border-radius: 16px;
padding: 22px;
margin-bottom: 18px;
box-shadow: var(--shadow-sm);
transition: border-color 0.2s ease, box-shadow 0.2s ease;
color: var(--text-primary);
}}

.card:hover {{
border-color: var(--border-focus);
box-shadow: var(--shadow-md);
}}

.card.elevated {{
background: var(--bg-elevated);
}}

.card-header {{
display: flex;
justify-content: space-between;
align-items: center;
margin-bottom: 14px;
}}

.card-title {{
font-size: 1.0rem;
font-weight: 600;
color: var(--text-primary);
display: flex;
align-items: center;
gap: 8px;
letter-spacing: -0.01em;
}}

/* Section Titles */
.section-heading {{
margin: 18px 0 14px;
}}

.section-heading h2 {{
font-size: 1.35rem;
font-weight: 700;
color: var(--text-primary) !important;
letter-spacing: -0.02em;
}}

.section-heading p {{
font-size: 0.88rem;
color: var(--text-secondary);
margin: 4px 0 0;
line-height: 1.5;
}}

/* ========================================================= */
/* HERO / TODAY'S MONITORING SECTION                         */
/* ========================================================= */
.hero-box {{
background: var(--bg-surface);
border: 1px solid var(--border-subtle);
border-radius: 18px;
padding: 26px;
margin-bottom: 22px;
box-shadow: var(--shadow-md);
position: relative;
overflow: hidden;
}}

.hero-box::before {{
content: "";
position: absolute;
top: 0;
left: 0;
right: 0;
height: 4px;
background: linear-gradient(90deg, var(--accent), var(--accent2));
}}

.hero-header {{
display: flex;
justify-content: space-between;
align-items: center;
margin-bottom: 18px;
flex-wrap: wrap;
gap: 12px;
}}

.hero-title {{
font-size: 1.25rem;
font-weight: 700;
color: var(--text-primary);
letter-spacing: -0.02em;
}}

.hero-subtitle {{
font-size: 0.86rem;
color: var(--text-secondary);
margin-top: 2px;
}}

.live-activity-callout {{
display: flex;
align-items: center;
justify-content: space-between;
padding: 20px 24px;
border-radius: 14px;
background: var(--bg-subtle);
border: 1px solid var(--border-subtle);
margin: 14px 0;
gap: 16px;
flex-wrap: wrap;
}}

.activity-display-label {{
font-size: 0.78rem;
font-weight: 600;
letter-spacing: 0.05em;
text-transform: uppercase;
color: var(--text-tertiary);
}}

.activity-display-val {{
font-size: 2.2rem;
font-weight: 800;
letter-spacing: -0.02em;
color: var(--text-primary);
line-height: 1.1;
margin-top: 4px;
}}

.activity-display-conf {{
font-size: 1.15rem;
font-weight: 600;
color: var(--accent);
margin-top: 2px;
}}

/* ========================================================= */
/* ACTIVITY CARDS (SITTING, STANDING, WALKING, OFF BALANCE, NORMAL, FALL) */
/* ========================================================= */
.activity-grid {{
display: grid;
grid-template-columns: repeat(6, 1fr);
gap: 12px;
margin: 16px 0 20px;
}}

@media (max-width: 1200px) {{
.activity-grid {{
grid-template-columns: repeat(3, 1fr);
}}
}}

@media (max-width: 640px) {{
.activity-grid {{
grid-template-columns: repeat(2, 1fr);
}}
}}

.activity-card {{
background: var(--bg-surface);
border: 1px solid var(--border-subtle);
border-radius: 16px;
padding: 18px;
box-shadow: var(--shadow-sm);
transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
position: relative;
}}

.activity-card:hover {{
border-color: var(--border-focus);
transform: translateY(-2px);
box-shadow: var(--shadow-md);
}}

.activity-card.active {{
border-color: var(--c);
background: color-mix(in srgb, var(--c) 8%, #FFFFFF);
box-shadow: 0 0 0 1px var(--c), var(--shadow-md);
}}

.activity-card-header {{
display: flex;
justify-content: space-between;
align-items: center;
margin-bottom: 8px;
}}

.activity-card-tag {{
font-size: 0.72rem;
font-weight: 600;
letter-spacing: 0.06em;
text-transform: uppercase;
color: var(--text-tertiary);
}}

.activity-card.active .activity-card-tag {{
color: var(--c);
font-weight: 700;
}}

.activity-card-icon {{
font-size: 1.6rem;
display: inline-block;
margin-bottom: 6px;
}}

.activity-card-label {{
font-size: 0.88rem;
font-weight: 600;
color: var(--text-secondary);
}}

.activity-card-val {{
font-size: 1.7rem;
font-weight: 700;
color: var(--text-primary);
letter-spacing: -0.02em;
margin-top: 3px;
}}

/* ========================================================= */
/* HORIZONTAL PROBABILITY INDICATORS                         */
/* ========================================================= */
.prob-row {{
display: flex;
align-items: center;
gap: 14px;
padding: 8px 0;
}}

.prob-label {{
width: 140px;
min-width: 130px;
font-size: 0.85rem;
font-weight: 600;
color: var(--text-primary);
display: flex;
align-items: center;
gap: 6px;
flex: none;
white-space: nowrap;
}}

.prob-track {{
flex: 1;
height: 10px;
border-radius: 9999px;
background: var(--bg-elevated);
overflow: hidden;
border: 1px solid var(--border-subtle);
}}

.prob-fill {{
height: 100%;
border-radius: 9999px;
transition: width 0.35s ease;
}}

.prob-pct {{
width: 60px;
font-size: 0.88rem;
font-weight: 700;
color: var(--text-primary);
text-align: right;
flex: none;
font-variant-numeric: tabular-nums;
}}

.progress-track {{
height: 8px;
border-radius: 9999px;
background: var(--bg-elevated);
overflow: hidden;
margin-top: 10px;
border: 1px solid var(--border-subtle);
}}

.progress-fill {{
height: 100%;
border-radius: 9999px;
background: var(--c);
transition: width 0.35s ease;
}}

/* ========================================================= */
/* FALL DETECTED SOPHISTICATED ALERT STATE                   */
/* ========================================================= */
.fall-alert-banner {{
background: var(--status-red-bg);
border: 1px solid rgba(217, 120, 120, 0.45);
border-left: 6px solid var(--status-red);
border-radius: 16px;
padding: 22px 26px;
margin: 18px 0;
box-shadow: var(--shadow-sm);
animation: alert-soft-pulse 2.8s ease-in-out infinite;
}}

@keyframes alert-soft-pulse {{
0%, 100% {{ box-shadow: 0 0 0 0 rgba(217, 120, 120, 0.20); }}
50% {{ box-shadow: 0 0 0 6px rgba(217, 120, 120, 0.08); }}
}}

.fall-alert-header {{
display: flex;
justify-content: space-between;
align-items: center;
flex-wrap: wrap;
gap: 12px;
margin-bottom: 8px;
}}

.fall-alert-title {{
font-size: 1.35rem;
font-weight: 800;
color: #B93838;
letter-spacing: -0.02em;
display: flex;
align-items: center;
gap: 10px;
}}

.fall-alert-subtitle {{
font-size: 0.90rem;
color: #8C3232;
font-weight: 500;
line-height: 1.4;
}}

.fall-alert-metrics {{
display: flex;
gap: 20px;
margin-top: 14px;
padding-top: 12px;
border-top: 1px solid rgba(217, 120, 120, 0.25);
flex-wrap: wrap;
}}

.fall-alert-stat {{
display: flex;
flex-direction: column;
}}

.fall-alert-stat .l {{
font-size: 0.72rem;
font-weight: 600;
text-transform: uppercase;
letter-spacing: 0.05em;
color: #8C3232;
}}

.fall-alert-stat .v {{
font-size: 1.15rem;
font-weight: 700;
color: #6E1B1B;
margin-top: 1px;
}}

/* ========================================================= */
/* ESCALATING ALARM & EMERGENCY SOS COMPONENTS               */
/* ========================================================= */
.alarm-stage-badge {{
display: inline-flex;
align-items: center;
gap: 8px;
padding: 6px 14px;
border-radius: 9999px;
font-size: 0.82rem;
font-weight: 700;
letter-spacing: 0.03em;
text-transform: uppercase;
}}

.alarm-stage-1 {{
background: rgba(217, 155, 82, 0.15);
color: #B45309;
border: 1px solid rgba(217, 155, 82, 0.40);
}}

.alarm-stage-2 {{
background: rgba(217, 120, 120, 0.18);
color: #B93838;
border: 1px solid rgba(217, 120, 120, 0.50);
animation: alert-soft-pulse 1.2s infinite;
}}

.alarm-stage-3 {{
background: #991B1B;
color: #FFFFFF;
border: 1px solid #7F1D1D;
box-shadow: 0 0 14px rgba(185, 28, 28, 0.55);
animation: alert-red-flash 0.6s infinite alternate;
}}

@keyframes alert-red-flash {{
from {{ background: #991B1B; box-shadow: 0 0 8px rgba(185, 28, 28, 0.4); }}
to {{ background: #DC2626; box-shadow: 0 0 20px rgba(220, 38, 38, 0.85); }}
}}

.sos-action-bar {{
display: flex;
gap: 12px;
margin-top: 14px;
flex-wrap: wrap;
align-items: center;
}}

.sos-btn {{
display: inline-flex;
align-items: center;
justify-content: center;
gap: 8px;
padding: 10px 18px;
border-radius: 12px;
font-size: 0.90rem;
font-weight: 700;
text-decoration: none !important;
transition: all 0.2s ease;
cursor: pointer;
}}

.sos-btn-primary {{
background: #DC2626;
color: #FFFFFF !important;
border: 1px solid #B91C1C;
box-shadow: 0 2px 8px rgba(220, 38, 38, 0.35);
}}

.sos-btn-primary:hover {{
background: #B91C1C;
transform: translateY(-1px);
box-shadow: 0 4px 12px rgba(220, 38, 38, 0.45);
}}

.sos-btn-maps {{
background: #1A73E8;
color: #FFFFFF !important;
border: 1px solid #1557B0;
box-shadow: 0 2px 8px rgba(26, 115, 232, 0.30);
}}

.sos-btn-maps:hover {{
background: #1557B0;
transform: translateY(-1px);
box-shadow: 0 4px 12px rgba(26, 115, 232, 0.45);
}}

.sos-btn-secondary {{
background: #FFFFFF;
color: #374151 !important;
border: 1px solid #D1D5DB;
}}

.sos-btn-secondary:hover {{
background: #F9FAFB;
border-color: #9CA3AF;
}}

.sos-circle-wrapper {{
display: flex;
flex-direction: column;
align-items: center;
justify-content: center;
padding: 24px 16px;
margin: 6px auto;
text-align: center;
}}

.sos-circle-btn {{
width: 170px;
height: 170px;
border-radius: 50% !important;
background: radial-gradient(circle at 35% 35%, #EF4444 0%, #DC2626 45%, #991B1B 100%) !important;
border: 4px solid #FCA5A5 !important;
color: #FFFFFF !important;
display: flex !important;
flex-direction: column !important;
align-items: center !important;
justify-content: center !important;
text-decoration: none !important;
box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6), 0 10px 30px rgba(185, 28, 28, 0.45) !important;
transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
cursor: pointer !important;
animation: sos-pulse-glow 2.5s infinite !important;
}}

.sos-circle-btn:hover {{
transform: scale(1.06) !important;
box-shadow: 0 0 0 16px rgba(239, 68, 68, 0.25), 0 16px 40px rgba(185, 28, 28, 0.6) !important;
border-color: #FFFFFF !important;
}}

.sos-circle-btn:active {{
transform: scale(0.96) !important;
}}

.sos-circle-icon {{
font-size: 2.4rem;
line-height: 1;
margin-bottom: 4px;
}}

.sos-circle-title {{
font-size: 1.55rem;
font-weight: 900;
letter-spacing: 0.08em;
line-height: 1;
}}

.sos-circle-sub {{
font-size: 0.72rem;
font-weight: 700;
letter-spacing: 0.05em;
opacity: 0.92;
margin-top: 3px;
}}

@keyframes sos-pulse-glow {{
0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6), 0 8px 24px rgba(185, 28, 28, 0.35); }}
70% {{ box-shadow: 0 0 0 20px rgba(239, 68, 68, 0), 0 8px 24px rgba(185, 28, 28, 0.35); }}
100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0), 0 8px 24px rgba(185, 28, 28, 0.35); }}
}}

.hospital-card {{
background: var(--bg-surface);
border: 1px solid var(--border-subtle);
border-radius: 14px;
padding: 18px;
margin-bottom: 12px;
display: flex;
justify-content: space-between;
align-items: center;
transition: all 0.2s ease;
box-shadow: var(--shadow-sm);
}}

.hospital-card:hover {{
border-color: var(--border-focus);
transform: translateY(-2px);
box-shadow: var(--shadow-md);
}}

.hospital-info h4 {{
font-size: 1.05rem;
font-weight: 700;
color: var(--text-primary) !important;
margin-bottom: 3px;
}}

.hospital-info p {{
font-size: 0.85rem;
color: var(--text-secondary);
margin: 0;
}}

.hospital-meta {{
display: flex;
gap: 8px;
margin-top: 6px;
align-items: center;
flex-wrap: wrap;
}}

.hospital-distance {{
font-size: 0.78rem;
font-weight: 700;
color: var(--accent);
background: rgba(var(--ar), 0.12);
padding: 3px 8px;
border-radius: 6px;
}}

.hospital-badge-er {{
font-size: 0.78rem;
font-weight: 600;
color: #047857;
background: rgba(4, 120, 87, 0.12);
padding: 3px 8px;
border-radius: 6px;
}}

.speed-dial-card {{
background: var(--bg-surface);
border: 1px solid var(--border-subtle);
border-radius: 14px;
padding: 16px;
display: flex;
align-items: center;
justify-content: space-between;
margin-bottom: 10px;
}}

.speed-dial-card .details .name {{
font-weight: 700;
font-size: 0.95rem;
color: var(--text-primary);
}}

.speed-dial-card .details .role {{
font-size: 0.82rem;
color: var(--text-secondary);
}}

.speed-dial-card .details .phone {{
font-size: 0.84rem;
font-family: 'JetBrains Mono', monospace;
color: var(--accent);
font-weight: 600;
margin-top: 2px;
}}

/* ========================================================= */
/* BENTO KPI STAT TILES                                      */
/* ========================================================= */
.stat-grid {{
display: grid;
grid-template-columns: repeat(4, 1fr);
gap: 14px;
margin: 16px 0 20px;
}}

.stat-tile {{
background: var(--bg-surface);
border: 1px solid var(--border-subtle);
border-radius: 14px;
padding: 16px 20px;
box-shadow: var(--shadow-sm);
transition: border-color 0.2s ease, transform 0.2s ease;
}}

.stat-tile:hover {{
border-color: var(--border-focus);
transform: translateY(-2px);
}}

.stat-tile .l {{
font-size: 0.74rem;
font-weight: 600;
text-transform: uppercase;
letter-spacing: 0.05em;
color: var(--text-tertiary);
}}

.stat-tile .v {{
font-size: 1.75rem;
font-weight: 700;
color: var(--text-primary);
margin-top: 4px;
letter-spacing: -0.02em;
line-height: 1.15;
}}

.stat-tile .sub {{
font-size: 0.78rem;
color: var(--text-secondary);
margin-top: 4px;
}}

/* ========================================================= */
/* MEDIA ANALYSIS PIPELINE & UPLOAD ZONE                     */
/* ========================================================= */
.pipeline-breadcrumb {{
display: flex;
align-items: center;
justify-content: space-between;
padding: 14px 20px;
background: var(--bg-elevated);
border-radius: 14px;
border: 1px solid var(--border-subtle);
margin-bottom: 20px;
flex-wrap: wrap;
gap: 10px;
}}

.pipeline-step {{
display: flex;
align-items: center;
gap: 8px;
font-size: 0.84rem;
font-weight: 600;
color: var(--text-secondary);
}}

.pipeline-step.active {{
color: var(--accent);
font-weight: 700;
}}

.pipeline-arrow {{
color: var(--text-tertiary);
font-size: 0.90rem;
}}

.upload-dropzone {{
border: 2px dashed var(--border-focus);
border-radius: 18px;
padding: 34px 24px;
text-align: center;
background: var(--bg-subtle);
transition: border-color 0.2s ease, background 0.2s ease;
margin-bottom: 20px;
}}

.upload-dropzone:hover {{
border-color: var(--accent);
background: #FFFFFF;
}}

.upload-icon {{
font-size: 2.2rem;
margin-bottom: 8px;
color: var(--accent);
}}

.upload-title {{
font-size: 1.15rem;
font-weight: 700;
color: var(--text-primary);
}}

.upload-desc {{
font-size: 0.86rem;
color: var(--text-secondary);
margin-top: 4px;
}}

/* ========================================================= */
/* DIAGNOSTIC VERDICT CARD                                   */
/* ========================================================= */
.verdict-box {{
display: flex;
justify-content: space-between;
align-items: center;
padding: 24px 28px;
border-radius: 16px;
border: 1px solid var(--border-subtle);
background: var(--bg-surface);
box-shadow: var(--shadow-sm);
margin-bottom: 18px;
gap: 20px;
flex-wrap: wrap;
}}

.verdict-box.fall {{
border-color: var(--status-red);
background: color-mix(in srgb, var(--status-red) 7%, #FFFFFF);
}}

.verdict-tag {{
font-size: 0.74rem;
font-weight: 700;
letter-spacing: 0.08em;
text-transform: uppercase;
color: var(--text-secondary);
}}

.verdict-val {{
font-size: 2.4rem;
font-weight: 800;
letter-spacing: -0.025em;
margin-top: 4px;
line-height: 1.1;
}}

.verdict-meta {{
font-size: 0.88rem;
color: var(--text-secondary);
margin-top: 6px;
}}

/* Circular Confidence Indicator */
.confidence-dial {{
width: 94px;
height: 94px;
border-radius: 50%;
display: grid;
place-items: center;
background: conic-gradient(var(--rc) calc(var(--p) * 1%), var(--bg-elevated) 0);
box-shadow: var(--shadow-sm);
flex: none;
}}

.confidence-dial span {{
width: 76px;
height: 76px;
border-radius: 50%;
background: var(--bg-surface);
display: grid;
place-items: center;
font-weight: 700;
font-size: 1.25rem;
color: var(--text-primary);
font-variant-numeric: tabular-nums;
}}

/* ========================================================= */
/* STREAMLIT WIDGET POLISH                                   */
/* ========================================================= */
.stButton > button, .stDownloadButton > button {{
border-radius: 10px !important;
border: 1px solid var(--border-subtle) !important;
background: var(--bg-surface) !important;
color: var(--text-primary) !important;
font-weight: 600 !important;
font-size: 0.88rem !important;
padding: 9px 18px !important;
box-shadow: var(--shadow-sm) !important;
transition: background 0.15s ease, border-color 0.15s ease, transform 0.15s ease !important;
}}

.stButton > button:hover, .stDownloadButton > button:hover {{
border-color: var(--border-focus) !important;
background: var(--bg-elevated) !important;
transform: translateY(-1px);
}}

[data-testid="stForm"] {{
background: var(--bg-surface) !important;
border: 1px solid var(--border-subtle) !important;
border-radius: 16px !important;
padding: 24px 28px !important;
box-shadow: var(--shadow-sm) !important;
margin-bottom: 14px !important;
}}

.stFormSubmitButton > button {{
background: var(--accent) !important;
color: #FFFFFF !important;
border: 1px solid var(--accent) !important;
box-shadow: 0 1px 3px rgba(var(--ar), 0.30) !important;
font-weight: 700 !important;
border-radius: 10px !important;
padding: 9px 20px !important;
}}

.stFormSubmitButton > button:hover {{
background: var(--accent-hover) !important;
border-color: var(--accent-hover) !important;
color: #FFFFFF !important;
}}

/* Clean Radiogroup Pills */
div[role="radiogroup"] {{
display: flex;
gap: 8px;
background: var(--bg-elevated);
padding: 6px;
border-radius: 14px;
border: 1px solid var(--border-subtle);
width: fit-content;
margin-bottom: 16px;
flex-wrap: wrap;
}}

div[role="radiogroup"] label {{
border: none !important;
padding: 8px 18px !important;
border-radius: 10px !important;
background: transparent !important;
font-weight: 600 !important;
font-size: 0.88rem !important;
color: var(--text-secondary) !important;
cursor: pointer;
transition: background 0.15s ease, color 0.15s ease, box-shadow 0.15s ease;
}}

div[role="radiogroup"] label:hover {{
color: var(--text-primary) !important;
}}

div[role="radiogroup"] label:has(input:checked) {{
background: var(--bg-surface) !important;
color: var(--text-primary) !important;
box-shadow: var(--shadow-sm) !important;
font-weight: 700 !important;
}}

/* Form Inputs */
[data-testid="stTextInput"] input {{
border-radius: 10px;
background: var(--input-bg) !important;
border: 1px solid var(--input-border) !important;
padding: 10px 14px;
color: var(--text-primary) !important;
font-size: 0.92rem;
transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}

[data-testid="stTextInput"] input:focus {{
border-color: var(--accent) !important;
box-shadow: 0 0 0 3px rgba(var(--ar), 0.15) !important;
}}

/* Expanders */
[data-testid="stExpander"] {{
background: var(--bg-surface);
border: 1px solid var(--border-subtle) !important;
border-radius: 14px;
box-shadow: var(--shadow-sm);
transition: border-color 0.15s ease;
color: var(--text-primary);
margin-bottom: 12px;
}}

[data-testid="stExpander"]:hover {{
border-color: var(--border-focus) !important;
}}

[data-testid="stExpander"] summary {{
color: var(--text-primary) !important;
font-weight: 600 !important;
font-size: 0.90rem !important;
}}

/* Sliders */
div[data-testid="stSlider"] [role="slider"] {{
background: var(--accent) !important;
box-shadow: 0 0 0 4px rgba(var(--ar), 0.20) !important;
}}

div[data-testid="stSliderThumbValue"] {{
color: var(--text-primary) !important;
font-weight: 600;
font-size: 0.82rem;
}}

/* Sidebar Styling */
section[data-testid="stSidebar"] {{
background: var(--sidebar-bg);
border-right: 1px solid var(--sidebar-border);
color: var(--text-primary);
}}

section[data-testid="stSidebar"] .block-container {{
padding-top: 1.5rem;
}}

.sidebar-brand {{
padding: 8px 4px 18px;
border-bottom: 1px solid var(--border-subtle);
margin-bottom: 18px;
}}

.sidebar-brand h2 {{
font-size: 1.25rem;
font-weight: 800;
color: var(--text-primary) !important;
letter-spacing: -0.025em;
}}

.sidebar-brand p {{
font-size: 0.78rem;
color: var(--text-secondary);
margin-top: 2px;
}}

.sidebar-profile {{
display: flex;
align-items: center;
gap: 12px;
padding: 12px 14px;
border-radius: 14px;
background: var(--bg-surface);
border: 1px solid var(--border-subtle);
margin-bottom: 14px;
box-shadow: var(--shadow-sm);
}}

.sidebar-avatar {{
width: 38px;
height: 38px;
border-radius: 50%;
display: grid;
place-items: center;
font-weight: 700;
font-size: 0.95rem;
color: #FFFFFF;
background: var(--accent);
flex: none;
}}

.sidebar-profile-info .nm {{
font-weight: 600;
font-size: 0.90rem;
color: var(--text-primary);
}}

.sidebar-profile-info .st {{
color: var(--text-secondary);
font-size: 0.76rem;
display: flex;
align-items: center;
gap: 5px;
}}

/* Responsive Breakpoints & Mobile Optimization */
@media (max-width: 960px) {{
.activity-grid {{ grid-template-columns: repeat(2, 1fr); }}
.stat-grid {{ grid-template-columns: repeat(2, 1fr); }}
.top-header {{ padding: 14px 18px; }}
.top-header-brand h1 {{ font-size: 1.25rem; }}
}}

@media (max-width: 768px) {{
.activity-grid {{ grid-template-columns: 1fr; }}
.stat-grid {{ grid-template-columns: 1fr; }}
.top-header {{ flex-direction: column; align-items: flex-start; gap: 10px; }}
.top-header-right {{ width: 100%; justify-content: flex-start; flex-wrap: wrap; }}
.verdict-box {{ flex-direction: column; align-items: flex-start; gap: 12px; }}
div[role="radiogroup"] {{ width: 100%; justify-content: space-between; }}
.emergency-armed-bar {{ flex-direction: column !important; align-items: stretch !important; gap: 12px !important; }}
.emergency-armed-bar .btn-group {{ flex-direction: column !important; width: 100% !important; }}
.emergency-armed-bar .sos-btn {{ width: 100% !important; justify-content: center !important; text-align: center; }}
.sos-btn {{ width: 100% !important; justify-content: center !important; text-align: center; }}
.prob-row {{ gap: 8px; }}
.prob-label {{ width: 110px; min-width: 105px; font-size: 0.78rem; }}
.prob-pct {{ width: 45px; font-size: 0.78rem; }}
.fall-alert-metrics {{ grid-template-columns: repeat(2, 1fr) !important; }}
.fall-alert-header {{ flex-direction: column; align-items: flex-start; }}
.sos-action-bar {{ flex-direction: column; width: 100%; }}
.hospital-card {{ flex-direction: column; align-items: flex-start; gap: 12px; }}
.hospital-actions {{ width: 100%; display: flex; flex-direction: column; gap: 8px; }}
}}

@media (max-width: 480px) {{
.stat-grid {{ grid-template-columns: 1fr; }}
.fall-alert-metrics {{ grid-template-columns: 1fr !important; }}
.card {{ padding: 16px; }}
}}
</style>
"""


def render_background_atmosphere(theme_mode: str = "light") -> str:
    """Minimal background helper. Kept clean without floating particles."""
    return ""


def inject_theme_styles(
    accent_hex: str = "#5E8B7A",
    accent2_hex: str = "#8EA8C3",
    theme_mode: str = "light",
    include_sidebar_suppression: bool = False
) -> None:
    """Inject dynamic healthcare stylesheet into Streamlit DOM."""
    css_content = generate_stylesheet(accent_hex, accent2_hex, theme_mode)
    if include_sidebar_suppression:
        css_content += (
            "<style>"
            'section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], '
            '[data-testid="collapsedControl"] { display: none !important; }'
            "</style>"
        )
    if hasattr(st, "html"):
        st.html(css_content)
    else:
        st.markdown(css_content, unsafe_allow_html=True)
