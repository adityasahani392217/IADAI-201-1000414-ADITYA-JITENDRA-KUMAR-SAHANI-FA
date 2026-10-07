"""
SafeFall AI - Visualizer & HUD Overlay Utility
Provides medical monitoring HUD graphics, bounding boxes, posture angles,
and fall emergency banners over video frames and static images.
100% strictly ASCII-compliant to ensure zero question mark glyphs in OpenCV.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

def draw_hud(frame: np.ndarray, 
             activity: str, 
             confidence: float, 
             metrics: Dict[str, Any],
             fps: float = 0.0) -> np.ndarray:
    """
    Renders clean, professional healthcare monitoring HUD banner and metrics panel.
    Strictly ASCII characters only to guarantee no question mark glyphs.
    """
    h, w, _ = frame.shape
    out = frame.copy()
    
    is_none = "no person" in activity.lower()
    is_fall = ("fall" in activity.lower()) and not is_none
    is_off_balance = ("off balance" in activity.lower() or "off-balance" in activity.lower())
    
    # 1. Top Banner
    banner_height = 60
    if is_fall:
        banner_color = (0, 0, 180)
    elif is_off_balance:
        banner_color = (0, 80, 180) # Amber-orange alert
    elif is_none:
        banner_color = (30, 36, 45)
    else:
        banner_color = (25, 30, 40)
    
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), banner_color, -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
    
    cv2.putText(out, "SafeFall AI | Patient Monitoring Feed", (15, 22), 
                cv2.FONT_HERSHEY_DUPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    
    if is_none:
        status_color = (200, 215, 225)
        status_text = "Status: NO PERSON DETECTED"
    elif is_fall:
        status_color = (60, 60, 255)
        status_text = f"Status: {activity.upper()}"
    elif is_off_balance:
        status_color = (0, 180, 255) # Bright amber/orange
        status_text = f"Status: WARNING - {activity.upper()}"
    else:
        status_color = (60, 255, 130)
        status_text = f"Status: {activity.upper()}"
        
    cv2.putText(out, status_text, (15, 48), 
                cv2.FONT_HERSHEY_DUPLEX, 0.70, status_color, 2, cv2.LINE_AA)
    
    if fps > 0:
        cv2.putText(out, f"FPS: {fps:.1f}", (w - 115, 35), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 240, 255), 1, cv2.LINE_AA)
        
    # 2. Bottom-Left Biomechanical Telemetry Panel
    if is_none:
        panel_w, panel_h = 240, 52
        px, py = 15, h - panel_h - 15
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        cv2.rectangle(out, (px, py), (px + panel_w, py + panel_h), (80, 80, 90), 1)
        cv2.putText(out, "PATIENT AREA: CLEAR", (px + 10, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(out, "Scanning for subject entry...", (px + 10, py + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    else:
        panel_w, panel_h = 245, 114
        px, py = 15, h - panel_h - 15
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        cv2.rectangle(out, (px, py), (px + panel_w, py + panel_h), (80, 80, 90), 1)
        
        cv2.putText(out, "BIOMECHANICAL METRICS", (px + 10, py + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA)
        
        torso = metrics.get("torso_angle_deg", 0.0)
        aspect = metrics.get("aspect_ratio", 0.0)
        knee_ang = metrics.get("avg_knee_angle_deg", 0.0)
        stab = float(metrics.get("stability_score", 100.0))
        b_status = str(metrics.get("balance_status", "Stable"))
        stab_color = (0, 255, 120) if stab >= 50.0 else ((0, 165, 255) if stab >= 35.0 else (60, 60, 255))
        
        cv2.putText(out, f"Torso Tilt: {torso:.1f} deg", (px + 10, py + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"Aspect Ratio: {aspect:.2f}", (px + 10, py + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"Knee Angle: {knee_ang:.1f} deg", (px + 10, py + 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"Off-Balancer: {stab:.0f}% ({b_status[:10]})", (px + 10, py + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, stab_color, 1, cv2.LINE_AA)
        
        # 3. Bounding Box
        bbox = metrics.get("bbox")
        if bbox:
            bx, by, bw, bh = bbox
            ix = int(bx * w)
            iy = int(by * h)
            iw = int(bw * w)
            ih = int(bh * h)
            if is_fall:
                box_color = (0, 0, 255) # Red
            elif is_off_balance:
                box_color = (0, 165, 255) # Amber
            else:
                box_color = (0, 255, 120) # Green
            cv2.rectangle(out, (ix, iy), (ix + iw, iy + ih), box_color, 2)
            cv2.putText(out, f"{activity}", (ix, max(20, iy - 8)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, box_color, 1, cv2.LINE_AA)
        
    return out
