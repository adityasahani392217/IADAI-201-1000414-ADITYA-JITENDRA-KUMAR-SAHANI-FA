"""
SafeFall AI - Emergency Alert Dispatch & Incident Logging Manager
Handles real-time emergency triggers, simulated SMS/Phone notification,
incident snapshot logging, and audio alarm generation.
"""

import os
import csv
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

class AlertManager:
    """
    Manages automated caregiver alerts, SMS notifications, and incident log records.
    """
    def __init__(self, log_file: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_file = os.path.join(base_dir, "data", "incident_logs.csv")

        if log_file is None:
            self.log_file = default_file
        else:
            p = str(log_file)
            # If a directory was mistakenly passed (such as falls_dir or data dir), use valid file path
            if os.path.isdir(p) or p.endswith(os.sep) or p.endswith("/") or p.endswith("\\"):
                if os.path.basename(os.path.normpath(p)) in ("falls", "outputs"):
                    self.log_file = default_file
                else:
                    self.log_file = os.path.join(p, "incident_logs.csv")
            else:
                self.log_file = p

        self.emergency_contacts = [
            {"name": "Dr. Sarah Mitchell", "role": "Primary Care Physician", "phone": "+1 (555) 019-2834", "status": "Active"},
            {"name": "Nurse Emily Roberts", "role": "On-Duty Caregiver", "phone": "+1 (555) 014-9921", "status": "Active"},
            {"name": "Central Hospital Emergency Response", "role": "Emergency Dispatch", "phone": "911 / EMS", "status": "24/7 Monitored"}
        ]
        self.last_alert_time = 0
        self.cooldown_seconds = 5  # Prevent repeated alert flooding
        self._ensure_log_file()

    def _ensure_log_file(self):
        try:
            if os.path.isdir(self.log_file):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.log_file = os.path.join(base_dir, "data", "incident_logs.csv")
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            if not os.path.exists(self.log_file) or os.path.getsize(self.log_file) == 0:
                with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Incident_ID", "Timestamp", "Activity", "Confidence",
                        "Torso_Angle_Deg", "Aspect_Ratio", "Emergency_Alert_Triggered",
                        "Dispatch_Status", "Notes"
                    ])
        except Exception:
            pass

    def trigger_fall_alert(self, 
                           confidence: float = 0.95, 
                           metrics: Optional[Dict[str, Any]] = None, 
                           patient_id: str = "PATIENT-8042",
                           room_loc: str = "Home_01 Living Room",
                           **kwargs) -> Dict[str, Any]:
        """
        Triggers emergency notification sequence and appends to incident log.
        Supports both positional and keyword variations (fall_confidence, room_name, sensor_metadata).
        """
        if "fall_confidence" in kwargs:
            confidence = float(kwargs["fall_confidence"])
        if "room_name" in kwargs:
            room_loc = str(kwargs["room_name"])
        if metrics is None:
            metrics = kwargs.get("sensor_metadata", {})
        if not isinstance(metrics, dict):
            metrics = {}

        now_ts = time.time()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        incident_id = f"FALL-{int(now_ts)}"
        
        torso_angle = metrics.get("torso_angle_deg", 0.0)
        aspect_ratio = metrics.get("aspect_ratio", 0.0)
        
        # Dispatch SMS summary
        sms_message = (
            f"[CRITICAL MEDICAL ALERT] Fall Detected for {patient_id} at {room_loc}! "
            f"Confidence: {confidence*100:.1f}%. Torso Angle: {torso_angle} deg. "
            f"Time: {now_str}. Dispatching emergency caregiver team."
        )
        
        # Log to file safely
        try:
            self._ensure_log_file()
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    incident_id, now_str, "Fall Detected", f"{confidence:.4f}",
                    torso_angle, aspect_ratio, "YES", "DISPATCHED_TO_CAREGIVERS",
                    f"Room: {room_loc} | Patient: {patient_id}"
                ])
        except Exception:
            pass
            
        self.last_alert_time = now_ts
        
        return {
            "incident_id": incident_id,
            "timestamp": now_str,
            "patient_id": patient_id,
            "location": room_loc,
            "confidence": confidence,
            "sms_payload": sms_message,
            "contacts_notified": [c["name"] for c in self.emergency_contacts],
            "status": "DISPATCH_SENT",
            "severity": "CRITICAL"
        }

    def trigger_off_balance_warning(self,
                                    confidence: float = 0.85,
                                    metrics: Optional[Dict[str, Any]] = None,
                                    patient_id: str = "PATIENT-8042",
                                    room_loc: str = "Home_01 Living Room",
                                    **kwargs) -> Dict[str, Any]:
        """
        Triggers early warning notification for loss of balance / stumbling.
        """
        if "warning_confidence" in kwargs:
            confidence = float(kwargs["warning_confidence"])
        if "room_name" in kwargs:
            room_loc = str(kwargs["room_name"])
        if metrics is None:
            metrics = kwargs.get("sensor_metadata", {})
        if not isinstance(metrics, dict):
            metrics = {}

        now_ts = time.time()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        incident_id = f"WARN-{int(now_ts)}"
        
        torso_angle = metrics.get("torso_angle_deg", 0.0)
        aspect_ratio = metrics.get("aspect_ratio", 0.0)
        
        sms_message = (
            f"[PRE-FALL WARNING] Posture instability detected for {patient_id} at {room_loc}! "
            f"Confidence: {confidence*100:.1f}%. Torso Angle: {torso_angle} deg. "
            f"Time: {now_str}. Proactive caregiver assistance advised."
        )
        
        try:
            self._ensure_log_file()
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    incident_id, now_str, "Off Balance", f"{confidence:.4f}",
                    torso_angle, aspect_ratio, "WARNING", "ASSISTANCE_ADVISED",
                    f"Room: {room_loc} | Patient: {patient_id}"
                ])
        except Exception:
            pass
            
        self.last_alert_time = now_ts
        
        return {
            "incident_id": incident_id,
            "timestamp": now_str,
            "patient_id": patient_id,
            "location": room_loc,
            "confidence": confidence,
            "sms_payload": sms_message,
            "contacts_notified": [c["name"] for c in self.emergency_contacts],
            "status": "WARNING_SENT",
            "severity": "WARNING"
        }

    def get_recent_incidents(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Returns the most recent logged incidents safely without throwing file errors."""
        try:
            if os.path.isdir(self.log_file):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.log_file = os.path.join(base_dir, "data", "incident_logs.csv")

            if not os.path.exists(self.log_file) or os.path.isdir(self.log_file):
                return []
            
            records = []
            with open(self.log_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
            return list(reversed(records))[:limit]
        except Exception:
            return []
