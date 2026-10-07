"""
SafeFall AI - Deep Learning Classifier & Hybrid Fall Detection Engine
PyTorch Deep Neural Network + Scikit-Learn baseline model for human activity
recognition and fall detection across 5 classes.
"""

import os
import math
import joblib
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any

CLASSES = [
    "Fall Detected",
    "Off Balance",
    "Normal Activity",
    "Sitting",
    "Standing",
    "Walking"
]

class SafeFallDeepNet(nn.Module):
    """
    Deep Neural Network for Pose Feature Classification (FA-2 Architecture).
    Takes 142 input features (132 3D landmark values + 10 biomechanical features).
    """
    def __init__(self, input_dim: int = 142, num_classes: int = 6, dropout_rate: float = 0.3):
        super(SafeFallDeepNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SafeFallClassifier:
    """
    Wrapper for loading trained models, performing inference,
    and triggering emergency alerts based on posture thresholds.
    """
    def __init__(self, 
                 model_dir: Optional[str] = None,
                 use_deep_learning: bool = True):
        
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
            
        self.model_dir = model_dir
        self.use_deep_learning = use_deep_learning
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.classes = CLASSES
        self.num_classes = len(CLASSES)
        
        # Load or initialize Neural Network
        self.nn_model = SafeFallDeepNet(input_dim=142, num_classes=self.num_classes).to(self.device)
        self.rf_model = None
        self.scaler = None
        self.label_encoder = None
        
        self.nn_weights_path = os.path.join(model_dir, "safefall_nn_model.pth")
        self.rf_weights_path = os.path.join(model_dir, "safefall_rf_model.joblib")
        self.scaler_path = os.path.join(model_dir, "feature_scaler.joblib")
        self.encoder_path = os.path.join(model_dir, "label_encoder.joblib")
        
        self.load_models()

    def load_models(self) -> bool:
        """Loads saved weights from disk if available."""
        loaded_any = False
        if os.path.exists(self.nn_weights_path):
            try:
                state_dict = torch.load(self.nn_weights_path, map_location=self.device)
                self.nn_model.load_state_dict(state_dict)
                self.nn_model.eval()
                loaded_any = True
            except Exception as e:
                print(f"Warning: could not load NN weights: {e}")
                
        if os.path.exists(self.rf_weights_path):
            try:
                self.rf_model = joblib.load(self.rf_weights_path)
                loaded_any = True
            except Exception as e:
                print(f"Warning: could not load RF weights: {e}")
                
        if os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
            except Exception as e:
                print(f"Warning: could not load scaler: {e}")
                
        if os.path.exists(self.encoder_path):
            try:
                self.label_encoder = joblib.load(self.encoder_path)
                if hasattr(self.label_encoder, "classes_"):
                    self.classes = [str(c) for c in self.label_encoder.classes_]
                    self.num_classes = len(self.classes)
            except Exception as e:
                print(f"Warning: could not load label encoder: {e}")
                
        return loaded_any

    def predict(self, feature_vector: np.ndarray, heuristic_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predicts the activity class and confidence for a single 142-dim feature vector.
        Incorporates hybrid fall verification logic.
        """
        # Ensure 2D shape (1, 142)
        if feature_vector.ndim == 1:
            X = feature_vector.reshape(1, -1)
        else:
            X = feature_vector
            
        # Scale if scaler is available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        # Compute PyTorch DeepNet Probabilities
        nn_probs = None
        if self.use_deep_learning and os.path.exists(self.nn_weights_path):
            self.nn_model.eval()
            with torch.no_grad():
                tensor_in = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
                logits = self.nn_model(tensor_in)
                nn_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
        # Compute Random Forest Probabilities
        rf_probs = None
        if self.rf_model is not None:
            rf_probs = self.rf_model.predict_proba(X_scaled)[0]
            
        # Optimal Soft-Voting Ensemble (45% DeepNet + 55% Random Forest)
        if nn_probs is not None and rf_probs is not None:
            combined_probs = 0.45 * nn_probs + 0.55 * rf_probs
        elif nn_probs is not None:
            combined_probs = nn_probs
        elif rf_probs is not None:
            combined_probs = rf_probs
        else:
            combined_probs = np.ones(len(self.classes)) / len(self.classes)
            
        pred_idx = int(np.argmax(combined_probs))
        pred_class = self.classes[pred_idx]
        confidence = float(combined_probs[pred_idx])
        
        # Clinical Safety Sanity Checks & Dynamic Off-Balancer Integration
        if heuristic_metrics:
            torso_deg = heuristic_metrics.get("torso_angle_deg", 0.0)
            avg_knee_deg = heuristic_metrics.get("avg_knee_angle_deg", 170.0)
            min_knee_deg = heuristic_metrics.get("min_knee_angle_deg", 170.0)
            aspect_ratio = heuristic_metrics.get("aspect_ratio", 0.0)
            cog_y = heuristic_metrics.get("center_of_gravity_y", 0.5)
            is_bending = heuristic_metrics.get("is_controlled_bending", False)
            is_unbal = heuristic_metrics.get("is_unbalanced", False)
            is_sitting = heuristic_metrics.get("is_sitting_posture", False)
            is_floor_fall = heuristic_metrics.get("is_floor_fall", False)
            is_walking = heuristic_metrics.get("is_walking_gait", False)
            hip_clearance = heuristic_metrics.get("hip_clearance", 0.3)
            
            # 1. Emergency Fail-Safe: Clear collapse on floor (even if knees are flexed or propped up)
            # A horizontal body lying on the floor (torso >= 60°, aspect >= 0.75, CoG > 0.58) is ALWAYS a Fall Detected
            if is_floor_fall or (torso_deg >= 60.0 and aspect_ratio >= 0.75 and cog_y > 0.58) or (torso_deg >= 70.0 and cog_y > 0.60):
                pred_class = "Fall Detected"
                confidence = max(0.95, float(combined_probs[self.classes.index("Fall Detected")] if "Fall Detected" in self.classes else 0.95))
                
            # 2. Sitting Posture Stabilizer: Person seated in chair / desk (knees bent, upright/compact aspect, elevated hips)
            # Cannot be a fall or normal activity if sitting comfortably in chair with bent knees
            elif is_sitting and pred_class in ["Fall Detected", "Off Balance", "Normal Activity", "Standing"]:
                pred_class = "Sitting"
                confidence = max(0.92, float(combined_probs[self.classes.index("Sitting")] if "Sitting" in self.classes else 0.92))
                
            # 3. Normal Activity Stabilizer: Controlled Forward Bending / Reaching / Tying Shoes (supported by straight legs)
            elif is_bending and pred_class in ["Standing", "Fall Detected", "Off Balance"]:
                pred_class = "Normal Activity"
                confidence = max(0.90, float(combined_probs[self.classes.index("Normal Activity")] if "Normal Activity" in self.classes else 0.90))
                
            # 4. Straight & Lateral Walking Gait: Alternating knee bend, foot lift, or depth/horizontal stride
            elif is_walking and pred_class in ["Standing", "Normal Activity"]:
                pred_class = "Walking"
                confidence = max(0.88, float(combined_probs[self.classes.index("Walking")] if "Walking" in self.classes else 0.88))
                
            # 5. Off-Balancer Instability: Clear loss of base-of-support without ground impact
            elif is_unbal and pred_class in ["Standing", "Walking", "Normal Activity"] and torso_deg > 20.0:
                pred_class = "Off Balance"
                confidence = max(0.88, float(combined_probs[self.classes.index("Off Balance")] if "Off Balance" in self.classes else 0.88))
                
            # 6. Stationary Upright Standing: Vertical torso, extended knees, minimal stride
            elif pred_class in ["Fall Detected", "Off Balance"] and torso_deg < 18.0 and avg_knee_deg > 162.0 and aspect_ratio < 0.42:
                pred_class = "Standing"
                confidence = max(0.92, float(combined_probs[self.classes.index("Standing")] if "Standing" in self.classes else 0.90))

        is_fall = (pred_class == "Fall Detected")
        is_off_balance = (pred_class == "Off Balance")
        
        probs_dict = {}
        for i, cls_name in enumerate(self.classes):
            probs_dict[cls_name] = round(float(combined_probs[i]), 4)
            
        if pred_class in probs_dict and probs_dict[pred_class] < confidence:
            probs_dict[pred_class] = round(confidence, 4)
            
        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "is_fall": is_fall,
            "is_off_balance": is_off_balance,
            "probabilities": probs_dict
        }

    def predict_user_code(self, landmarks, last_tilt=None, hips=None, heuristic_metrics=None) -> Dict[str, Any]:
        """
        Direct detection using the user's exact kinematic activity_from_pose algorithm.
        """
        if isinstance(landmarks, np.ndarray):
            pts = [UserPoseLandmark(row[0], row[1], row[2] if len(row) > 2 else 0.0, row[3] if len(row) > 3 else 1.0) for row in landmarks]
        elif isinstance(landmarks, list):
            pts = landmarks
        else:
            return {
                "predicted_class": "Normal Activity",
                "confidence": 0.0,
                "is_fall": False,
                "is_off_balance": False,
                "tilt": 0.0,
                "probabilities": {c: 0.0 for c in self.classes}
            }
            
        label, conf, tilt = user_activity_from_pose(pts, last_tilt=last_tilt, hips=hips)
        
        USER_MAP = {
            "fall": "Fall Detected",
            "off balance": "Off Balance",
            "sitting": "Sitting",
            "standing": "Standing",
            "walking": "Walking",
            "normal": "Normal Activity"
        }
        clinical_name = USER_MAP.get(label, "Standing")
        
        # Clinical safety check for sitting at desk vs floor fall
        if heuristic_metrics:
            if heuristic_metrics.get("is_floor_fall", False) and clinical_name != "Fall Detected":
                clinical_name = "Fall Detected"
                conf = 0.95
            elif heuristic_metrics.get("is_sitting_posture", False) and clinical_name in ["Fall Detected", "Normal Activity"]:
                clinical_name = "Sitting"
                conf = 0.90
        
        probs_dict = {c: 0.05 for c in self.classes}
        probs_dict[clinical_name] = round(conf, 2)
        
        return {
            "predicted_class": clinical_name,
            "confidence": round(conf, 2),
            "is_fall": (clinical_name == "Fall Detected"),
            "is_off_balance": (clinical_name == "Off Balance"),
            "tilt": tilt,
            "user_label": label,
            "probabilities": probs_dict
        }


# =========================================================
# USER'S DETECTION FUNCTIONS (EXACT IMPLEMENTATION)
# =========================================================
class UserPoseLandmark:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)

def user_knee_bend(hip, knee, ankle):
    ax, ay = hip.x - knee.x, hip.y - knee.y
    bx, by = ankle.x - knee.x, ankle.y - knee.y
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na * nb == 0:
        return 180.0
    cos = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(cos))

def user_seen(point):
    visibility = point.visibility if point.visibility is not None else 1.0
    return visibility >= 0.5

def user_body_features(landmarks):
    def get_c(pt, attr):
        if hasattr(pt, attr):
            return getattr(pt, attr)
        elif isinstance(pt, (list, tuple, np.ndarray)):
            idx_map = {"x": 0, "y": 1, "z": 2, "visibility": 3}
            i = idx_map.get(attr, 0)
            return float(pt[i]) if len(pt) > i else 0.0
        return 0.0

    sh_lx, sh_ly = get_c(landmarks[11], "x"), get_c(landmarks[11], "y")
    sh_rx, sh_ry = get_c(landmarks[12], "x"), get_c(landmarks[12], "y")
    shoulder_x = (sh_lx + sh_rx) / 2.0
    shoulder_y = (sh_ly + sh_ry) / 2.0
    
    hip_lx, hip_ly = get_c(landmarks[23], "x"), get_c(landmarks[23], "y")
    hip_rx, hip_ry = get_c(landmarks[24], "x"), get_c(landmarks[24], "y")
    hip_x = (hip_lx + hip_rx) / 2.0
    hip_y = (hip_ly + hip_ry) / 2.0
    
    left_bend = user_knee_bend(landmarks[23], landmarks[25], landmarks[27])
    right_bend = user_knee_bend(landmarks[24], landmarks[26], landmarks[28])
    bend = (left_bend + right_bend) / 2.0
    min_bend = min(left_bend, right_bend)
    knee_asym = abs(left_bend - right_bend)
    
    ank_lx, ank_ly = get_c(landmarks[27], "x"), get_c(landmarks[27], "y")
    ank_rx, ank_ry = get_c(landmarks[28], "x"), get_c(landmarks[28], "y")
    ankle_gap = abs(ank_lx - ank_rx)
    hip_width = abs(hip_lx - hip_rx)
    foot_lift = abs(ank_ly - ank_ry)
    
    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y
    tilt_2d = abs(math.degrees(math.atan2(dx, dy))) if (dx or dy) else 0.0
    
    sh_lz, sh_rz = get_c(landmarks[11], "z"), get_c(landmarks[12], "z")
    hip_lz, hip_rz = get_c(landmarks[23], "z"), get_c(landmarks[24], "z")
    dz = ((hip_lz + hip_rz) / 2.0) - ((sh_lz + sh_rz) / 2.0)
    norm_3d = math.hypot(dx, dy, dz) + 1e-7
    cos_3d = max(-1.0, min(1.0, dy / norm_3d))
    tilt_3d = math.degrees(math.acos(cos_3d))
    tilt = max(tilt_2d, tilt_3d)
    
    xs = [get_c(lm, "x") for lm in landmarks]
    ys = [get_c(lm, "y") for lm in landmarks]
    w = max(xs) - min(xs) + 1e-5
    h = max(ys) - min(ys) + 1e-5
    aspect_ratio = w / h
    cog_y = hip_y
    
    bos_min_x = min(ank_lx, ank_rx)
    bos_max_x = max(ank_lx, ank_rx)
    bos_width = max(0.06, bos_max_x - bos_min_x)
    bos_center_x = (bos_min_x + bos_max_x) / 2.0
    com_x = 0.55 * hip_x + 0.45 * shoulder_x
    balance_deviation = abs(com_x - bos_center_x)
    balance_ratio = balance_deviation / (0.5 * bos_width + 1e-5)
    
    ground_y = max(ys)
    hip_clearance = ground_y - hip_y

    return {
        "tilt": tilt,
        "bend": bend,
        "min_bend": min_bend,
        "knee_asym": knee_asym,
        "ankle_gap": ankle_gap,
        "hip_width": hip_width,
        "foot_lift": foot_lift,
        "hip_x": hip_x,
        "hip_y": hip_y,
        "aspect_ratio": aspect_ratio,
        "cog_y": cog_y,
        "balance_ratio": balance_ratio,
        "hip_clearance": hip_clearance
    }

def user_hip_is_moving(hips, hip_x, hip_y):
    hips.append((hip_x, hip_y))
    if len(hips) > 10:
        del hips[0]
    if len(hips) < 6:
        return False
    path = 0.0
    for start, end in zip(hips, hips[1:]):
        path += math.hypot(end[0] - start[0], end[1] - start[1])
    net = math.hypot(hips[-1][0] - hips[0][0], hips[-1][1] - hips[0][1])
    return path > 0.10 and net > path * 0.45

def user_thigh_span(hip, knee):
    return math.hypot(knee.x - hip.x, knee.y - hip.y)

def user_knees_toward_camera(landmarks):
    hip_z = (getattr(landmarks[23], "z", 0.0) + getattr(landmarks[24], "z", 0.0)) / 2
    knee_z = (getattr(landmarks[25], "z", 0.0) + getattr(landmarks[26], "z", 0.0)) / 2
    return hip_z - knee_z > 0.08

def user_chair_sit(landmarks):
    def readable(point):
        visibility = point.visibility if point.visibility is not None else 1.0
        return visibility >= 0.35

    if not readable(landmarks[25]) or not readable(landmarks[26]):
        return False
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2
    torso = hip_y - shoulder_y
    if torso < 0.05:
        return False
    left_drop = landmarks[25].y - hip_y
    right_drop = landmarks[26].y - hip_y
    knees_up = -0.02 < left_drop < torso * 0.60 and -0.02 < right_drop < torso * 0.60
    return knees_up or user_knees_toward_camera(landmarks)

def user_activity_from_pose(landmarks, last_tilt=None, hips=None):
    f = user_body_features(landmarks)
    tilt = f["tilt"]
    bend = f["bend"]
    min_bend = f["min_bend"]
    knee_asym = f["knee_asym"]
    ankle_gap = f["ankle_gap"]
    hip_width = f["hip_width"]
    foot_lift = f["foot_lift"]
    aspect_ratio = f["aspect_ratio"]
    cog_y = f["cog_y"]
    balance_ratio = f["balance_ratio"]
    hip_clearance = f["hip_clearance"]
    hip_x, hip_y = f["hip_x"], f["hip_y"]

    growing = last_tilt is not None and (tilt - last_tilt) > 12 and tilt > 18
    
    # 1. Floor Collapse Fail-Safe (Clear fall on ground)
    is_floor_collapse = (
        (tilt >= 60.0 and aspect_ratio >= 0.75 and cog_y > 0.58) or
        (tilt >= 70.0 and (aspect_ratio >= 0.65 or cog_y > 0.60)) or
        (aspect_ratio >= 0.95 and tilt > 45.0 and cog_y > 0.55)
    )
    
    # 2. Sitting Indicators (Must have flexed knees and elevated hips)
    is_seated_angles = (bend < 138.0 or min_bend < 132.0) and bend < 150.0
    is_sit = is_seated_angles and (aspect_ratio < 0.78 and cog_y < 0.85) and tilt < 58.0 and not is_floor_collapse
    
    # 3. Walking Indicators (Dynamic stride, knee asymmetry, foot lift)
    has_stride = (ankle_gap > max(0.06, hip_width * 2.2))
    has_knee_stride = (knee_asym > 15.0)
    has_foot_lift = (foot_lift > 0.035)
    moving = hips is not None and user_hip_is_moving(hips, hip_x, hip_y)
    is_walking = (has_stride or has_knee_stride or has_foot_lift or moving) and tilt < 22.0 and not is_sit and not is_floor_collapse
    
    # 4. Controlled Bending (Normal Activity - requires supporting extended legs and counterbalanced feet)
    is_controlled_bend = (
        (24.0 <= tilt <= 75.0) and
        (aspect_ratio < 0.90) and
        (hip_clearance > 0.16) and
        (balance_ratio <= 1.25) and
        (bend >= 140.0 and min_bend >= 130.0) and
        not is_floor_collapse
    )

    # 5. Off-Balance Instability (Stumble / Pre-Fall)
    is_off_balance = (
        (tilt >= 20.0 or (tilt >= 15.0 and balance_ratio > 1.25) or growing) and
        not is_sit and
        not is_floor_collapse and
        not is_controlled_bend and
        not is_walking and
        aspect_ratio < 1.10
    )

    # Classification Hierarchy
    if is_floor_collapse or tilt > 65.0:
        label = "fall"
        confidence = min(0.99, max(0.90, tilt / 70.0))
    elif is_sit:
        label = "sitting"
        confidence = 0.92
    elif is_controlled_bend:
        label = "normal"
        confidence = 0.88
    elif is_walking:
        label = "walking"
        confidence = 0.88
    elif is_off_balance:
        label = "off balance"
        confidence = min(0.95, max(0.85, max(tilt, 20) / 40.0))
    elif tilt < 20.0 and bend > 155.0:
        label = "standing"
        confidence = 0.90
    else:
        label = "standing"
        confidence = 0.85

    return label, round(float(confidence), 2), tilt


