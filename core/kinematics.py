"""
core/kinematics.py
==================
Biomechanical kinematics and geometric posture evaluation engine.
Extracts joint angle telemetry, centroid trajectories, and dynamic motion vectors
from 17-point human pose landmarks to classify SITTING, STANDING, WALKING, and FALL.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

# Core 6-class clinical activity taxonomy
ACTIVITY_CLASSES = [
    "SITTING",
    "STANDING",
    "WALKING",
    "OFF_BALANCE",
    "NORMAL_ACTIVITY",
    "FALL"
]

ACTIVITY_DISPLAY_NAMES = {
    "SITTING": "Sitting",
    "STANDING": "Standing",
    "WALKING": "Walking",
    "OFF_BALANCE": "Off Balance",
    "NORMAL_ACTIVITY": "Normal Activity",
    "FALL": "Fall Detected"
}

SEQUENCE_LENGTH = 30
SEQUENCE_STRIDE = 5
REFERENCE_FPS = 25.0
FEATURE_DIMENSION = 51
MIN_KEYPOINT_CONFIDENCE = 0.18

# COCO 17-keypoint skeleton connectivity graph
COCO_TOPOLOGY = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Baseline probability vectors across 6 postures:
# [SITTING, STANDING, WALKING, OFF_BALANCE, NORMAL_ACTIVITY, FALL]
STATE_PRIORS = {
    "SITTING": np.array([0.86, 0.03, 0.03, 0.03, 0.03, 0.02], dtype=np.float32),
    "STANDING": np.array([0.03, 0.88, 0.03, 0.02, 0.02, 0.02], dtype=np.float32),
    "WALKING": np.array([0.03, 0.04, 0.85, 0.04, 0.02, 0.02], dtype=np.float32),
    "OFF_BALANCE": np.array([0.02, 0.04, 0.04, 0.84, 0.03, 0.03], dtype=np.float32),
    "NORMAL_ACTIVITY": np.array([0.03, 0.04, 0.03, 0.03, 0.84, 0.03], dtype=np.float32),
    "FALL_ACUTE": np.array([0.01, 0.01, 0.01, 0.02, 0.01, 0.94], dtype=np.float32),
    "FALL_STATIC": np.array([0.02, 0.02, 0.02, 0.03, 0.02, 0.89], dtype=np.float32),
    "DESK_SEATED": np.array([0.82, 0.08, 0.03, 0.03, 0.02, 0.02], dtype=np.float32),
    "DESK_UPRIGHT": np.array([0.06, 0.82, 0.04, 0.04, 0.02, 0.02], dtype=np.float32),
    "DESK_LOCOMOTION": np.array([0.04, 0.08, 0.78, 0.04, 0.04, 0.02], dtype=np.float32),
}


def calculate_bounding_box_iou(box_a: Optional[np.ndarray], box_b: Optional[np.ndarray]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    if box_a is None or box_b is None:
        return 0.0
    inter_x1 = max(float(box_a[0]), float(box_b[0]))
    inter_y1 = max(float(box_a[1]), float(box_b[1]))
    inter_x2 = min(float(box_a[2]), float(box_b[2]))
    inter_y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area

    return float(inter_area / union_area) if union_area > 0.0 else 0.0


def extract_normalized_features(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    bbox: np.ndarray
) -> np.ndarray:
    """
    Extract 51-dimensional normalized posture vector:
    17 keypoints x (x_rel, y_rel) relative to bounding box + 17 keypoint confidences.
    """
    kp = np.asarray(keypoints, dtype=np.float32).reshape(17, 2)
    cf = np.asarray(confidences, dtype=np.float32).reshape(17)

    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    normalized_coords = np.empty((17, 2), dtype=np.float32)
    normalized_coords[:, 0] = np.clip((kp[:, 0] - x1) / box_w, 0.0, 1.0)
    normalized_coords[:, 1] = np.clip((kp[:, 1] - y1) / box_h, 0.0, 1.0)

    return np.concatenate([normalized_coords.reshape(-1), cf]).astype(np.float32)


def compute_segment_midpoint(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    indices: Tuple[int, int],
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE
) -> Optional[np.ndarray]:
    """Compute the 2D spatial centroid between two landmark indices if sufficiently visible."""
    valid_points = [
        keypoints[idx] for idx in indices
        if confidences[idx] >= min_confidence and (keypoints[idx][0] > 0 or keypoints[idx][1] > 0)
    ]
    if not valid_points:
        valid_points = [
            keypoints[idx] for idx in indices
            if confidences[idx] >= 0.10 and (keypoints[idx][0] > 0 or keypoints[idx][1] > 0)
        ]
        if not valid_points:
            return None
    return np.mean(valid_points, axis=0)


def compute_joint_angle_degrees(point_a: np.ndarray, vertex_b: np.ndarray, point_c: np.ndarray) -> float:
    """Compute interior angle at vertex_b formed by segments (b -> a) and (b -> c) with C-speed hypot."""
    v_ba_x = float(point_a[0] - vertex_b[0])
    v_ba_y = float(point_a[1] - vertex_b[1])
    v_bc_x = float(point_c[0] - vertex_b[0])
    v_bc_y = float(point_c[1] - vertex_b[1])

    norm_ba = math.hypot(v_ba_x, v_ba_y)
    norm_bc = math.hypot(v_bc_x, v_bc_y)

    if norm_ba * norm_bc == 0.0:
        return 180.0

    dot_product = (v_ba_x * v_bc_x + v_ba_y * v_bc_y) / (norm_ba * norm_bc)
    dot_product = max(-1.0, min(1.0, dot_product))
    return float(math.degrees(math.acos(dot_product)))


class KinematicPostureEngine:
    """
    Real-time biomechanical analysis engine determining posture distribution
    from temporal sequence of 17-point pose landmark observations.
    """

    def __init__(self, desk_mode: bool = False, temporal_depth: int = 150):
        self.desk_mode = desk_mode
        self.history: Deque[Dict[str, Any]] = deque(maxlen=temporal_depth)

    def reset(self) -> None:
        """Clear temporal kinematic memory buffer."""
        self.history.clear()

    def register_frame(
        self,
        timestamp: float,
        keypoints: np.ndarray,
        confidences: np.ndarray,
        bbox: np.ndarray,
        allow_dynamic_motion: bool = True
    ) -> Optional[np.ndarray]:
        """
        Ingest current frame pose observations, calculate geometric metrics,
        and infer class probability distribution.
        """
        kp = np.asarray(keypoints, dtype=np.float32)
        cf = np.asarray(confidences, dtype=np.float32)

        bw = float(bbox[2] - bbox[0])
        bh = float(bbox[3] - bbox[1])
        center_x = float(bbox[0] + bbox[2]) / 2.0
        center_y = float(bbox[1] + bbox[3]) / 2.0

        shoulder_mid = compute_segment_midpoint(kp, cf, (5, 6))
        hip_mid = compute_segment_midpoint(kp, cf, (11, 12))
        knee_mid = compute_segment_midpoint(kp, cf, (13, 14))
        ankle_mid = compute_segment_midpoint(kp, cf, (15, 16))

        # Lateral separation between ankles relative to bounding box height
        ankle_stride_norm = None
        if cf[15] >= MIN_KEYPOINT_CONFIDENCE and cf[16] >= MIN_KEYPOINT_CONFIDENCE:
            ankle_stride_norm = abs(float(kp[15][0] - kp[16][0])) / max(bh, 1.0)

        # Torso segment length
        torso_length = None
        if shoulder_mid is not None and hip_mid is not None:
            torso_length = float(math.hypot(hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]))

        observation = {
            "timestamp": float(timestamp),
            "cx": center_x,
            "cy": center_y,
            "bw": bw,
            "bh": bh,
            "shoulder_mid": shoulder_mid,
            "hip_mid": hip_mid,
            "knee_mid": knee_mid,
            "ankle_mid": ankle_mid,
            "ankle_stride": ankle_stride_norm,
            "torso_length": torso_length,
            "torso_tilt": None
        }
        self.history.append(observation)
        return self._evaluate_kinematics(kp, cf, allow_dynamic_motion)

    def _evaluate_kinematics(self, kp: np.ndarray, cf: np.ndarray, allow_dynamic_motion: bool = True) -> Optional[np.ndarray]:
        """Execute mathematical posture and fall trajectory classification across all 6 clinical activities."""
        if not self.history:
            return None

        current = self.history[-1]
        sh = current["shoulder_mid"]
        hip = current["hip_mid"]
        kn = current["knee_mid"]
        an = current["ankle_mid"]
        torso = current["torso_length"]

        if sh is None:
            # Check if head / face keypoints (nose=0, eyes=1,2, ears=3,4) are visible:
            head_mid = compute_segment_midpoint(kp, cf, (1, 2), min_confidence=0.10)
            if head_mid is None and cf[0] >= 0.10:
                head_mid = kp[0]
            if head_mid is not None and (head_mid[0] > 0 or head_mid[1] > 0):
                # Approximate shoulder location from head position
                sh = np.array([head_mid[0], head_mid[1] + current["bh"] * 0.22], dtype=np.float32)
                current["shoulder_mid"] = sh

        if sh is None:
            # Fallback to bounding box geometry (e.g. extreme closeup or occluded torso)
            aspect_ratio = current["bw"] / max(current["bh"], 1.0)
            if aspect_ratio >= 1.25:
                return STATE_PRIORS["FALL_ACUTE"].copy()
            elif aspect_ratio <= 0.82:
                return STATE_PRIORS["STANDING"].copy()
            else:
                return STATE_PRIORS["DESK_SEATED"].copy() if self.desk_mode else STATE_PRIORS["NORMAL_ACTIVITY"].copy()

        aspect_ratio = current["bw"] / max(current["bh"], 1.0)

        # Torso inclination angle relative to gravitational vertical (0° = vertical upright, 90° = horizontal)
        torso_tilt_deg = None
        if hip is not None and torso is not None and torso > 3.0:
            trunk_vector = hip - sh
            torso_tilt_deg = math.degrees(math.atan2(abs(float(trunk_vector[0])), float(trunk_vector[1])))
            current["torso_tilt"] = torso_tilt_deg

        hips_tracked = torso_tilt_deg is not None

        # Lower-limb horizontal alignment (thigh orientation)
        femur_horizontal = None
        if hips_tracked and kn is not None:
            femur_vec = kn - hip
            femur_horizontal = abs(float(femur_vec[0])) > 0.78 * abs(float(femur_vec[1]))

        vertical_drop_ratio = 0.0
        lateral_velocity = 0.0
        zoom_rate = 0.0
        gait_oscillation = False
        tilt_growth = 0.0

        if allow_dynamic_motion and len(self.history) >= 2:
            temporal_window = [
                obs for obs in self.history
                if current["timestamp"] - obs["timestamp"] <= 1.0
            ]
            baseline = temporal_window[0]
            dt = current["timestamp"] - baseline["timestamp"]

            if dt >= 0.25:
                ref_height = max(baseline["bh"], 1.0)
                vertical_drop_ratio = (current["cy"] - baseline["cy"]) / ref_height
                lateral_velocity = abs(current["cx"] - baseline["cx"]) / ref_height / dt
                zoom_rate = abs(current["bh"] - baseline["bh"]) / ref_height / dt

                prev_tilt = baseline.get("torso_tilt")
                if torso_tilt_deg is not None and prev_tilt is not None:
                    tilt_growth = abs(torso_tilt_deg - prev_tilt)

            strides = [obs["ankle_stride"] for obs in temporal_window if obs["ankle_stride"] is not None]
            if len(strides) >= 6:
                stride_delta = max(strides) - min(strides)
                gait_oscillation = stride_delta > 0.17

        # Balance & Base-of-Support (BoS) metrics
        com_x = (0.55 * hip[0] + 0.45 * sh[0]) if (hip is not None and sh is not None) else current["cx"]
        if cf[15] >= MIN_KEYPOINT_CONFIDENCE and cf[16] >= MIN_KEYPOINT_CONFIDENCE:
            bos_min_x = min(kp[15][0], kp[16][0])
            bos_max_x = max(kp[15][0], kp[16][0])
            bos_w = max(bos_max_x - bos_min_x, current["bw"] * 0.12)
            bos_center_x = (bos_min_x + bos_max_x) / 2.0
            balance_ratio = abs(com_x - bos_center_x) / (0.5 * bos_w + 1e-5)
        else:
            balance_ratio = abs(com_x - current["cx"]) / (0.5 * current["bw"] + 1e-5)

        # Knee angle flexion and asymmetry
        left_knee_flex = compute_joint_angle_degrees(kp[11], kp[13], kp[15]) if (cf[11] >= 0.3 and cf[13] >= 0.3 and cf[15] >= 0.3) else None
        right_knee_flex = compute_joint_angle_degrees(kp[12], kp[14], kp[16]) if (cf[12] >= 0.3 and cf[14] >= 0.3 and cf[16] >= 0.3) else None
        knee_asym = abs(left_knee_flex - right_knee_flex) if (left_knee_flex is not None and right_knee_flex is not None) else 0.0

        # ---------------- 1. FALL EVALUATION (HIGHEST SAFETY PRIORITY) ----------------
        is_recumbent = hips_tracked and (
            (torso_tilt_deg > 55.0 and (femur_horizontal is True or (femur_horizontal is None and aspect_ratio > 1.05)))
            or (aspect_ratio > 1.25 and femur_horizontal is not False)
        )
        is_dynamic_fall = (
            allow_dynamic_motion
            and vertical_drop_ratio > 0.20
            and (aspect_ratio > 0.90 or (torso_tilt_deg or 0.0) > 42.0)
            and femur_horizontal is not False
        )

        if is_recumbent and is_dynamic_fall:
            return STATE_PRIORS["FALL_ACUTE"].copy()
        if is_recumbent or is_dynamic_fall:
            return STATE_PRIORS["FALL_STATIC"].copy()

        # ---------------- 2. LOWER LIMBS DETECTED: SITTING / NORMAL / OFF-BALANCE / WALKING / STANDING ----------------
        if hips_tracked and kn is not None:
            vertical_knee_hip_ratio = float((kn[1] - hip[1]) / max(torso, 1.0))
            knee_flexion_angle = compute_joint_angle_degrees(hip, kn, an) if an is not None else None

            # SITTING: compressed vertical thigh length or flexed knee angle < 135 deg with elevated hips
            is_sitting = (
                (vertical_knee_hip_ratio < 0.56 or (knee_flexion_angle is not None and knee_flexion_angle < 135.0))
                and (torso_tilt_deg is None or torso_tilt_deg < 45.0)
            )
            if is_sitting:
                return STATE_PRIORS["SITTING"].copy()

            # NORMAL ACTIVITY: Controlled forward bending / reaching / stretching with extended straight legs
            is_controlled_bend = (
                torso_tilt_deg is not None
                and 22.0 <= torso_tilt_deg <= 75.0
                and (knee_flexion_angle is None or knee_flexion_angle >= 130.0)
                and aspect_ratio < 1.05
                and vertical_drop_ratio < 0.18
                and balance_ratio <= 1.35
            )
            if is_controlled_bend:
                return STATE_PRIORS["NORMAL_ACTIVITY"].copy()

            # OFF BALANCE: Posture instability, stumbling, rapid tilt change, or CoM outside Base of Support
            is_off_balance = (
                torso_tilt_deg is not None
                and 18.0 <= torso_tilt_deg <= 55.0
                and (
                    balance_ratio > 1.20
                    or tilt_growth > 8.0
                    or (allow_dynamic_motion and lateral_velocity > 0.22 and not gait_oscillation)
                    or knee_asym > 18.0
                )
            )
            if is_off_balance:
                return STATE_PRIORS["OFF_BALANCE"].copy()

            # WALKING: dynamic gait oscillation, lateral velocity, or wide stride separation
            is_walking_gait = (
                (allow_dynamic_motion and (lateral_velocity > 0.35 or zoom_rate > 0.45 or gait_oscillation))
                or (current["ankle_stride"] is not None and current["ankle_stride"] > 0.20)
            )
            if is_walking_gait:
                return STATE_PRIORS["WALKING"].copy()

            # STANDING: default upright posture with vertical torso and extended legs
            return STATE_PRIORS["STANDING"].copy()

        # ---------------- 3. UPPER-BODY FRAMING (WEBCAM / OCCLUDED LOWER LIMBS) ----------------
        if allow_dynamic_motion and (lateral_velocity > 0.35 or zoom_rate > 0.40):
            return STATE_PRIORS["DESK_LOCOMOTION"].copy()

        # OFF BALANCE in upper body: sudden tilt acceleration or lateral sway
        if torso_tilt_deg is not None and 20.0 <= torso_tilt_deg <= 52.0 and (tilt_growth > 9.0 or (allow_dynamic_motion and lateral_velocity > 0.22)):
            return STATE_PRIORS["OFF_BALANCE"].copy()

        # NORMAL ACTIVITY in upper body: steady forward lean (reading, reaching, writing at desk)
        if torso_tilt_deg is not None and 24.0 <= torso_tilt_deg <= 65.0:
            return STATE_PRIORS["NORMAL_ACTIVITY"].copy()

        # STANDING in upper body: upright torso and vertical aspect ratio
        if hips_tracked and torso_tilt_deg is not None and torso_tilt_deg < 24.0:
            if aspect_ratio < 0.95 or (current["bh"] > current["bw"] * 1.02):
                return STATE_PRIORS["STANDING"].copy()
            return STATE_PRIORS["DESK_UPRIGHT"].copy()

        # SITTING at desk:
        if self.desk_mode and (aspect_ratio > 0.95 or (torso_tilt_deg is not None and torso_tilt_deg > 26.0)):
            return STATE_PRIORS["DESK_SEATED"].copy()

        return STATE_PRIORS["STANDING"].copy()
