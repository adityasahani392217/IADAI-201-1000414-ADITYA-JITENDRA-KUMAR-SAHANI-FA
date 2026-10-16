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


COCO_TO_MP33_INDICES = {
    0: 0,   # nose -> nose
    1: 2,   # left_eye -> left_eye
    2: 5,   # right_eye -> right_eye
    3: 7,   # left_ear -> left_ear
    4: 8,   # right_ear -> right_ear
    5: 11,  # left_shoulder -> left_shoulder
    6: 12,  # right_shoulder -> right_shoulder
    7: 13,  # left_elbow -> left_elbow
    8: 14,  # right_elbow -> right_elbow
    9: 15,  # left_wrist -> left_wrist
    10: 16, # right_wrist -> right_wrist
    11: 23, # left_hip -> left_hip
    12: 24, # right_hip -> right_hip
    13: 25, # left_knee -> left_knee
    14: 26, # right_knee -> right_knee
    15: 27, # left_ankle -> left_ankle
    16: 28  # right_ankle -> right_ankle
}


def extract_142_features_from_coco(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    bbox: np.ndarray,
    img_shape: Tuple[int, int] = (480, 640)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract 142-dimensional feature vector and clinical heuristic metrics
    from 17 COCO landmarks + bounding box for SafeFallClassifier ensemble inference.
    """
    h, w = img_shape[:2]
    kp = np.asarray(keypoints, dtype=np.float32).reshape(17, 2)
    cf = np.asarray(confidences, dtype=np.float32).reshape(17)

    # 1. Map 17 COCO keypoints into 33-point MediaPipe topology
    lm33 = np.zeros((33, 4), dtype=np.float32)
    for c_idx, mp_idx in COCO_TO_MP33_INDICES.items():
        lm33[mp_idx] = [kp[c_idx, 0] / max(float(w), 1.0), kp[c_idx, 1] / max(float(h), 1.0), 0.0, cf[c_idx]]

    # Interpolate facial, hand, and foot auxiliary landmarks
    lm33[1] = lm33[2]
    lm33[3] = lm33[2]
    lm33[4] = lm33[5]
    lm33[6] = lm33[5]
    lm33[9] = (lm33[0] + lm33[11]) / 2.0
    lm33[10] = (lm33[0] + lm33[12]) / 2.0
    for k in (17, 19, 21):
        lm33[k] = lm33[15]
    for k in (18, 20, 22):
        lm33[k] = lm33[16]
    for k in (29, 31):
        lm33[k] = lm33[27]
    for k in (30, 32):
        lm33[k] = lm33[28]

    raw_flat = lm33.flatten()  # 132 features

    # Extract anatomical points
    nose = lm33[0]
    l_sh, r_sh = lm33[11], lm33[12]
    mid_sh = (l_sh + r_sh) / 2.0
    l_hip, r_hip = lm33[23], lm33[24]
    mid_hip = (l_hip + r_hip) / 2.0
    l_kn, r_kn = lm33[25], lm33[26]
    l_an, r_an = lm33[27], lm33[28]
    mid_an = (l_an + r_an) / 2.0

    # 1. Omnidirectional Torso Inclination (Depth 3D + 2D Planar)
    torso_3d = mid_sh[:3] - mid_hip[:3]
    norm_3d = float(np.linalg.norm(torso_3d) + 1e-7)
    cos_tilt_3d = float(np.clip((-torso_3d[1]) / norm_3d, -1.0, 1.0))
    torso_angle_3d = float(math.degrees(math.acos(cos_tilt_3d)))

    dx_2d = abs(float(mid_sh[0] - mid_hip[0]))
    dy_2d = abs(float(mid_sh[1] - mid_hip[1])) + 1e-7
    torso_angle_2d = float(math.degrees(math.atan2(dx_2d, dy_2d)))
    torso_angle = float(max(torso_angle_2d, torso_angle_3d))

    # 2. Bounding box aspect ratio
    bx1, by1, bx2, by2 = [float(v) for v in bbox]
    bbox_w = max(1e-5, (bx2 - bx1) / max(float(w), 1.0))
    bbox_h = max(1e-5, (by2 - by1) / max(float(h), 1.0))
    aspect_ratio = float((bx2 - bx1) / max(1.0, (by2 - by1)))

    # 3. Center of gravity elevation (mid-hip Y)
    cog_y = float(mid_hip[1])

    # 4. Head to hip vertical difference
    head_hip_dy = float(nose[1] - mid_hip[1])

    # 5. Knee and hip flexion angles
    def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
        return float(math.degrees(math.acos(max(-1.0, min(1.0, float(cos_val))))))

    l_knee_ang = _joint_angle(l_hip, l_kn, l_an)
    r_knee_ang = _joint_angle(r_hip, r_kn, r_an)
    avg_knee_ang = float((l_knee_ang + r_knee_ang) / 2.0)
    min_knee_ang = float(min(l_knee_ang, r_knee_ang))
    knee_asym = abs(float(l_knee_ang - r_knee_ang))

    l_hip_ang = _joint_angle(l_sh, l_hip, l_kn)
    r_hip_ang = _joint_angle(r_sh, r_hip, r_kn)
    avg_hip_ang = float((l_hip_ang + r_hip_ang) / 2.0)

    # 6. Dimensions and gait strides
    body_height_norm = float(np.linalg.norm(mid_sh[:2] - mid_an[:2]))
    shoulder_width_norm = float(np.linalg.norm(l_sh[:2] - r_sh[:2]))
    shoulder_height_ratio = float(shoulder_width_norm / (body_height_norm + 1e-5))
    ankle_stride = float(np.linalg.norm(l_an[:2] - r_an[:2]))
    ankle_dy = abs(float(l_an[1] - r_an[1]))

    # 7. Omnidirectional heuristic fall score
    lat_score = 0.0
    if torso_angle > 45.0 or aspect_ratio > 0.90:
        lat_score = min(1.0, max(0.0, (torso_angle - 25.0) / 45.0) * 0.5 + max(0.0, (aspect_ratio - 0.7) / 0.8) * 0.5)

    persp_score = 0.0
    if (torso_angle_3d > 45.0 or (head_hip_dy >= -0.12 and torso_angle > 35.0)) and cog_y > 0.55:
        persp_score = min(1.0, (torso_angle_3d / 65.0) * 0.6 + (cog_y / 0.75) * 0.4)

    diag_score = 0.0
    if torso_angle > 40.0 and aspect_ratio > 0.65 and cog_y > 0.52:
        diag_score = min(1.0, (torso_angle / 60.0) * 0.6 + (aspect_ratio / 1.1) * 0.4)

    collapse_score = 0.0
    if cog_y > 0.55 and (avg_knee_ang < 75.0 or avg_hip_ang < 75.0 or (torso_angle > 45.0 and aspect_ratio > 0.55)):
        collapse_score = min(1.0, 0.60 + (cog_y / 0.75) * 0.25 + (1.0 - min(avg_knee_ang, 90.0) / 90.0) * 0.20)

    is_standing_geometry = (torso_angle < 20.0 and avg_knee_ang > 162.0 and aspect_ratio < 0.45)
    is_upright_bending = (torso_angle < 45.0 and avg_knee_ang > 155.0 and aspect_ratio < 0.50 and head_hip_dy < -0.13)
    if is_standing_geometry or is_upright_bending:
        heuristic_fall_score = 0.05
    else:
        heuristic_fall_score = float(max(lat_score, persp_score, diag_score, collapse_score))

    bio_features = np.array([
        torso_angle,
        aspect_ratio,
        cog_y,
        head_hip_dy,
        avg_knee_ang,
        avg_hip_ang,
        body_height_norm,
        shoulder_width_norm,
        shoulder_height_ratio,
        heuristic_fall_score
    ], dtype=np.float32)

    vec142 = np.concatenate([raw_flat, bio_features]).astype(np.float32)

    # 8. Dynamic Off-Balancer Kinematic Stability Analysis
    foot_xs = [l_an[0], r_an[0], lm33[29][0], lm33[30][0], lm33[31][0], lm33[32][0]]
    foot_ys = [l_an[1], r_an[1], lm33[29][1], lm33[30][1], lm33[31][1], lm33[32][1]]
    min_bos_x, max_bos_x = min(foot_xs), max(foot_xs)
    bos_w = max(0.06, max_bos_x - min_bos_x)
    bos_cx = (min_bos_x + max_bos_x) / 2.0
    ground_y = max(foot_ys)
    com_x = 0.55 * mid_hip[0] + 0.45 * mid_sh[0]
    balance_deviation = abs(com_x - bos_cx)
    balance_ratio = float(balance_deviation / (0.5 * bos_w + 1e-5))
    hip_clearance = float(ground_y - mid_hip[1])

    if hip_clearance < 0.12 and aspect_ratio > 1.10:
        stability_score = max(5.0, hip_clearance * 80.0)
    else:
        instability_penalty = max(0.0, (balance_ratio - 0.75) * 45.0) + (ankle_dy * 70.0)
        stability_score = max(10.0, min(100.0, 100.0 - instability_penalty))

    is_sitting_posture = bool(
        (avg_knee_ang <= 138.0 or min_knee_ang <= 130.0)
        and (aspect_ratio < 0.75)
        and (torso_angle < 60.0)
        and (cog_y < 0.85)
    )
    is_floor_fall = bool(
        (torso_angle >= 60.0 and aspect_ratio >= 0.75 and cog_y > 0.58)
        or (torso_angle >= 70.0 and (aspect_ratio >= 0.65 or cog_y > 0.60))
        or (aspect_ratio >= 0.95 and torso_angle > 45.0 and cog_y > 0.55)
    )
    is_controlled_bending = bool(
        (24.0 <= torso_angle <= 75.0)
        and (aspect_ratio < 0.90)
        and (hip_clearance > 0.16)
        and (stability_score >= 48.0)
        and (cog_y < 0.68)
        and (avg_knee_ang >= 142.0 and min_knee_ang >= 132.0)
    )
    has_stride = (ankle_stride > 0.13)
    has_knee_stride = (knee_asym > 18.0 and avg_knee_ang > 125.0)
    has_foot_lift = (ankle_dy > 0.035 and avg_knee_ang > 125.0)
    is_walking_gait = bool((torso_angle < 25.0) and not is_sitting_posture and not is_floor_fall and (has_stride or has_knee_stride or has_foot_lift))
    is_unbalanced = bool((torso_angle >= 20.0) and not is_sitting_posture and not is_floor_fall and (balance_ratio > 1.25 or ankle_dy > 0.08) and (stability_score < 45.0) and (aspect_ratio < 1.10))

    metrics = {
        "torso_angle_deg": round(torso_angle, 1),
        "torso_angle_3d_deg": round(torso_angle_3d, 1),
        "aspect_ratio": round(aspect_ratio, 2),
        "center_of_gravity_y": round(cog_y, 3),
        "avg_knee_angle_deg": round(avg_knee_ang, 1),
        "avg_hip_angle_deg": round(avg_hip_ang, 1),
        "knee_asymmetry_deg": round(knee_asym, 1),
        "min_knee_angle_deg": round(min_knee_ang, 1),
        "ankle_stride": round(ankle_stride, 3),
        "ankle_dy": round(ankle_dy, 3),
        "vertical_span": round(bbox_h, 3),
        "head_hip_dy": round(head_hip_dy, 3),
        "heuristic_fall_score": round(heuristic_fall_score, 2),
        "stability_score": round(float(stability_score), 1),
        "balance_ratio": round(float(balance_ratio), 2),
        "hip_clearance": round(float(hip_clearance), 3),
        "bos_width": round(float(bos_w), 3),
        "is_sitting_posture": is_sitting_posture,
        "is_floor_fall": is_floor_fall,
        "is_controlled_bending": is_controlled_bending,
        "is_walking_gait": is_walking_gait,
        "is_unbalanced": is_unbalanced
    }
    return vec142, metrics


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
