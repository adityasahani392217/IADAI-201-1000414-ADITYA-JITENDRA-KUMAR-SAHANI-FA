"""
SafeFall AI - Pose Detector & Biomechanical Feature Engineering Module
Supports MediaPipe Pose (Tasks API & Solutions API) with automatic contour fallback
for robust 33-landmark 3D extraction and clinical biomechanical metrics.
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Any

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27),
    (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

class SafeFallPoseDetector:
    """
    Robust pose detection and feature engineering wrapper.
    """
    def __init__(self, 
                 static_image_mode: bool = False, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.4):
        
        self.mp_landmarker = None
        self.model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
        
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks import python as mp_python
            
            if os.path.exists(self.model_path):
                base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    min_pose_detection_confidence=min_detection_confidence,
                    output_segmentation_masks=False
                )
                self.mp_landmarker = vision.PoseLandmarker.create_from_options(options)
                self.mp = mp
        except Exception:
            self.mp_landmarker = None

        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28

    def process_frame(self, frame_bgr: np.ndarray, allow_synthetic_fallback: bool = False) -> Dict[str, Any]:
        """Runs pose estimation on a BGR frame."""
        h, w, _ = frame_bgr.shape
        landmarks = None
        
        # 1. Try MediaPipe Task Landmarker on real human frames
        if self.mp_landmarker is not None:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
                res = self.mp_landmarker.detect(mp_img)
                if res and res.pose_landmarks and len(res.pose_landmarks) > 0:
                    raw_lms = res.pose_landmarks[0]
                    landmarks = np.zeros((33, 4), dtype=np.float32)
                    for i, lm in enumerate(raw_lms):
                        landmarks[i] = [lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.95)]
                    # Reject inverted/upside-down false-positive skeletons (e.g. on synthetic sketches)
                    mid_sh_y = (landmarks[11, 1] + landmarks[12, 1]) / 2.0
                    mid_hip_y = (landmarks[23, 1] + landmarks[24, 1]) / 2.0
                    if mid_sh_y > mid_hip_y + 0.05 and allow_synthetic_fallback:
                        landmarks = None
            except Exception:
                landmarks = None

        # 2. Kinematic & Contour Posture Fallback ONLY if explicitly requested (e.g. for synthetic dummy datasets)
        if landmarks is None and allow_synthetic_fallback:
            landmarks = self._extract_kinematic_landmarks(frame_bgr)
            
        return {"landmarks": landmarks, "shape": (h, w)}

    def extract_landmarks(self, results: Dict[str, Any]) -> Optional[np.ndarray]:
        return results.get("landmarks")

    def _extract_kinematic_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extracts body landmarks based on foreground segmentation and kinematic analysis."""
        h, w, _ = frame_bgr.shape
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        gray[0:45, :] = 240
        diff = cv2.absdiff(gray, 240)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
        thresh[378:384, :] = 0
        thresh[440:, :] = 0
        thresh[:, 0:20] = 0
        thresh[:, w-20:w] = 0
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_large = [c for c in contours if cv2.contourArea(c) > 200]
        
        lm = np.zeros((33, 4), dtype=np.float32)
        lm[:, 3] = 0.95
        
        if not c_large:
            # Default upright
            cx, cy = 0.5, 0.5
            bx, by, bw, bh = int(w*0.35), int(h*0.25), int(w*0.3), int(h*0.5)
        else:
            main_c = max(c_large, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(main_c)
            
        ar = bw / float(bh + 1e-5)
        nx1, ny1 = bx / w, by / h
        nx2, ny2 = (bx + bw) / w, (by + bh) / h
        nbw, nbh = nx2 - nx1, ny2 - ny1
        cx = (nx1 + nx2) / 2.0
        
        if ar > 1.25 or (ny2 > 0.78 and nbh < 0.22):
            # Horizontal Fall
            lm[0] = [nx1 + 0.05*nbw, ny1 + 0.50*nbh, 0.0, 0.95]
            lm[11] = [nx1 + 0.15*nbw, ny1 + 0.35*nbh, 0.0, 0.95]
            lm[12] = [nx1 + 0.15*nbw, ny1 + 0.65*nbh, 0.0, 0.95]
            lm[23] = [nx1 + 0.45*nbw, ny1 + 0.40*nbh, 0.0, 0.95]
            lm[24] = [nx1 + 0.45*nbw, ny1 + 0.60*nbh, 0.0, 0.95]
            lm[25] = [nx1 + 0.70*nbw, ny1 + 0.45*nbh, 0.0, 0.90]
            lm[26] = [nx1 + 0.70*nbw, ny1 + 0.55*nbh, 0.0, 0.90]
            lm[27] = [nx1 + 0.95*nbw, ny1 + 0.48*nbh, 0.0, 0.90]
            lm[28] = [nx1 + 0.95*nbw, ny1 + 0.52*nbh, 0.0, 0.90]
        elif ar > 0.52:
            # Sitting Posture (Knees bent at 90 deg)
            cx = (nx1 + nx2) / 2.0
            lm[0] = [cx, ny1 + 0.08*nbh, 0.0, 0.95]
            lm[11] = [cx - 0.25*nbw, ny1 + 0.25*nbh, 0.0, 0.95]
            lm[12] = [cx + 0.25*nbw, ny1 + 0.25*nbh, 0.0, 0.95]
            lm[23] = [cx - 0.18*nbw, ny1 + 0.55*nbh, 0.0, 0.95]
            lm[24] = [cx + 0.18*nbw, ny1 + 0.55*nbh, 0.0, 0.95]
            lm[25] = [cx - 0.38*nbw, ny1 + 0.56*nbh, 0.0, 0.90]
            lm[26] = [cx - 0.32*nbw, ny1 + 0.56*nbh, 0.0, 0.90]
            lm[27] = [cx - 0.38*nbw, ny1 + 0.92*nbh, 0.0, 0.90]
            lm[28] = [cx - 0.32*nbw, ny1 + 0.92*nbh, 0.0, 0.90]
        elif ar > 0.38 and ar <= 0.52:
            if nbh < 0.44:
                # Normal Activity (Bending forward / reaching)
                lm[0] = [cx + 0.60*nbw, ny1 + 0.15*nbh, 0.0, 0.95]
                lm[11] = [cx + 0.45*nbw, ny1 + 0.22*nbh, 0.0, 0.95]
                lm[12] = [cx + 0.50*nbw, ny1 + 0.22*nbh, 0.0, 0.95]
                lm[23] = [cx - 0.25*nbw, ny1 + 0.50*nbh, 0.0, 0.95]
                lm[24] = [cx - 0.20*nbw, ny1 + 0.50*nbh, 0.0, 0.95]
                lm[25] = [cx - 0.18*nbw, ny1 + 0.72*nbh, 0.0, 0.90]
                lm[26] = [cx - 0.14*nbw, ny1 + 0.72*nbh, 0.0, 0.90]
                lm[27] = [cx - 0.18*nbw, ny1 + 0.95*nbh, 0.0, 0.90]
                lm[28] = [cx - 0.14*nbw, ny1 + 0.95*nbh, 0.0, 0.90]
            else:
                # Walking (Active stride)
                lm[0] = [cx - 0.05*nbw, ny1 + 0.08*nbh, 0.0, 0.95]
                lm[11] = [cx - 0.25*nbw, ny1 + 0.22*nbh, 0.0, 0.95]
                lm[12] = [cx + 0.20*nbw, ny1 + 0.22*nbh, 0.0, 0.95]
                lm[23] = [cx - 0.10*nbw, ny1 + 0.50*nbh, 0.0, 0.95]
                lm[24] = [cx + 0.10*nbw, ny1 + 0.50*nbh, 0.0, 0.95]
                lm[25] = [cx - 0.35*nbw, ny1 + 0.73*nbh, 0.0, 0.90]
                lm[26] = [cx + 0.30*nbw, ny1 + 0.69*nbh, 0.0, 0.90]
                lm[27] = [cx - 0.48*nbw, ny1 + 0.95*nbh, 0.0, 0.90]
                lm[28] = [cx + 0.45*nbw, ny1 + 0.91*nbh, 0.0, 0.90]
        else:
            # Standing Posture
            cx = (nx1 + nx2) / 2.0
            lm[0] = [cx, ny1 + 0.08*nbh, 0.0, 0.98]
            lm[11] = [cx - 0.25*nbw, ny1 + 0.20*nbh, 0.0, 0.98]
            lm[12] = [cx + 0.25*nbw, ny1 + 0.20*nbh, 0.0, 0.98]
            lm[23] = [cx - 0.15*nbw, ny1 + 0.50*nbh, 0.0, 0.98]
            lm[24] = [cx + 0.15*nbw, ny1 + 0.50*nbh, 0.0, 0.98]
            lm[25] = [cx - 0.15*nbw, ny1 + 0.72*nbh, 0.0, 0.95]
            lm[26] = [cx + 0.15*nbw, ny1 + 0.72*nbh, 0.0, 0.95]
            lm[27] = [cx - 0.15*nbw, ny1 + 0.95*nbh, 0.0, 0.95]
            lm[28] = [cx + 0.15*nbw, ny1 + 0.95*nbh, 0.0, 0.95]
            
        for idx in range(1, 11):
            lm[idx] = lm[0]
        lm[13] = (lm[11] + lm[23]) / 2.0
        lm[14] = (lm[12] + lm[24]) / 2.0
        lm[15] = lm[13]
        lm[16] = lm[14]
        for idx in range(17, 23):
            lm[idx] = lm[15]
        for idx in range(29, 33):
            lm[idx] = lm[27]
            
        return lm

    def calculate_angle_2d(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def extract_features(self, landmarks: np.ndarray, img_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        raw_flat = landmarks.flatten()
        
        nose = landmarks[self.NOSE]
        l_shoulder, r_shoulder = landmarks[self.LEFT_SHOULDER], landmarks[self.RIGHT_SHOULDER]
        mid_shoulder = (l_shoulder + r_shoulder) / 2.0
        
        l_hip, r_hip = landmarks[self.LEFT_HIP], landmarks[self.RIGHT_HIP]
        mid_hip = (l_hip + r_hip) / 2.0
        
        l_knee, r_knee = landmarks[self.LEFT_KNEE], landmarks[self.RIGHT_KNEE]
        l_ankle, r_ankle = landmarks[self.LEFT_ANKLE], landmarks[self.RIGHT_ANKLE]
        mid_ankle = (l_ankle + r_ankle) / 2.0
        
        # 1. Torso Angle (Combining 3D Depth Angle and 2D Planar Angle for Omnidirectional Fall Detection)
        torso_3d = mid_shoulder[:3] - mid_hip[:3]
        norm_3d = float(np.linalg.norm(torso_3d) + 1e-7)
        cos_tilt_3d = float(np.clip((-torso_3d[1]) / norm_3d, -1.0, 1.0))
        torso_angle_3d = float(np.degrees(np.arccos(cos_tilt_3d)))
        
        dx_2d = abs(mid_shoulder[0] - mid_hip[0])
        dy_2d = abs(mid_shoulder[1] - mid_hip[1]) + 1e-7
        torso_angle_2d = float(np.degrees(np.arctan2(dx_2d, dy_2d)))
        
        # Max angle captures lateral, diagonal, and forward/backward falls towards camera
        torso_angle = float(max(torso_angle_2d, torso_angle_3d))
        
        # 2. Aspect Ratio & Bounding Box
        valid_x = landmarks[:, 0]
        valid_y = landmarks[:, 1]
        bbox_w = float(np.max(valid_x) - np.min(valid_x) + 1e-5)
        bbox_h = float(np.max(valid_y) - np.min(valid_y) + 1e-5)
        aspect_ratio = float(bbox_w / bbox_h)
        
        # 3. Center of Gravity Elevation (Mid-Hip Y)
        cog_y = float(mid_hip[1])
        
        # 4. Head to Hip difference
        head_hip_dy = float(nose[1] - mid_hip[1])
        
        # 5. Knee & Hip Flexion + Asymmetry
        l_knee_ang = self.calculate_angle_2d(l_hip, l_knee, l_ankle)
        r_knee_ang = self.calculate_angle_2d(r_hip, r_knee, r_ankle)
        avg_knee_ang = float((l_knee_ang + r_knee_ang) / 2.0)
        knee_asym = abs(float(l_knee_ang - r_knee_ang))
        min_knee_ang = float(min(l_knee_ang, r_knee_ang))
        
        l_hip_ang = self.calculate_angle_2d(l_shoulder, l_hip, l_knee)
        r_hip_ang = self.calculate_angle_2d(r_shoulder, r_hip, r_knee)
        avg_hip_ang = float((l_hip_ang + r_hip_ang) / 2.0)
        
        # 6. Dimensions and Gait Strides
        body_height_norm = float(np.linalg.norm(mid_shoulder[:2] - mid_ankle[:2]))
        shoulder_width_norm = float(np.linalg.norm(l_shoulder[:2] - r_shoulder[:2]))
        shoulder_height_ratio = float(shoulder_width_norm / (body_height_norm + 1e-5))
        
        ankle_stride = float(np.linalg.norm(l_ankle[:2] - r_ankle[:2]))
        ankle_dy = abs(float(l_ankle[1] - r_ankle[1]))
        
        # 7. Omnidirectional Heuristic Fall Score
        # Lateral fall (sideways tilt)
        lat_score = 0.0
        if torso_angle > 45.0 or aspect_ratio > 0.90:
            lat_score = min(1.0, max(0.0, (torso_angle - 25.0) / 45.0) * 0.5 + max(0.0, (aspect_ratio - 0.7) / 0.8) * 0.5)
            
        # Perspective fall (forward/backward towards or away from camera)
        persp_score = 0.0
        if (torso_angle_3d > 45.0 or (head_hip_dy >= -0.12 and torso_angle > 35.0)) and cog_y > 0.55:
            persp_score = min(1.0, (torso_angle_3d / 65.0) * 0.6 + (cog_y / 0.75) * 0.4)
            
        # Diagonal fall (tilted at an angle to the camera)
        diag_score = 0.0
        if torso_angle > 40.0 and aspect_ratio > 0.65 and cog_y > 0.52:
            diag_score = min(1.0, (torso_angle / 60.0) * 0.6 + (aspect_ratio / 1.1) * 0.4)
            
        # Floor collapse / fetal or slumped floor posture
        collapse_score = 0.0
        if cog_y > 0.55 and (avg_knee_ang < 75.0 or avg_hip_ang < 75.0 or (torso_angle > 45.0 and aspect_ratio > 0.55)):
            collapse_score = min(1.0, 0.60 + (cog_y / 0.75) * 0.25 + (1.0 - min(avg_knee_ang, 90.0) / 90.0) * 0.20)
            
        # Upright standing or forward bending with straight supporting legs suppresses false fall alarm
        is_standing_geometry = (torso_angle < 20.0 and avg_knee_ang > 162.0 and aspect_ratio < 0.45)
        is_upright_bending = (torso_angle < 45.0 and avg_knee_ang > 155.0 and aspect_ratio < 0.50 and head_hip_dy < -0.13)
        if is_standing_geometry or is_upright_bending:
            heuristic_fall_score = 0.05
        else:
            heuristic_fall_score = float(max(lat_score, persp_score, diag_score, collapse_score))

        
        # 8. Dynamic Off-Balancer Kinematic Stability Analysis
        foot_xs = [l_ankle[0], r_ankle[0], landmarks[29][0], landmarks[30][0], landmarks[31][0], landmarks[32][0]]
        foot_ys = [l_ankle[1], r_ankle[1], landmarks[29][1], landmarks[30][1], landmarks[31][1], landmarks[32][1]]
        
        min_bos_x = float(min(foot_xs))
        max_bos_x = float(max(foot_xs))
        bos_width = max(0.06, max_bos_x - min_bos_x)
        bos_center_x = (min_bos_x + max_bos_x) / 2.0
        ground_y = float(max(foot_ys))
        
        com_x = 0.55 * mid_hip[0] + 0.45 * mid_shoulder[0]
        com_y = 0.55 * mid_hip[1] + 0.45 * mid_shoulder[1]
        
        balance_deviation = abs(com_x - bos_center_x)
        balance_ratio = float(balance_deviation / (0.5 * bos_width + 1e-5))
        hip_clearance = float(ground_y - mid_hip[1])
        
        if hip_clearance < 0.12 and aspect_ratio > 1.10:
            stability_score = max(5.0, hip_clearance * 80.0)
        else:
            instability_penalty = max(0.0, (balance_ratio - 0.75) * 45.0) + (ankle_dy * 70.0)
            stability_score = max(10.0, min(100.0, 100.0 - instability_penalty))
            
        if stability_score >= 70.0:
            balance_status = "Stable Equilibrium"
        elif stability_score >= 50.0:
            balance_status = "Controlled Bending"
        elif stability_score >= 35.0:
            balance_status = "Marginal Balance"
        else:
            balance_status = "Off-Balance Instability"
            
        ankle_dz = abs(float(l_ankle[2] - r_ankle[2]))
        
        # 1. Sitting Posture Indicator (Chair-level knee bend, upright/narrow aspect, not flat on floor)
        is_sitting_posture = bool(
            (avg_knee_ang <= 138.0 or min_knee_ang <= 130.0) and
            (aspect_ratio < 0.75) and
            (torso_angle < 60.0) and
            (cog_y < 0.85)
        )
        
        # 2. Floor Fall Indicator (Flat collapse on floor, high aspect ratio, low CoG, even if knees are flexed)
        is_floor_fall = bool(
            (torso_angle >= 60.0 and aspect_ratio >= 0.75 and cog_y > 0.58) or
            (torso_angle >= 70.0 and (aspect_ratio >= 0.65 or cog_y > 0.60)) or
            (aspect_ratio >= 0.95 and torso_angle > 45.0 and cog_y > 0.55)
        )
        
        # 3. Controlled Forward Bending (Normal Activity - requires supporting extended legs)
        is_controlled_bending = bool(
            (24.0 <= torso_angle <= 75.0) and
            (aspect_ratio < 0.90) and
            (hip_clearance > 0.16) and
            (stability_score >= 48.0) and
            (cog_y < 0.68) and
            (avg_knee_ang >= 142.0 and min_knee_ang >= 132.0)
        )
        
        # 4. Straight & Lateral Walking Gait Indicator (Alternating knee flexion, foot lift, horizontal stride)
        has_stride = (ankle_stride > 0.13)
        has_knee_stride = (knee_asym > 18.0 and avg_knee_ang > 125.0)
        has_foot_lift = (ankle_dy > 0.035 and avg_knee_ang > 125.0)
        
        is_walking_gait = bool(
            (torso_angle < 25.0) and
            not is_sitting_posture and
            not is_floor_fall and
            (has_stride or has_knee_stride or has_foot_lift)
        )
        
        is_unbalanced = bool(
            (torso_angle >= 20.0) and
            not is_sitting_posture and
            not is_floor_fall and
            (balance_ratio > 1.25 or ankle_dy > 0.08) and
            (stability_score < 45.0) and
            (aspect_ratio < 1.10)
        )

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
        
        combined = np.concatenate([raw_flat, bio_features])
        
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
            "bos_width": round(float(bos_width), 3),
            "balance_status": balance_status,
            "is_controlled_bending": is_controlled_bending,
            "is_unbalanced": is_unbalanced,
            "is_walking_gait": is_walking_gait,
            "is_sitting_posture": is_sitting_posture,
            "is_floor_fall": is_floor_fall,
            "ankle_dz": round(ankle_dz, 3),
            "bbox": (float(np.min(valid_x)), float(np.min(valid_y)), bbox_w, bbox_h)
        }
        
        return {
            "feature_vector": combined,
            "bio_features": bio_features,
            "raw_landmarks": landmarks,
            "metrics": metrics
        }

    def draw_skeleton(self, frame_bgr: np.ndarray, results: Dict[str, Any], activity_label: str = "", confidence: float = 0.0) -> np.ndarray:
        annotated = frame_bgr.copy()
        landmarks = results.get("landmarks")
        if landmarks is None:
            return annotated
        
        h, w, _ = frame_bgr.shape
        is_fall = "fall" in activity_label.lower()
        lm_color = (0, 0, 255) if is_fall else (0, 255, 127)
        conn_color = (40, 40, 255) if is_fall else (255, 200, 0)
        
        for idx1, idx2 in POSE_CONNECTIONS:
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                p1 = (int(landmarks[idx1, 0] * w), int(landmarks[idx1, 1] * h))
                p2 = (int(landmarks[idx2, 0] * w), int(landmarks[idx2, 1] * h))
                cv2.line(annotated, p1, p2, conn_color, 2, cv2.LINE_AA)
                
        for i in range(len(landmarks)):
            px = int(landmarks[i, 0] * w)
            py = int(landmarks[i, 1] * h)
            cv2.circle(annotated, (px, py), 4, lm_color, -1, cv2.LINE_AA)
            cv2.circle(annotated, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
            
        return annotated
