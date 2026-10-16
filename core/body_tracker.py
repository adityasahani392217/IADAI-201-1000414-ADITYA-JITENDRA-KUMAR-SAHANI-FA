"""
core/body_tracker.py
====================
Primary subject tracking and landmark association across sequential video frames.
Matches bounding boxes and skeletal joints using spatial IoU, centroid displacement,
and detector confidence to maintain identity consistency during motion.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.kinematics import (
    calculate_bounding_box_iou,
    extract_normalized_features
)


def extract_pose_candidates(detection_result: Any) -> List[Dict[str, Any]]:
    """Parse YOLO pose estimation output into list of detected human candidate dictionaries."""
    if detection_result is None or detection_result.boxes is None or detection_result.keypoints is None:
        return []

    if len(detection_result.boxes) == 0:
        return []

    try:
        boxes_xyxy = detection_result.boxes.xyxy.cpu().numpy()
        coords_xy = detection_result.keypoints.xy.cpu().numpy()
        conf_tensor = detection_result.keypoints.conf

        if conf_tensor is not None:
            confidences = conf_tensor.cpu().numpy()
        else:
            confidences = np.ones((len(coords_xy), 17), dtype=np.float32)

        total_subjects = min(len(boxes_xyxy), len(coords_xy), len(confidences))
        candidates = []

        for idx in range(total_subjects):
            cand = {
                "bbox": boxes_xyxy[idx].astype(np.float32),
                "keypoints": coords_xy[idx].astype(np.float32),
                "confidences": confidences[idx].astype(np.float32),
                "mean_confidence": float(np.mean(confidences[idx]))
            }
            candidates.append(cand)
        return candidates
    except Exception:
        return []


class SubjectVisualTracker:
    """
    Maintains subject continuity across video frames by associating detections
    based on spatial overlap (IoU) and normalized Euclidean distance.
    """

    MAX_NORMALIZED_DISTANCE: float = 0.32
    MAX_CONSECUTIVE_MISSED_FRAMES: int = 5

    def __init__(self):
        self.last_bbox: Optional[np.ndarray] = None
        self.missed_frame_count: int = 0

    def reset(self) -> None:
        """Reset subject tracklet state."""
        self.last_bbox = None
        self.missed_frame_count = 0

    def update(
        self,
        detection_result: Any,
        frame_dimensions: Tuple[int, int, ...]
    ) -> Optional[Dict[str, Any]]:
        """
        Ingest current YOLO detection results, resolve most probable primary subject,
        and attach extracted normalized features.
        """
        h, w = frame_dimensions[:2]
        frame_diagonal = max(1.0, math.hypot(w, h))

        candidates = extract_pose_candidates(detection_result)
        selected_subject: Optional[Dict[str, Any]] = None

        if candidates:
            if self.last_bbox is None:
                # First detection: select candidate with highest mean confidence
                selected_subject = max(candidates, key=lambda c: c["mean_confidence"])
            else:
                prev_cx = (self.last_bbox[0] + self.last_bbox[2]) / 2.0
                prev_cy = (self.last_bbox[1] + self.last_bbox[3]) / 2.0
                prev_center = np.array([prev_cx, prev_cy], dtype=np.float32)

                top_candidate = None
                top_match_score = -1e9

                for cand in candidates:
                    curr_cx = (cand["bbox"][0] + cand["bbox"][2]) / 2.0
                    curr_cy = (cand["bbox"][1] + cand["bbox"][3]) / 2.0
                    curr_center = np.array([curr_cx, curr_cy], dtype=np.float32)

                    norm_dist = float(np.linalg.norm(curr_center - prev_center)) / frame_diagonal
                    iou_overlap = calculate_bounding_box_iou(self.last_bbox, cand["bbox"])

                    # Composite similarity metric: rewards IoU and confidence, penalizes distance jump
                    score = (2.2 * iou_overlap) + cand["mean_confidence"] - (2.0 * norm_dist)
                    cand["_norm_dist"] = norm_dist
                    cand["_iou"] = iou_overlap

                    if score > top_match_score:
                        top_match_score = score
                        top_candidate = cand

                if top_candidate is not None:
                    if top_candidate["_norm_dist"] <= self.MAX_NORMALIZED_DISTANCE or top_candidate["_iou"] > 0.04:
                        selected_subject = top_candidate
                    else:
                        # Fallback: subject moved quickly, stood up, or fell - maintain continuous tracking
                        selected_subject = max(candidates, key=lambda c: c["mean_confidence"])

        if selected_subject is not None:
            self.last_bbox = selected_subject["bbox"].copy()
            self.missed_frame_count = 0
            # Precompute normalized 51-dim feature vector
            selected_subject["features"] = extract_normalized_features(
                selected_subject["keypoints"],
                selected_subject["confidences"],
                selected_subject["bbox"]
            )
            return selected_subject

        self.missed_frame_count += 1
        if self.missed_frame_count > self.MAX_CONSECUTIVE_MISSED_FRAMES:
            self.last_bbox = None

        return None
