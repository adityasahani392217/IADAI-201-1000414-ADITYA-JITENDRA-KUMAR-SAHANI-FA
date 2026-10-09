"""
core/vision_pipeline.py
=======================
Unified computer vision, YOLOv8 pose landmarking, and deep temporal inference pipeline.
Orchestrates static image analysis, video file processing, and live WebRTC streaming.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from core.body_tracker import SubjectVisualTracker
from core.kinematics import (
    ACTIVITY_CLASSES,
    COCO_TOPOLOGY,
    FEATURE_DIMENSION,
    REFERENCE_FPS,
    SEQUENCE_LENGTH,
    SEQUENCE_STRIDE,
    KinematicPostureEngine,
    extract_normalized_features
)
from core.temporal_filter import (
    TemporalDecisionFilter,
    aggregate_detection_intervals,
    resolve_activity_label
)

try:
    import av
    from streamlit_webrtc import VideoProcessorBase
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False
    VideoProcessorBase = object  # type: ignore


# -------------------------------------------------------------
# IMAGE PREPROCESSING & VISUALIZATION
# -------------------------------------------------------------
def enhance_lowlight_image(image_bgr: np.ndarray, gamma: float = 1.6) -> np.ndarray:
    """Enhance low-light video frames using power-law gamma transformation and CLAHE."""
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255.0 for i in range(256)], dtype=np.uint8)
    gamma_corrected = cv2.LUT(image_bgr, lut)
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def render_pose_skeleton(
    image_bgr: np.ndarray,
    keypoints: Optional[np.ndarray],
    bbox: Optional[np.ndarray] = None,
    is_alert: bool = False,
    status_label: Optional[str] = None
) -> np.ndarray:
    """Overlay clean skeletal topology, bounding box, and joint nodes on target BGR frame with Red-Yellow-Green traffic light lines."""
    if keypoints is None and bbox is None:
        return image_bgr

    # Dynamic Red-Yellow-Green Line Color Determination:
    # RED: Acute Fall Detected / Alert
    # YELLOW: Off Balance / Postural Instability / Warning
    # GREEN: Safe Nominal Activity (Standing, Walking, Sitting, Normal Activity)
    norm_status = (status_label or "").upper()
    if is_alert or norm_status == "FALL":
        line_color = (45, 45, 230)    # Crisp Coral Red (BGR)
        joint_color = (80, 80, 255)   # Vivid Red node
    elif norm_status in ("OFF_BALANCE", "OFF BALANCE", "WARNING"):
        line_color = (30, 200, 245)   # Bright Amber Yellow (BGR)
        joint_color = (50, 225, 255)  # Vivid Yellow node
    else:
        line_color = (70, 195, 60)    # Crisp Healthcare Green (BGR)
        joint_color = (100, 230, 95)  # Vivid Green node

    # Draw subject bounding box if present
    if bbox is not None and len(bbox) == 4:
        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        cv2.rectangle(image_bgr, (bx1, by1), (bx2, by2), line_color, 2, cv2.LINE_AA)
        tag_text = norm_status.replace("_", " ").title() if norm_status else "Subject Tracked"
        (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)
        tag_y1 = max(0, by1 - th - 8)
        tag_y2 = max(th + 8, by1)
        cv2.rectangle(image_bgr, (bx1, tag_y1), (bx1 + tw + 14, tag_y2), line_color, -1)
        cv2.putText(
            image_bgr,
            tag_text,
            (bx1 + 6, tag_y2 - 5),
            cv2.FONT_HERSHEY_DUPLEX,
            0.50,
            (255, 255, 255) if norm_status == "FALL" else (15, 15, 15),
            1,
            cv2.LINE_AA
        )

    if keypoints is not None:
        for joint_a, joint_b in COCO_TOPOLOGY:
            if joint_a < len(keypoints) and joint_b < len(keypoints):
                x1, y1 = int(keypoints[joint_a][0]), int(keypoints[joint_a][1])
                x2, y2 = int(keypoints[joint_b][0]), int(keypoints[joint_b][1])
                if min(x1, y1, x2, y2) > 0:
                    cv2.line(image_bgr, (x1, y1), (x2, y2), line_color, 3, cv2.LINE_AA)

        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            if x > 0 and y > 0:
                cv2.circle(image_bgr, (x, y), 5, joint_color, -1, cv2.LINE_AA)
                cv2.circle(image_bgr, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

    return image_bgr


# -------------------------------------------------------------
# PYTORCH NEURAL ARCHITECTURE
# -------------------------------------------------------------
class TemporalBiLSTMNetwork(nn.Module):
    """Bidirectional LSTM sequence classifier for temporal posture kinematics."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIMENSION,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.30,
        num_classes: int = 4
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # Classify from the final temporal state
        return self.fc_head(lstm_out[:, -1, :])


# -------------------------------------------------------------
# PIPELINE COORDINATOR
# -------------------------------------------------------------
class SafeFallPipelineCoordinator:
    """Central engine managing YOLOv8 pose detector, kinematic logic, and neural models."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_device = 0 if torch.cuda.is_available() else "cpu"
        self.inference_lock = threading.Lock()
        if self.device.type == "cpu":
            num_cores = os.cpu_count() or 4
            torch.set_num_threads(min(4, max(2, num_cores // 2)))

        self.pose_model_path = self._locate_pose_model()
        self.pose_detector = YOLO(self.pose_model_path)
        # Warmup YOLO to eliminate first-frame compilation latency
        try:
            dummy = np.zeros((192, 192, 3), dtype=np.uint8)
            with torch.inference_mode():
                self.pose_detector.predict(dummy, verbose=False, imgsz=192, device=self.yolo_device, classes=[0], max_det=1)
        except Exception:
            pass
        self.neural_model, self.is_trained_4class, self.engine_status = self._init_neural_classifier()


    def _locate_pose_model(self) -> str:
        """Locate YOLOv8 pose weights in models/ or project root."""
        search_paths = [
            self.root_dir / "models" / "pose_model" / "yolov8n-pose.pt",
            self.root_dir / "models" / "yolov8n-pose.pt",
            self.root_dir / "yolov8n-pose.pt"
        ]
        for p in search_paths:
            if p.exists():
                return str(p)
        return "yolov8n-pose.pt"

    def _init_neural_classifier(self) -> Tuple[Optional[nn.Module], bool, str]:
        """Attempt to load trained 4-class BiLSTM weights from disk."""
        candidate_paths = [
            self.root_dir / "models" / "activity_model" / "best_model.pt",
            self.root_dir / "model" / "safefall_nn_model.pth"
        ]
        for ckpt in candidate_paths:
            if ckpt.exists():
                try:
                    payload = torch.load(ckpt, map_location=self.device, weights_only=False)
                    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
                    if hasattr(state, "state_dict"):
                        state = state.state_dict()
                    w = state.get("lstm.weight_ih_l0")
                    c = state.get("fc_head.3.weight") or state.get("classifier.3.weight")
                    if w is not None and c is not None and tuple(w.shape) == (256, 51) and tuple(c.shape) == (4, 64):
                        net = TemporalBiLSTMNetwork()
                        net.load_state_dict(state, strict=False)
                        net.to(self.device).eval()
                        return net, True, "Trained 4-class BiLSTM Neural Network"
                except Exception:
                    pass
        return None, False, "Rule-based kinematic pose engine"

    def uses_neural_model(self, force_legacy: bool = False) -> bool:
        """Check whether neural model inference is active."""
        return self.neural_model is not None and (self.is_trained_4class or force_legacy)

    def extract_subject_pose(
        self,
        frame_bgr: np.ndarray,
        tracker: SubjectVisualTracker,
        img_size: int = 224
    ) -> Optional[Dict[str, Any]]:
        """Run YOLOv8 pose detector with inference_mode and class filtering for maximum FPS."""
        with self.inference_lock, torch.inference_mode():
            yolo_results = self.pose_detector.predict(
                frame_bgr,
                verbose=False,
                conf=0.15,
                imgsz=min(int(img_size), 224),
                device=self.yolo_device,
                classes=[0],
                max_det=1
            )[0]
        return tracker.update(yolo_results, frame_bgr.shape)


    def infer_neural_probabilities(self, sequence_batch: np.ndarray) -> np.ndarray:
        """Inference (N, 30, 51) sequence tensor -> (N, 4) probability array."""
        tensor = torch.as_tensor(np.asarray(sequence_batch, dtype=np.float32), device=self.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        with self.inference_lock, torch.no_grad():
            logits = self.neural_model(tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    # ---------------- SINGLE IMAGE ANALYSIS ----------------
    def analyze_single_image(self, frame_bgr: np.ndarray, options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate posture and fall likelihood for a single photograph."""
        tracker = SubjectVisualTracker()
        subject = self.extract_subject_pose(frame_bgr, tracker, options.get("imgsz", 480))
        if subject is None:
            # User requirement: Instead of no person detected, show Normal Activity because when there is no one, it is normal!
            norm_probs = [0.0] * len(ACTIVITY_CLASSES)
            norm_idx = ACTIVITY_CLASSES.index("NORMAL_ACTIVITY") if "NORMAL_ACTIVITY" in ACTIVITY_CLASSES else 0
            norm_probs[norm_idx] = 0.96
            stand_idx = ACTIVITY_CLASSES.index("STANDING") if "STANDING" in ACTIVITY_CLASSES else -1
            if stand_idx != -1 and stand_idx != norm_idx:
                norm_probs[stand_idx] = 0.04

            preview_img = frame_bgr.copy()
            cv2.rectangle(preview_img, (16, 16), (320, 60), (250, 252, 250), -1)
            cv2.rectangle(preview_img, (16, 16), (320, 60), (70, 195, 60), 2, cv2.LINE_AA)
            cv2.circle(preview_img, (36, 38), 6, (70, 195, 60), -1, cv2.LINE_AA)
            cv2.putText(preview_img, "Normal Activity (Clear)", (52, 44), cv2.FONT_HERSHEY_DUPLEX, 0.58, (35, 45, 30), 1, cv2.LINE_AA)

            return {
                "label": "NORMAL_ACTIVITY",
                "confidence": 0.96,
                "probs": norm_probs,
                "frames": 1,
                "detected": 0,
                "rate": 0.0,
                "windows": 1,
                "preview": preview_img,
                "timeline": None,
                "times": None,
                "votes": None,
                "fall_time": None,
                "engine": "SafeFall Active Sentinel",
                "kind": "image"
            }


        kinematics = KinematicPostureEngine(desk_mode=options.get("desk_mode", True))
        rule_probs = kinematics.register_frame(
            0.0,
            subject["keypoints"],
            subject["confidences"],
            subject["bbox"],
            allow_dynamic_motion=False
        )

        use_ai = options.get("use_model", False) and self.uses_neural_model()
        if use_ai:
            repeated_seq = np.repeat(subject["features"][None], SEQUENCE_LENGTH, axis=0)[None]
            probs = self.infer_neural_probabilities(repeated_seq)[0]
        elif rule_probs is not None:
            probs = rule_probs
        else:
            return None

        label = resolve_activity_label(probs, options.get("fall_thr", 0.60))
        conf = float(probs[ACTIVITY_CLASSES.index(label)])
        is_fall = label == "FALL"

        preview_img = render_pose_skeleton(
            frame_bgr.copy(),
            subject["keypoints"],
            subject["bbox"],
            is_alert=is_fall,
            status_label=label
        )

        engine_name = "Trained 4-class BiLSTM" if use_ai else "Rule-based kinematic engine"

        return {
            "label": label,
            "confidence": conf,
            "probs": probs,
            "frames": 1,
            "detected": 1,
            "rate": 1.0,
            "windows": 1,
            "preview": preview_img,
            "timeline": None,
            "times": None,
            "votes": None,
            "fall_time": None,
            "engine": engine_name,
            "kind": "image"
        }

    # ---------------- VIDEO ANALYSIS ----------------
    def analyze_video_file(
        self,
        video_path: str,
        options: Dict[str, Any],
        max_frames: int = 900,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Optional[Dict[str, Any]]:
        """Perform full-duration kinematic and sliding-window analysis on video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or not np.isfinite(fps) or fps < 1:
            fps = REFERENCE_FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames

        tracker = SubjectVisualTracker()
        kinematics = KinematicPostureEngine(desk_mode=options.get("desk_mode", True))

        feature_history: List[np.ndarray] = []
        rule_probabilities: List[np.ndarray] = []
        keypoint_history: List[Optional[np.ndarray]] = []
        box_history: List[Optional[np.ndarray]] = []
        detected_count = 0
        frame_idx = 0

        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            subject = self.extract_subject_pose(frame, tracker, options.get("imgsz", 480))
            if subject is None:
                feature_history.append(np.zeros(FEATURE_DIMENSION, dtype=np.float32))
                rule_probabilities.append(np.full(len(ACTIVITY_CLASSES), np.nan))
                keypoint_history.append(None)
                box_history.append(None)
            else:
                detected_count += 1
                feature_history.append(subject["features"])
                keypoint_history.append(subject["keypoints"])
                box_history.append(subject["bbox"])
                rp = kinematics.register_frame(
                    frame_idx / fps,
                    subject["keypoints"],
                    subject["confidences"],
                    subject["bbox"]
                )
                rule_probabilities.append(np.full(len(ACTIVITY_CLASSES), np.nan) if rp is None else rp.astype(float))

            frame_idx += 1
            if progress_callback and frame_idx % 4 == 0:
                progress_callback(min(frame_idx / max(target_frames, 1), 1.0))

        cap.release()
        if frame_idx == 0 or detected_count == 0:
            return None

        feats_arr = np.asarray(feature_history, dtype=np.float32)
        rprobs_arr = np.asarray(rule_probabilities, dtype=float)
        has_pose = np.array([k is not None for k in keypoint_history])

        starts = list(range(0, frame_idx - SEQUENCE_LENGTH + 1, SEQUENCE_STRIDE)) if frame_idx >= SEQUENCE_LENGTH else [0]
        use_ai = options.get("use_model", False) and self.uses_neural_model()

        if use_ai:
            padded = feats_arr if frame_idx >= SEQUENCE_LENGTH else np.concatenate(
                [feats_arr, np.repeat(feats_arr[-1:], SEQUENCE_LENGTH - frame_idx, axis=0)]
            )
            ai_probs = self.infer_neural_probabilities([padded[s:s + SEQUENCE_LENGTH] for s in starts])

        window_rows: List[np.ndarray] = []
        window_centers: List[int] = []

        for i, s in enumerate(starts):
            e = min(s + SEQUENCE_LENGTH, frame_idx)
            if has_pose[s:e].mean() < 0.45:
                continue

            if use_ai:
                pr = ai_probs[i]
            else:
                seg = rprobs_arr[s:e]
                seg = seg[~np.isnan(seg[:, 0])]
                if len(seg) == 0:
                    continue
                pr = seg.mean(axis=0)

            window_rows.append(pr)
            window_centers.append(min(s + SEQUENCE_LENGTH // 2, frame_idx - 1))

        if not window_rows:
            return None

        window_rows_arr = np.asarray(window_rows)
        summary = aggregate_detection_intervals(window_rows_arr, options.get("fall_thr", 0.60))
        timestamps = [c / fps for c in window_centers]
        fall_time = timestamps[summary["sustained_fall_run"][0]] if summary["sustained_fall_run"] else None

        preview_frame = None
        try:
            peak_frame_idx = window_centers[summary["peak_window_idx"]]
            valid_k = next((j for j in range(peak_frame_idx, -1, -1) if keypoint_history[j] is not None), None)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, peak_frame_idx)
            ok, fr = cap.read()
            cap.release()
            if ok:
                if valid_k is not None:
                    render_pose_skeleton(
                        fr,
                        keypoint_history[valid_k],
                        box_history[valid_k],
                        is_alert=(summary["verdict"] == "FALL"),
                        status_label=summary["verdict"]
                    )
                h, w = fr.shape[:2]
                if w > 640:
                    fr = cv2.resize(fr, (640, int(h * 640 / w)))
                preview_frame = fr
        except Exception:
            preview_frame = None

        engine_name = "Trained 4-class BiLSTM" if use_ai else "Rule-based kinematic engine"

        return {
            "label": summary["verdict"],
            "confidence": summary["confidence"],
            "probs": summary["distribution"],
            "frames": frame_idx,
            "detected": detected_count,
            "rate": detected_count / max(frame_idx, 1),
            "windows": len(window_rows),
            "preview": preview_frame,
            "timeline": window_rows_arr,
            "times": timestamps,
            "votes": summary["activity_votes"],
            "fall_time": fall_time,
            "engine": engine_name,
            "kind": "video"
        }


# -------------------------------------------------------------
# WEBRTC LIVE STREAM PROCESSOR
# -------------------------------------------------------------
class LiveStreamWorker(VideoProcessorBase):
    """Real-time WebRTC worker processing camera frames in background threads."""

    def __init__(self, coordinator: SafeFallPipelineCoordinator, outputs_dir: Optional[Path] = None):
        self.coordinator = coordinator
        self.outputs_dir = outputs_dir
        self.config = {
            "fall_thr": 0.60,
            "need": 2,
            "alpha": 0.65,
            "stride": 1,
            "enhance": False,
            "gamma": 1.6,
            "imgsz": 256,
            "desk_mode": True,
            "force_legacy": False
        }

        self.tracker = SubjectVisualTracker()
        self.kinematics = KinematicPostureEngine()
        self.decision_filter = TemporalDecisionFilter()
        self.feature_buffer = deque(maxlen=SEQUENCE_LENGTH)

        self.current_state = {"label": "WARMING UP", "conf": 0.0, "probs": [1.0 / len(ACTIVITY_CLASSES)] * len(ACTIVITY_CLASSES)}
        self.last_subject: Optional[Dict[str, Any]] = None
        self.consecutive_misses: int = 0
        self.frame_counter: int = 0
        self.fps_meter: float = 0.0
        self._last_frame_time = time.time()
        self._last_push_time = 0.0

        self._state_lock = threading.Lock()
        self._fall_onset_time: float = 0.0
        self._public_telemetry = {
            "label": "STARTING",
            "conf": 0.0,
            "probs": [1.0 / len(ACTIVITY_CLASSES)] * len(ACTIVITY_CLASSES),
            "fps": 0.0,
            "buffer": 0,
            "person": False,
            "fall": False,
            "fall_duration": 0.0,
            "events": [],
            "error": "",
            "engine": ""
        }
        self._prior_fall_state: bool = False

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Safely extract instantaneous telemetry snapshot."""
        with self._state_lock:
            snap = dict(self._public_telemetry)
            snap["events"] = list(snap["events"])
            return snap

    def _set_telemetry(self, **kwargs) -> None:
        """Update internal public telemetry values safely."""
        with self._state_lock:
            self._public_telemetry.update(kwargs)

    def _push_features(self, feat: np.ndarray, timestamp: float) -> bool:
        """Pushes feature vectors maintaining 25 FPS pacing."""
        dt = timestamp - self._last_push_time
        if dt < 0.8 / REFERENCE_FPS:
            return False
        repeats = int(min(6, max(1, round(dt * REFERENCE_FPS))))
        for _ in range(repeats):
            self.feature_buffer.append(feat)
        self._last_push_time = timestamp
        return True

    def _log_fall_incident(self, frame_bgr: np.ndarray, confidence: float) -> Dict[str, Any]:
        """Record fall timestamp, write snapshot image, and append to CSV log."""
        ts = datetime.now()
        filename = f"fall_{ts:%Y%m%d_%H%M%S}.jpg"
        if self.outputs_dir is not None:
            try:
                self.outputs_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.outputs_dir / filename), frame_bgr)
                csv_path = self.outputs_dir.parent / "fall_events.csv"
                is_fresh = not csv_path.exists()
                with open(csv_path, "a", encoding="utf-8") as fh:
                    if is_fresh:
                        fh.write("time,confidence,snapshot\n")
                    fh.write(f"{ts.isoformat(timespec='seconds')},{confidence:.3f},{filename}\n")
            except Exception:
                filename = ""
        return {
            "time": ts.strftime("%H:%M:%S"),
            "conf": round(float(confidence), 3),
            "file": filename
        }

    def _render_hud_overlay(
        self,
        frame_bgr: np.ndarray,
        label: str,
        confidence: float,
        is_fall: bool,
        engine_tag: str
    ) -> np.ndarray:
        """Draw clean, modern healthcare posture pill badge and perimeter alert."""
        h, w = frame_bgr.shape[:2]

        # Dynamic Red-Yellow-Green color system in BGR:
        norm_label = label.upper()
        if is_fall or norm_label == "FALL":
            dot_color = (45, 45, 230)    # Red
            badge_border = (45, 45, 230)
        elif norm_label in ("OFF_BALANCE", "OFF BALANCE", "WARNING"):
            dot_color = (30, 200, 245)   # Yellow / Amber
            badge_border = (30, 200, 245)
        else:
            dot_color = (70, 195, 60)    # Green
            badge_border = (200, 215, 195)
        text_color = (35, 45, 30)  # Deep Charcoal

        # Sleek pill badge in top-left
        badge_w = 260
        badge_h = 44
        bx, by = 16, 16

        # Fast solid rounded pill badge (zero memory allocation)
        cv2.rectangle(frame_bgr, (bx, by), (bx + badge_w, by + badge_h), (250, 252, 250), -1)
        cv2.rectangle(frame_bgr, (bx, by), (bx + badge_w, by + badge_h), badge_border, 1, cv2.LINE_AA)

        # Status dot
        cv2.circle(frame_bgr, (bx + 16, by + 22), 6, dot_color, -1, cv2.LINE_AA)

        # Activity label and confidence
        display_label = label.replace("_", " ").title()
        cv2.putText(
            frame_bgr,
            f"{display_label}",
            (bx + 32, by + 29),
            cv2.FONT_HERSHEY_DUPLEX,
            0.64,
            text_color,
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame_bgr,
            f"{confidence:.0%}",
            (bx + badge_w - 56, by + 29),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            dot_color,
            1,
            cv2.LINE_AA
        )

        if is_fall:
            # Urgent perimeter alert line in Red
            cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1), (45, 45, 230), 6)
            cv2.putText(
                frame_bgr,
                "🚨 ACUTE FALL DETECTED",
                (bx + 10, by + badge_h + 36),
                cv2.FONT_HERSHEY_DUPLEX,
                0.90,
                (45, 45, 230),
                2,
                cv2.LINE_AA
            )
        elif norm_label in ("OFF_BALANCE", "OFF BALANCE"):
            # Caution perimeter alert line in Yellow
            cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1), (30, 200, 245), 4)
            cv2.putText(
                frame_bgr,
                "⚠️ POSTURE UNSTABLE (OFF BALANCE)",
                (bx + 10, by + badge_h + 36),
                cv2.FONT_HERSHEY_DUPLEX,
                0.80,
                (30, 200, 245),
                2,
                cv2.LINE_AA
            )

        return frame_bgr

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Process incoming video frame through tracking, posture heuristics, and filtering."""
        curr_time = time.time()
        frame_interval = curr_time - self._last_frame_time
        self._last_frame_time = curr_time

        if frame_interval > 0:
            self.fps_meter = (0.90 * self.fps_meter) + (0.10 / frame_interval) if self.fps_meter else 1.0 / frame_interval

        cfg = self.config
        if cfg["enhance"]:
            frame_bgr = enhance_lowlight_image(frame_bgr, cfg["gamma"])

        use_ai = self.coordinator.uses_neural_model(cfg["force_legacy"])
        self.kinematics.desk_mode = cfg["desk_mode"]
        self.frame_counter += 1

        stride_step = max(int(cfg["stride"]), 1)
        if self.frame_counter % stride_step == 0:
            subject = self.coordinator.extract_subject_pose(frame_bgr, self.tracker, cfg["imgsz"])
            frame_probabilities = None

            if subject is not None:
                self.consecutive_misses = 0
                self.last_subject = subject

                kin_probs = self.kinematics.register_frame(
                    curr_time,
                    subject["keypoints"],
                    subject["confidences"],
                    subject["bbox"]
                )

                if use_ai:
                    if self._push_features(subject["features"], curr_time) and len(self.feature_buffer) == SEQUENCE_LENGTH:
                        batch = np.asarray(self.feature_buffer, dtype=np.float32)[None]
                        ai_probs = self.coordinator.infer_neural_probabilities(batch)[0]
                        frame_probabilities = (0.60 * ai_probs) + (0.40 * (kin_probs if kin_probs is not None else ai_probs))
                    else:
                        frame_probabilities = kin_probs
                else:
                    frame_probabilities = kin_probs


                if frame_probabilities is not None:
                    label, conf, smoothed = self.decision_filter.update(
                        frame_probabilities,
                        cfg["alpha"],
                        cfg["fall_thr"],
                        cfg["need"]
                    )
                    self.current_state = {
                        "label": label,
                        "conf": conf,
                        "probs": smoothed.tolist()
                    }
            else:
                self.consecutive_misses += 1
                if use_ai and self.consecutive_misses <= 15:
                    self._push_features(np.zeros(FEATURE_DIMENSION, dtype=np.float32), curr_time)
                if self.consecutive_misses > 15:
                    self.feature_buffer.clear()
                    self.kinematics.reset()
                    self.decision_filter.reset()
                    norm_probs = [0.0] * len(ACTIVITY_CLASSES)
                    norm_idx = ACTIVITY_CLASSES.index("NORMAL_ACTIVITY") if "NORMAL_ACTIVITY" in ACTIVITY_CLASSES else 0
                    norm_probs[norm_idx] = 0.95
                    self.current_state = {"label": "NORMAL_ACTIVITY", "conf": 0.95, "probs": norm_probs}


        person_visible = self.consecutive_misses <= 3 and self.last_subject is not None
        curr_label = self.current_state["label"]
        curr_conf = self.current_state["conf"]
        is_fall = curr_label == "FALL"

        if person_visible and self.last_subject is not None:
            render_pose_skeleton(
                frame_bgr,
                self.last_subject["keypoints"],
                self.last_subject["bbox"],
                is_alert=is_fall,
                status_label=curr_label
            )

        frame_bgr = self._render_hud_overlay(
            frame_bgr,
            curr_label,
            curr_conf,
            is_fall,
            "AI" if use_ai else "Rules"
        )

        if is_fall:
            if not self._prior_fall_state or self._fall_onset_time == 0.0:
                self._fall_onset_time = curr_time
            fall_duration = max(0.0, curr_time - self._fall_onset_time)
        else:
            self._fall_onset_time = 0.0
            fall_duration = 0.0

        events_list = self._public_telemetry["events"]
        if is_fall and not self._prior_fall_state:
            events_list = (events_list + [self._log_fall_incident(frame_bgr, curr_conf)])[-20:]
        self._prior_fall_state = is_fall

        self._set_telemetry(
            label=curr_label,
            conf=float(curr_conf),
            probs=list(self.current_state["probs"]),
            fps=float(self.fps_meter),
            buffer=len(self.feature_buffer),
            person=bool(person_visible),
            fall=is_fall,
            fall_duration=round(fall_duration, 1),
            events=events_list,
            error="",
            engine="AI" if use_ai else "Rules"
        )

        return frame_bgr

    def recv(self, frame: Any) -> Any:
        """Streamlit-webrtc worker frame receptor."""
        img = frame.to_ndarray(format="bgr24")
        try:
            h, w = img.shape[:2]
            if w > 480:
                scale = 480.0 / w
                img = cv2.resize(img, (480, int(h * scale)), interpolation=cv2.INTER_LINEAR)
            img = self.process_frame(img)
        except Exception as e:
            self._set_telemetry(error=str(e)[:140])
            cv2.putText(img, "Processing latency", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

