"""
SafeFall AI - 6-Class Le2i Fall Dataset & Multi-Angle Image Processor
Extracts MediaPipe Pose landmarks (132 raw features) and 10 clinical biomechanical metrics (142 total features)
for 6 distinct classes:
  1. Fall Detected
  2. Off Balance
  3. Normal Activity
  4. Sitting
  5. Standing
  6. Walking
Produces stratified 70% Train, 15% Val, 15% Test datasets.
"""

import os
import sys
import glob
import re
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.pose_detector import SafeFallPoseDetector
from model.fall_classifier import CLASSES

LE2I_DIR = os.path.join(PROJECT_ROOT, "data", "le2i_dataset")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "internet_images")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def find_video_pairs(dataset_dir: str) -> List[Tuple[str, Optional[str]]]:
    """Finds all .avi video files and their corresponding annotation .txt files."""
    video_files = glob.glob(os.path.join(dataset_dir, "**", "*.avi"), recursive=True)
    pairs = []
    
    for v_path in video_files:
        v_dir = os.path.dirname(v_path)
        v_name = os.path.splitext(os.path.basename(v_path))[0]
        
        parent_dir = os.path.dirname(v_dir)
        annot_candidates = [
            os.path.join(v_dir, f"{v_name}.txt"),
            os.path.join(parent_dir, "Annotation_files", f"{v_name}.txt"),
            os.path.join(parent_dir, "Annotation_files", f"{v_name.replace(' ', '%20')}.txt"),
            os.path.join(parent_dir, "Annotation_files", f"{v_name.replace('%20', ' ')}.txt"),
            os.path.join(v_dir, "..", "Annotation_files", f"{v_name}.txt")
        ]
        
        annot_path = None
        for cand in annot_candidates:
            if os.path.exists(cand):
                annot_path = cand
                break
                
        if annot_path is None:
            norm_name = re.sub(r'[\s%20]+', '', v_name.lower())
            for t_file in glob.glob(os.path.join(parent_dir, "**", "*.txt"), recursive=True):
                t_base = re.sub(r'[\s%20]+', '', os.path.splitext(os.path.basename(t_file))[0].lower())
                if t_base == norm_name:
                    annot_path = t_file
                    break
                    
        pairs.append((v_path, annot_path))
        
    return pairs

def parse_annotation(annot_path: Optional[str]) -> Tuple[int, int, Dict[int, int]]:
    """
    Parses start, end frame, and per-frame activity code dictionary from Le2i annotation.
    Returns (start_frame, end_frame, frame_codes_dict).
    """
    if annot_path is None or not os.path.exists(annot_path):
        return -1, -1, {}
        
    start_f = -1
    end_f = -1
    frame_codes = {}
    try:
        with open(annot_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) >= 2:
            try:
                start_f = int(float(lines[0]))
                end_f = int(float(lines[1]))
            except Exception:
                pass
        for line in lines[2:]:
            parts = line.split(',')
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].strip().isdigit():
                f_idx = int(parts[0])
                c_idx = int(parts[1])
                frame_codes[f_idx] = c_idx
    except Exception:
        pass
    return start_f, end_f, frame_codes

def classify_adl_posture(metrics: Dict[str, Any]) -> str:
    """Classifies non-fall/non-collapse posture into Sitting, Walking, Standing, or Normal Activity."""
    torso_angle = metrics.get("torso_angle_deg", 0.0)
    aspect_ratio = metrics.get("aspect_ratio", 0.3)
    avg_knee_ang = metrics.get("avg_knee_angle_deg", 170.0)
    avg_hip_ang = metrics.get("avg_hip_angle_deg", 170.0)
    stride = metrics.get("ankle_stride", 0.05)
    cog_y = metrics.get("center_of_gravity_y", 0.5)
    head_hip_dy = metrics.get("head_hip_dy", -0.25)
    
    # Sitting: knees and hips flexed (~90-130 deg) or lowered CoG with wider aspect ratio
    if (avg_knee_ang < 135.0 and avg_hip_ang < 135.0) or (cog_y > 0.60 and aspect_ratio > 0.50):
        return "Sitting"
    
    # Normal Activity: bending forward, reaching, picking objects
    if torso_angle >= 25.0 or head_hip_dy >= -0.15:
        return "Normal Activity"
    
    # Walking: active ankle stride / leg alternation with upright torso
    if stride >= 0.065 and avg_knee_ang > 135.0:
        return "Walking"
        
    return "Standing"

def process_dataset():
    detector = SafeFallPoseDetector(static_image_mode=False)
    video_pairs = find_video_pairs(LE2I_DIR)
    print(f"Total Le2i video files to process: {len(video_pairs)}", flush=True)
    
    class_samples: Dict[str, List[np.ndarray]] = {c: [] for c in CLASSES}
    
    # 1. Process Le2i Videos
    for v_idx, (v_path, a_path) in enumerate(video_pairs):
        start_fall, end_fall, frame_codes = parse_annotation(a_path)
        has_fall = (start_fall > 0 and end_fall > start_fall)
        
        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue
            
        frames_to_sample = {}
        if frame_codes:
            for f_idx, c_code in frame_codes.items():
                if f_idx > total_frames:
                    continue
                if c_code == 8:
                    # Off Balance / Falling onset
                    frames_to_sample[f_idx] = "Off Balance"
                elif c_code == 7:
                    # Fall Detected (impact / lying on floor)
                    frames_to_sample[f_idx] = "Fall Detected"
                elif f_idx % 4 == 0:
                    frames_to_sample[f_idx] = "ADL"
        elif has_fall:
            # Fall videos without frame-by-frame codes
            # Off Balance phase: from start_fall to end_fall
            for f in range(max(1, start_fall), min(total_frames, end_fall), 1):
                frames_to_sample[f] = "Off Balance"
            # Fall Detected phase: from end_fall to end_fall + 40
            for f in range(end_fall, min(total_frames, end_fall + 40), 2):
                frames_to_sample[f] = "Fall Detected"
            # Pre-fall ADL
            if start_fall > 15:
                for f in range(1, start_fall - 4, 4):
                    frames_to_sample[f] = "ADL"
        else:
            # Non-fall ADL video
            for f in range(1, total_frames, 4):
                frames_to_sample[f] = "ADL"
                
        sorted_frame_indices = set(frames_to_sample.keys())
        
        curr_f = 0
        while cap.isOpened() and curr_f < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            curr_f += 1
            
            if curr_f in sorted_frame_indices:
                expected_type = frames_to_sample[curr_f]
                res = detector.process_frame(frame, allow_synthetic_fallback=False)
                lms = res.get("landmarks")
                
                if lms is not None:
                    feat_dict = detector.extract_features(lms, img_shape=frame.shape[:2])
                    feat_vec = feat_dict["feature_vector"]
                    
                    if expected_type == "Fall Detected":
                        lbl = "Fall Detected"
                    elif expected_type == "Off Balance":
                        lbl = "Off Balance"
                    else:
                        lbl = classify_adl_posture(feat_dict["metrics"])
                        
                    class_samples[lbl].append(feat_vec)
                    
        cap.release()
        
        if (v_idx + 1) % 25 == 0 or (v_idx + 1) == len(video_pairs):
            counts_summary = {c: len(class_samples[c]) for c in CLASSES}
            print(f"[{v_idx+1:03d}/{len(video_pairs)}] Current samples from Le2i: {counts_summary}", flush=True)

    # 2. Ingest 600 Real Multi-Angle Images from data/internet_images/
    img_folders = {
        "Fall_Detected": "Fall Detected",
        "Off_Balance": "Off Balance",
        "Normal_Activity": "Normal Activity",
        "Sitting": "Sitting",
        "Standing": "Standing",
        "Walking": "Walking"
    }
    
    print("\nProcessing verified multi-angle images from data/internet_images/...", flush=True)
    img_counts = {c: 0 for c in CLASSES}
    
    for folder_name, target_label in img_folders.items():
        folder_path = os.path.join(IMAGES_DIR, folder_name)
        if not os.path.exists(folder_path):
            continue
            
        img_files = glob.glob(os.path.join(folder_path, "*.*"))
        for img_p in img_files:
            img = cv2.imread(img_p)
            if img is None:
                continue
            res = detector.process_frame(img, allow_synthetic_fallback=False)
            lms = res.get("landmarks")
            if lms is not None:
                feat_dict = detector.extract_features(lms, img_shape=img.shape[:2])
                class_samples[target_label].append(feat_dict["feature_vector"])
                img_counts[target_label] += 1
                
    print(f"Extracted features from multi-angle images: {img_counts}", flush=True)

    # Build Unified DataFrame
    col_names = [f"lm_{i}_{coord}" for i in range(33) for coord in ["x", "y", "z", "vis"]]
    bio_col_names = [
        "torso_angle_deg",
        "aspect_ratio",
        "center_of_gravity_y",
        "head_hip_dy",
        "avg_knee_angle_deg",
        "avg_hip_angle_deg",
        "body_height_norm",
        "shoulder_width_norm",
        "shoulder_height_ratio",
        "heuristic_fall_score"
    ]
    all_cols = col_names + bio_col_names
    
    rows = []
    labels = []
    for cls_name in CLASSES:
        for feat in class_samples[cls_name]:
            rows.append(feat)
            labels.append(cls_name)

    df = pd.DataFrame(rows, columns=all_cols)
    df["activity_label"] = labels
    
    print(f"\n==========================================", flush=True)
    print(f"Full Combined Dataset (6 Classes): {len(df)} samples", flush=True)
    print(df["activity_label"].value_counts(), flush=True)
    print(f"==========================================", flush=True)
    
    # Stratified Train/Val/Test Split (70% / 15% / 15%)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["activity_label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["activity_label"])
    
    train_path = os.path.join(DATA_DIR, "train_dataset.csv")
    val_path = os.path.join(DATA_DIR, "val_dataset.csv")
    test_path = os.path.join(DATA_DIR, "test_dataset.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved Train set: {len(train_df)} samples -> {train_path}", flush=True)
    print(f"Saved Val set:   {len(val_df)} samples -> {val_path}", flush=True)
    print(f"Saved Test set:  {len(test_df)} samples -> {test_path}", flush=True)

if __name__ == "__main__":
    process_dataset()
