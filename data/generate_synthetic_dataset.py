"""
SafeFall AI - Biomechanical Dataset Generator & Preprocessor
Generates diverse, realistic clinical posture distributions modeled after MediaPipe
landmark geometry and benchmark fall datasets (Le2i, UR Fall, FallFree).
Covers 5 activity classes:
 - Fall Detected (Lateral, Diagonal, Perspective Forward/Backward, Floor Collapse)
 - Normal Activity (Reaching, Bending Forward, Picking Objects)
 - Sitting (Office Chair Foreshortened, Side/Diagonal View)
 - Standing (Upright Postures Across Multi-Scale Camera Viewpoints)
 - Walking (Active Dynamic Gait, Stride Variation, Leg Elevation)
Features are mathematically computed directly from landmark geometry using
SafeFallPoseDetector.extract_features() to ensure 100% mathematical consistency.
Strictly splits into 70% Train, 15% Validation, 15% Test sets.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.pose_detector import SafeFallPoseDetector
from model.fall_classifier import CLASSES

_detector = SafeFallPoseDetector()

def generate_landmarks_for_class(activity: str, num_samples: int, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    features_list = []
    
    for _ in range(num_samples):
        lm = np.zeros((33, 4), dtype=np.float32)
        lm[:, 3] = np.random.uniform(0.90, 0.99, size=33)
        x_shift = np.random.uniform(-0.10, 0.10)
        base_x = 0.5 + x_shift

        if activity == "Fall Detected":
            fall_type = np.random.choice(["lateral", "diagonal", "perspective", "floor_collapse"], p=[0.25, 0.25, 0.25, 0.25])
            
            if fall_type == "lateral":
                cog_y = np.random.uniform(0.65, 0.88)
                lm[0] = [base_x - 0.25, cog_y - 0.02, 0.0, 0.95]
                lm[11] = [base_x - 0.15, cog_y - 0.03, 0.0, 0.95]
                lm[12] = [base_x - 0.15, cog_y + 0.03, 0.0, 0.95]
                lm[23] = [base_x + 0.05, cog_y - 0.02, 0.0, 0.95]
                lm[24] = [base_x + 0.05, cog_y + 0.02, 0.0, 0.95]
                lm[25] = [base_x + 0.20, cog_y - 0.02, 0.0, 0.90]
                lm[26] = [base_x + 0.20, cog_y + 0.02, 0.0, 0.90]
                lm[27] = [base_x + 0.35, cog_y - 0.02, 0.0, 0.90]
                lm[28] = [base_x + 0.35, cog_y + 0.02, 0.0, 0.90]
                
            elif fall_type == "diagonal":
                cog_y = np.random.uniform(0.62, 0.85)
                lm[0] = [base_x - 0.18, cog_y - 0.15, 0.0, 0.95]
                lm[11] = [base_x - 0.10, cog_y - 0.10, 0.0, 0.95]
                lm[12] = [base_x - 0.08, cog_y - 0.06, 0.0, 0.95]
                lm[23] = [base_x + 0.06, cog_y, 0.0, 0.95]
                lm[24] = [base_x + 0.08, cog_y + 0.04, 0.0, 0.95]
                lm[25] = [base_x + 0.18, cog_y + 0.08, 0.0, 0.90]
                lm[26] = [base_x + 0.20, cog_y + 0.12, 0.0, 0.90]
                lm[27] = [base_x + 0.28, cog_y + 0.14, 0.0, 0.90]
                lm[28] = [base_x + 0.30, cog_y + 0.18, 0.0, 0.90]
                
            elif fall_type == "perspective":
                cog_y = np.random.uniform(0.65, 0.88)
                lm[0] = [base_x, cog_y - 0.06, -0.30, 0.95]
                lm[11] = [base_x - 0.12, cog_y - 0.02, -0.20, 0.95]
                lm[12] = [base_x + 0.12, cog_y - 0.02, -0.20, 0.95]
                lm[23] = [base_x - 0.08, cog_y + 0.05, 0.05, 0.95]
                lm[24] = [base_x + 0.08, cog_y + 0.05, 0.05, 0.95]
                lm[25] = [base_x - 0.10, cog_y + 0.12, 0.25, 0.90]
                lm[26] = [base_x + 0.10, cog_y + 0.12, 0.25, 0.90]
                lm[27] = [base_x - 0.10, cog_y + 0.18, 0.40, 0.90]
                lm[28] = [base_x + 0.10, cog_y + 0.18, 0.40, 0.90]
                
            else:
                # Floor collapse
                cog_y = np.random.uniform(0.64, 0.86)
                lm[0] = [base_x - 0.08, cog_y - 0.12, 0.0, 0.95]
                lm[11] = [base_x - 0.06, cog_y - 0.06, 0.0, 0.95]
                lm[12] = [base_x + 0.06, cog_y - 0.06, 0.0, 0.95]
                lm[23] = [base_x - 0.05, cog_y, 0.0, 0.95]
                lm[24] = [base_x + 0.05, cog_y, 0.0, 0.95]
                lm[25] = [base_x - 0.12, cog_y + 0.05, 0.10, 0.90]
                lm[26] = [base_x + 0.10, cog_y + 0.05, 0.10, 0.90]
                lm[27] = [base_x - 0.10, cog_y + 0.08, 0.15, 0.90]
                lm[28] = [base_x + 0.12, cog_y + 0.08, 0.15, 0.90]

        elif activity == "Sitting":
            cog_y = np.random.uniform(0.54, 0.68)
            head_y = cog_y - np.random.uniform(0.24, 0.32)
            sh_y = head_y + 0.07
            is_angle_view = np.random.choice([True, False], p=[0.5, 0.5])
            
            if is_angle_view:
                # 90-degree side/diagonal seated posture
                thigh_dx = np.random.uniform(0.12, 0.18)
                lm[0] = [base_x, head_y, 0.0, 0.95]
                lm[11] = [base_x - 0.07, sh_y, 0.0, 0.95]
                lm[12] = [base_x + 0.07, sh_y, 0.0, 0.95]
                lm[23] = [base_x - 0.05, cog_y, 0.0, 0.95]
                lm[24] = [base_x + 0.05, cog_y, 0.0, 0.95]
                lm[25] = [base_x - 0.05 + thigh_dx, cog_y + 0.06, 0.10, 0.90]
                lm[26] = [base_x + 0.05 + thigh_dx, cog_y + 0.06, 0.10, 0.90]
                lm[27] = [base_x - 0.05 + thigh_dx, cog_y + 0.26, 0.10, 0.90]
                lm[28] = [base_x + 0.05 + thigh_dx, cog_y + 0.26, 0.10, 0.90]
            else:
                # Front-view office desk sitting
                lm[0] = [base_x, head_y, 0.0, 0.95]
                lm[11] = [base_x - 0.09, sh_y, 0.0, 0.95]
                lm[12] = [base_x + 0.09, sh_y, 0.0, 0.95]
                lm[23] = [base_x - 0.06, cog_y, 0.0, 0.95]
                lm[24] = [base_x + 0.06, cog_y, 0.0, 0.95]
                lm[25] = [base_x - 0.08, cog_y + 0.10, 0.15, 0.90]
                lm[26] = [base_x + 0.08, cog_y + 0.10, 0.15, 0.90]
                lm[27] = [base_x - 0.07, cog_y + 0.28, 0.15, 0.90]
                lm[28] = [base_x + 0.07, cog_y + 0.28, 0.15, 0.90]

        elif activity == "Standing":
            cog_y = np.random.uniform(0.48, 0.54)
            scale = np.random.uniform(0.75, 1.0)
            head_y = cog_y - 0.32 * scale
            sh_y = head_y + 0.08 * scale
            lm[0] = [base_x, head_y, 0.0, 0.98]
            lm[11] = [base_x - 0.07 * scale, sh_y, 0.0, 0.98]
            lm[12] = [base_x + 0.07 * scale, sh_y, 0.0, 0.98]
            lm[23] = [base_x - 0.04 * scale, cog_y, 0.0, 0.98]
            lm[24] = [base_x + 0.04 * scale, cog_y, 0.0, 0.98]
            lm[25] = [base_x - 0.04 * scale, cog_y + 0.18 * scale, 0.0, 0.95]
            lm[26] = [base_x + 0.04 * scale, cog_y + 0.18 * scale, 0.0, 0.95]
            lm[27] = [base_x - 0.04 * scale, cog_y + 0.36 * scale, 0.0, 0.95]
            lm[28] = [base_x + 0.04 * scale, cog_y + 0.36 * scale, 0.0, 0.95]

        elif activity == "Walking":
            cog_y = np.random.uniform(0.48, 0.54)
            scale = np.random.uniform(0.75, 1.0)
            head_y = cog_y - 0.30 * scale
            sh_y = head_y + 0.08 * scale
            stride = np.random.uniform(0.08, 0.14) * scale
            dy = np.random.uniform(0.02, 0.05) * scale
            lm[0] = [base_x, head_y, 0.0, 0.95]
            lm[11] = [base_x - 0.07 * scale, sh_y, 0.0, 0.95]
            lm[12] = [base_x + 0.07 * scale, sh_y, 0.0, 0.95]
            lm[23] = [base_x - 0.04 * scale, cog_y, 0.0, 0.95]
            lm[24] = [base_x + 0.04 * scale, cog_y, 0.0, 0.95]
            lm[25] = [base_x - stride * 0.5, cog_y + 0.17 * scale, 0.05, 0.90]
            lm[26] = [base_x + stride * 0.5, cog_y + 0.17 * scale - dy * 0.5, -0.05, 0.90]
            lm[27] = [base_x - stride, cog_y + 0.35 * scale, 0.10, 0.90]
            lm[28] = [base_x + stride, cog_y + 0.35 * scale - dy, -0.10, 0.90]

        else: # Normal Activity (Bending forward, reaching down)
            cog_y = np.random.uniform(0.48, 0.58)
            reach_x = np.random.uniform(0.04, 0.18)
            head_y = cog_y - np.random.uniform(0.14, 0.25)
            sh_y = head_y + 0.07
            lm[0] = [base_x + reach_x * 1.2, head_y, 0.0, 0.95]
            lm[11] = [base_x + reach_x - 0.05, sh_y, 0.0, 0.95]
            lm[12] = [base_x + reach_x + 0.05, sh_y, 0.0, 0.95]
            lm[23] = [base_x - 0.04, cog_y, 0.0, 0.95]
            lm[24] = [base_x + 0.04, cog_y, 0.0, 0.95]
            lm[25] = [base_x - 0.04, cog_y + 0.16, 0.0, 0.90]
            lm[26] = [base_x + 0.04, cog_y + 0.16, 0.0, 0.90]
            lm[27] = [base_x - 0.04, cog_y + 0.33, 0.0, 0.90]
            lm[28] = [base_x + 0.04, cog_y + 0.33, 0.0, 0.90]

        # Upper limb interpolation
        for idx in range(1, 11):
            if np.all(lm[idx] == 0):
                lm[idx] = lm[0] + np.random.uniform(-0.015, 0.015, size=4)
                lm[idx, 3] = 0.95
        lm[13] = (lm[11] + lm[23]) / 2 + np.random.uniform(-0.02, 0.02, size=4)
        lm[14] = (lm[12] + lm[24]) / 2 + np.random.uniform(-0.02, 0.02, size=4)
        lm[15] = lm[13] + np.random.uniform(-0.03, 0.03, size=4)
        lm[16] = lm[14] + np.random.uniform(-0.03, 0.03, size=4)
        for idx in range(17, 23):
            lm[idx] = lm[15] + np.random.uniform(-0.02, 0.02, size=4)
        for idx in range(29, 33):
            lm[idx] = lm[27] + np.random.uniform(-0.02, 0.02, size=4)

        noise = np.random.normal(0, 0.004, size=(33, 4)).astype(np.float32)
        noise[:, 3] = 0
        lm = np.clip(lm + noise, 0.0, 1.0)

        # Extract features mathematically using the exact detector engine
        fd = _detector.extract_features(lm)
        features_list.append(fd["feature_vector"])

    return np.array(features_list, dtype=np.float32)

def build_and_save_dataset(base_dir: str = None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    
    samples_per_class = {c: 1000 for c in CLASSES}
    
    all_rows = []
    all_labels = []
    
    seed = 101
    for cls_name, count in samples_per_class.items():
        feats = generate_landmarks_for_class(cls_name, count, seed=seed)
        seed += 37
        for row in feats:
            all_rows.append(row)
            all_labels.append(cls_name)
            
    X = np.array(all_rows, dtype=np.float32)
    y = np.array(all_labels)
    
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
    
    df = pd.DataFrame(X, columns=all_cols)
    df["activity_label"] = y
    
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["activity_label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["activity_label"])
    
    train_path = os.path.join(base_dir, "train_dataset.csv")
    val_path = os.path.join(base_dir, "val_dataset.csv")
    test_path = os.path.join(base_dir, "test_dataset.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("Dataset generated successfully!")
    print(f" - Train Set: {len(train_df)} samples (70%) -> {train_path}")
    print(f" - Val Set:   {len(val_df)} samples (15%) -> {val_path}")
    print(f" - Test Set:  {len(test_df)} samples (15%) -> {test_path}")

if __name__ == "__main__":
    build_and_save_dataset()
