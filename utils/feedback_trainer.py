"""
SafeFall AI - Active Learning & Human Feedback Training Module
Allows real-time logging of user feedback postures and fast incremental
model retraining to adapt to specific camera viewpoints and angles.
"""

import os
import csv
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_CSV = os.path.join(PROJECT_ROOT, "data", "user_feedback_data.csv")

CLASSES = [
    "Fall Detected",
    "Off Balance",
    "Normal Activity",
    "Sitting",
    "Standing",
    "Walking"
]

def log_feedback_sample(feature_vector: np.ndarray, ground_truth_label: str, feedback_path: str = FEEDBACK_CSV) -> int:
    """Appends a 142-feature sample with user ground truth label to CSV."""
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    
    file_exists = os.path.exists(feedback_path)
    feat_flat = feature_vector.flatten()
    
    with open(feedback_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            # Header: 142 feature names + activity_label
            header = [f"f_{i}" for i in range(len(feat_flat))] + ["activity_label"]
            writer.writerow(header)
        writer.writerow(list(feat_flat) + [ground_truth_label])
        
    return count_feedback_samples(feedback_path)

def count_feedback_samples(feedback_path: str = FEEDBACK_CSV) -> int:
    """Returns the count of logged user feedback samples."""
    if not os.path.exists(feedback_path):
        return 0
    try:
        df = pd.read_csv(feedback_path)
        return len(df)
    except Exception:
        return 0

def clear_feedback_samples(feedback_path: str = FEEDBACK_CSV):
    """Resets the feedback dataset."""
    if os.path.exists(feedback_path):
        os.remove(feedback_path)

def retrain_model_with_feedback(model_dir: Optional[str] = None, 
                                feedback_path: str = FEEDBACK_CSV,
                                epochs: int = 15) -> Dict[str, Any]:
    """
    Retrains Random Forest and fine-tunes PyTorch DeepNet using base train dataset + user feedback.
    Applies oversampling to user feedback samples to adapt to user's camera angles.
    """
    if model_dir is None:
        model_dir = os.path.join(PROJECT_ROOT, "model")
    base_train_csv = os.path.join(PROJECT_ROOT, "data", "train_dataset.csv")
    
    # 1. Load Base Training Data
    base_df = pd.read_csv(base_train_csv)
    feat_cols = [c for c in base_df.columns if c != "activity_label"]
    
    # 2. Append User Feedback Data (Oversampled 5x so new user samples strongly guide the weights)
    feedback_count = 0
    if os.path.exists(feedback_path):
        try:
            fb_df = pd.read_csv(feedback_path)
            if len(fb_df) > 0:
                feedback_count = len(fb_df)
                fb_df.columns = feat_cols + ["activity_label"]
                # Oversample 5x
                fb_oversampled = pd.concat([fb_df] * 5, ignore_index=True)
                combined_df = pd.concat([base_df, fb_oversampled], ignore_index=True)
            else:
                combined_df = base_df
        except Exception:
            combined_df = base_df
    else:
        combined_df = base_df
        
    X_raw = combined_df[feat_cols].values.astype(np.float32)
    y_raw = combined_df["activity_label"].values
    
    # 3. Fit Scaler and Encoder
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    encoder = LabelEncoder()
    encoder.fit(CLASSES)
    y_encoded = encoder.transform(y_raw)
    
    # Save Scaler & Encoder
    joblib.dump(scaler, os.path.join(model_dir, "feature_scaler.joblib"))
    joblib.dump(encoder, os.path.join(model_dir, "label_encoder.joblib"))
    
    # 4. Retrain Random Forest
    rf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42)
    rf.fit(X_scaled, y_encoded)
    rf_acc = rf.score(X_scaled, y_encoded) * 100.0
    joblib.dump(rf, os.path.join(model_dir, "safefall_rf_model.joblib"))
    
    # 5. Fine-Tune PyTorch DeepNet
    from model.fall_classifier import SafeFallDeepNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = SafeFallDeepNet(input_dim=len(feat_cols), num_classes=len(CLASSES)).to(device)
    
    # Load existing weights if present
    nn_path = os.path.join(model_dir, "safefall_nn_model.pth")
    if os.path.exists(nn_path):
        try:
            nn_model.load_state_dict(torch.load(nn_path, map_location=device))
        except Exception:
            pass
            
    dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_encoded, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(nn_model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    nn_model.train()
    for _ in range(epochs):
        for b_X, b_y in loader:
            b_X, b_y = b_X.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = nn_model(b_X)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            
    torch.save(nn_model.state_dict(), nn_path)
    
    return {
        "status": "SUCCESS",
        "feedback_samples_used": feedback_count,
        "total_training_samples": len(combined_df),
        "rf_accuracy": round(rf_acc, 2)
    }
