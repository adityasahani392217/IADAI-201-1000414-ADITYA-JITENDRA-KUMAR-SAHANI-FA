"""
SafeFall AI - Model Training Engine (FA-2 Step 5)
Trains PyTorch Deep Neural Network and Random Forest baseline on 70% Train, 15% Val.
Saves model weights, scaler, encoder, and epoch metrics history.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from model.fall_classifier import SafeFallDeepNet, CLASSES

def train_pipeline(base_dir: str = None):
    if base_dir is None:
        base_dir = PROJECT_ROOT
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "model")
    training_dir = os.path.join(base_dir, "training")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    
    # 1. Load Data
    train_path = os.path.join(data_dir, "train_dataset.csv")
    val_path = os.path.join(data_dir, "val_dataset.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    feature_cols = [col for col in train_df.columns if col != "activity_label"]
    
    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    y_train_raw = train_df["activity_label"].values
    
    X_val_raw = val_df[feature_cols].values.astype(np.float32)
    y_val_raw = val_df["activity_label"].values
    
    # 2. Fit Scaler & Label Encoder
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    label_encoder = LabelEncoder()
    # Fit encoder with fixed classes order
    label_encoder.fit(CLASSES)
    y_train = label_encoder.transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    
    # Save Scaler & Encoder
    joblib.dump(scaler, os.path.join(model_dir, "feature_scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.joblib"))
    print("Feature scaler and label encoder saved successfully.")
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(len(CLASSES)), y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Computed Balanced Class Weights: {np.round(class_weights, 3)}")

    # 3. Train Baseline Random Forest
    print("\n--- Training Random Forest Baseline (Balanced) ---")
    rf = RandomForestClassifier(n_estimators=160, max_depth=16, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_train_acc = rf.score(X_train, y_train)
    rf_val_acc = rf.score(X_val, y_val)
    print(f"Random Forest - Train Accuracy: {rf_train_acc*100:.2f}%, Val Accuracy: {rf_val_acc*100:.2f}%")
    joblib.dump(rf, os.path.join(model_dir, "safefall_rf_model.joblib"))
    
    # 4. Train Deep Learning Neural Network (PyTorch)
    print("\n--- Training SafeFall Deep Neural Network (PyTorch) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = SafeFallDeepNet(input_dim=len(feature_cols), num_classes=len(CLASSES), dropout_rate=0.25).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    epochs = 45
    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_loss = float("inf")
    best_model_path = os.path.join(model_dir, "safefall_nn_model.pth")
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                running_val_loss += loss.item() * batch_X.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0
        scheduler.step(epoch_val_loss)
        
        history["epochs"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 2))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 2))
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || "
                  f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
            
    # Save training history JSON
    history_path = os.path.join(training_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
        
    print(f"\nTraining Complete! Best model saved to: {best_model_path}")
    print(f"Training history saved to: {history_path}")

if __name__ == "__main__":
    train_pipeline()
