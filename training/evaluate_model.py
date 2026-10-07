"""
SafeFall AI - Model Evaluation & Analytics Module (FA-2 Step 6)
Evaluates trained models on the unseen Test Set (15% split = 300 samples).
Generates publication-quality charts for Accuracy, Loss, Confusion Matrix,
Precision/Recall/F1-score, and exports metrics JSON.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from model.fall_classifier import SafeFallDeepNet, CLASSES

def evaluate_pipeline(base_dir: str = None):
    if base_dir is None:
        base_dir = PROJECT_ROOT
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "model")
    training_dir = os.path.join(base_dir, "training")
    assets_dir = os.path.join(base_dir, "assets")
    
    os.makedirs(assets_dir, exist_ok=True)
    
    # 1. Load Test Data
    test_path = os.path.join(data_dir, "test_dataset.csv")
    test_df = pd.read_csv(test_path)
    feature_cols = [c for c in test_df.columns if c != "activity_label"]
    
    X_test_raw = test_df[feature_cols].values.astype(np.float32)
    y_test_raw = test_df["activity_label"].values
    
    # Load Scaler & Encoder
    scaler = joblib.load(os.path.join(model_dir, "feature_scaler.joblib"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    
    X_test = scaler.transform(X_test_raw)
    y_test = label_encoder.transform(y_test_raw)
    
    # 2. Load PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SafeFallDeepNet(input_dim=len(feature_cols), num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "safefall_nn_model.pth"), map_location=device))
    model.eval()
    
    with torch.no_grad():
        tensor_in = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(tensor_in)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = np.argmax(probs, axis=1)
        
    # 3. Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, labels=range(len(CLASSES)))
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    
    cm = confusion_matrix(y_test, y_pred, labels=range(len(CLASSES)))
    
    print("\n=======================================================")
    print("      SAFEFALL AI - TEST SET EVALUATION METRICS       ")
    print("=======================================================")
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro Precision:   {macro_prec * 100:.2f}%")
    print(f"Macro Recall:      {macro_rec * 100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1 * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))
    
    # Save Metrics JSON
    metrics_summary = {
        "overall_accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": {}
    }
    for i, cls_name in enumerate(CLASSES):
        metrics_summary["per_class"][cls_name] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(support[i])
        }
        
    with open(os.path.join(assets_dir, "evaluation_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    # 4. Generate Visual Plots
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=True,
                annot_kws={"size": 12, "weight": "bold"})
    plt.title("SafeFall AI - Activity Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Activity Class", fontsize=12, fontweight="bold")
    plt.ylabel("True Activity Class", fontsize=12, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(assets_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved: {cm_path}")
    
    # Plot 2 & 3: Loss & Accuracy Graphs from Training History
    history_file = os.path.join(training_dir, "training_history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            hist = json.load(f)
            
        epochs = hist["epochs"]
        
        # Accuracy Curve
        plt.figure(figsize=(8, 5), dpi=300)
        plt.plot(epochs, hist["train_acc"], label="Training Accuracy", color="#1E88E5", linewidth=2.5)
        plt.plot(epochs, hist["val_acc"], label="Validation Accuracy", color="#43A047", linewidth=2.5, linestyle="--")
        plt.title("SafeFall AI - Model Accuracy vs. Epochs", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11, fontweight="bold")
        plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
        plt.legend(loc="lower right", frameon=True)
        plt.ylim(70, 102)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        acc_path = os.path.join(assets_dir, "accuracy_curve.png")
        plt.savefig(acc_path)
        plt.close()
        print(f"Saved: {acc_path}")
        
        # Loss Curve
        plt.figure(figsize=(8, 5), dpi=300)
        plt.plot(epochs, hist["train_loss"], label="Training Loss", color="#E53935", linewidth=2.5)
        plt.plot(epochs, hist["val_loss"], label="Validation Loss", color="#FB8C00", linewidth=2.5, linestyle="--")
        plt.title("SafeFall AI - Model Loss vs. Epochs (Cross-Entropy)", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11, fontweight="bold")
        plt.ylabel("Loss Value", fontsize=11, fontweight="bold")
        plt.legend(loc="upper right", frameon=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        loss_path = os.path.join(assets_dir, "loss_curve.png")
        plt.savefig(loss_path)
        plt.close()
        print(f"Saved: {loss_path}")
        
    # Plot 4: Precision, Recall, F1 Bar Chart per Class
    x = np.arange(len(CLASSES))
    width = 0.25
    plt.figure(figsize=(10, 5), dpi=300)
    plt.bar(x - width, prec * 100, width, label="Precision", color="#0288D1")
    plt.bar(x, rec * 100, width, label="Recall", color="#388E3C")
    plt.bar(x + width, f1 * 100, width, label="F1-Score", color="#F57C00")
    plt.title("SafeFall AI - Per-Class Performance Metrics (%)", fontsize=13, fontweight="bold")
    plt.xlabel("Activity Class", fontsize=11, fontweight="bold")
    plt.ylabel("Percentage Score (%)", fontsize=11, fontweight="bold")
    plt.xticks(x, CLASSES, rotation=15)
    plt.ylim(80, 105)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    bar_path = os.path.join(assets_dir, "precision_recall_f1_bar.png")
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved: {bar_path}")
    
    # Plot 5: Class Distribution Chart
    plt.figure(figsize=(8, 4.5), dpi=300)
    counts = test_df["activity_label"].value_counts()[CLASSES]
    colors = ["#DC2626", "#EA580C", "#D97706", "#7C3AED", "#059669", "#2563EB"]
    plt.bar(CLASSES, counts.values, color=colors, edgecolor="black", alpha=0.85)
    plt.title("SafeFall AI - Test Set Class Distribution (6 Classes)", fontsize=13, fontweight="bold")
    plt.xlabel("Activity Class", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Samples", fontsize=11, fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    dist_path = os.path.join(assets_dir, "class_distribution.png")
    plt.savefig(dist_path)
    plt.close()
    print(f"Saved: {dist_path}")

if __name__ == "__main__":
    evaluate_pipeline()
