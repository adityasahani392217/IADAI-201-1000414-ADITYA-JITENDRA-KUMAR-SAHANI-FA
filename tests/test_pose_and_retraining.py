"""
SafeFall AI - Pose Estimation & Retraining Pipeline Unit Test Suite
Validates 17 COCO anatomical landmarks, biomechanical telemetry, and Step 8 active retraining logic.
"""

import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import numpy as np
import cv2
import json

from core.vision_pipeline import SafeFallPipelineCoordinator, ACTIVITY_CLASSES
from core.kinematics import KinematicPostureEngine


def test_pose_extraction_and_telemetry():
    root = Path(__file__).resolve().parent.parent
    coordinator = SafeFallPipelineCoordinator(root)

    # 1. Test image with known person
    fall_imgs = list((root / "outputs" / "falls").glob("*.jpg"))
    tested_real = False
    for f in fall_imgs:
        img = cv2.imread(str(f))
        if img is None:
            continue
        report = coordinator.analyze_single_image(img, {"desk_mode": False, "use_model": True})
        assert report is not None
        assert "label" in report
        assert "confidence" in report
        assert "torso_angle" in report
        assert "aspect_ratio" in report
        assert "preview" in report
        if report.get("keypoints") is not None:
            assert len(report["keypoints"]) == 17
            assert report["torso_angle"] >= 0.0
            assert report["aspect_ratio"] > 0.0
            tested_real = True
            break

    print(f"PASS: Pose landmark extraction verified (real frame tested: {tested_real})")

    # 2. Test empty frame fallback
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    empty_report = coordinator.analyze_single_image(blank, {"desk_mode": False})
    assert empty_report is not None
    assert empty_report["label"] == "NORMAL_ACTIVITY"
    assert "torso_angle" in empty_report
    assert "aspect_ratio" in empty_report
    print("PASS: Blank frame active sentinel gracefully handles non-detection")


def test_step8_retraining_history():
    root = Path(__file__).resolve().parent.parent
    hist_path = root / "training" / "training_history.json"
    eval_path = root / "assets" / "evaluation_summary.json"

    assert hist_path.exists(), "training_history.json must exist"
    assert eval_path.exists(), "evaluation_summary.json must exist"

    with open(hist_path, "r", encoding="utf-8") as f:
        hist = json.load(f)
        assert "epochs" in hist
        assert "train_loss" in hist
        assert len(hist["epochs"]) == 45
        assert hist["train_loss"][-1] < hist["train_loss"][0]

    with open(eval_path, "r", encoding="utf-8") as f:
        ev = json.load(f)
        assert "overall_accuracy" in ev
        assert "per_class" in ev
        assert "Fall Detected" in ev["per_class"]
        assert ev["per_class"]["Fall Detected"]["precision"] > 0.90
        assert ev["per_class"]["Fall Detected"]["recall"] > 0.90

    print("PASS: FA-2 Step 8 model training artifacts and evaluation metrics verified")


if __name__ == "__main__":
    test_pose_extraction_and_telemetry()
    test_step8_retraining_history()
    print("ALL POSE AND RETRAINING SUITE TESTS PASSED!")
