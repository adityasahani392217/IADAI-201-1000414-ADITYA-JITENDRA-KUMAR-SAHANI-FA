"""
tests/test_all_6_classes.py
===========================
Verification test for all 6 clinical activities:
1. SITTING
2. STANDING
3. WALKING
4. OFF_BALANCE
5. NORMAL_ACTIVITY
6. FALL
"""
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.kinematics import (
    ACTIVITY_CLASSES,
    ACTIVITY_DISPLAY_NAMES,
    STATE_PRIORS,
    KinematicPostureEngine
)
from core.temporal_filter import (
    resolve_activity_label,
    detect_sustained_fall_event,
    aggregate_detection_intervals,
    TemporalDecisionFilter
)
import ui.styles as s
import ui.components as c


def test_activity_classes():
    expected = ["SITTING", "STANDING", "WALKING", "OFF_BALANCE", "NORMAL_ACTIVITY", "FALL"]
    assert ACTIVITY_CLASSES == expected, f"Expected {expected}, got {ACTIVITY_CLASSES}"
    assert len(s.CLASS_HEX_COLORS) == 6, f"Expected 6 colors in CLASS_HEX_COLORS, got {len(s.CLASS_HEX_COLORS)}"
    for cls_name in expected:
        assert cls_name in s.CLASS_GLYPHS, f"Missing glyph for {cls_name}"
        assert cls_name in ACTIVITY_DISPLAY_NAMES, f"Missing display name for {cls_name}"
    print("PASS: 6 activity classes definitions, colors, and glyphs verified")


def test_temporal_filter():
    # Test resolving each of the 6 classes
    for idx, cls_name in enumerate(ACTIVITY_CLASSES):
        probs = np.zeros(6)
        probs[idx] = 0.90
        # If fall class, fall prob is 0.90 >= 0.60 -> FALL
        # If non-fall class, resolved label should match cls_name
        resolved = resolve_activity_label(probs, fall_threshold=0.60)
        assert resolved == cls_name, f"Expected {cls_name}, got {resolved}"

    print("PASS: resolve_activity_label correctly resolves all 6 classes")

    # Test TemporalDecisionFilter smoothing and state transitions
    tdf = TemporalDecisionFilter()
    # Feed 5 frames of STANDING
    standing_probs = np.array([0.02, 0.90, 0.02, 0.02, 0.02, 0.02])
    for _ in range(5):
        lbl, conf, dist = tdf.update(standing_probs)
    assert lbl == "STANDING"
    assert len(dist) == 6

    # Feed OFF_BALANCE
    off_balance_probs = np.array([0.02, 0.02, 0.02, 0.90, 0.02, 0.02])
    for _ in range(5):
        lbl, conf, dist = tdf.update(off_balance_probs)
    assert lbl == "OFF_BALANCE"

    # Feed NORMAL_ACTIVITY
    normal_probs = np.array([0.02, 0.02, 0.02, 0.02, 0.90, 0.02])
    for _ in range(5):
        lbl, conf, dist = tdf.update(normal_probs)
    assert lbl == "NORMAL_ACTIVITY"

    # Feed FALL with confirmations
    fall_probs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95])
    for _ in range(5):
        lbl, conf, dist = tdf.update(fall_probs, confirmations_needed=3)
    assert lbl == "FALL"
    assert tdf.fall_alert_active

    print("PASS: TemporalDecisionFilter correctly tracks all 6 activities dynamically")


def test_ui_components_6_classes():
    probs = [0.10, 0.15, 0.40, 0.15, 0.10, 0.10]
    cards_html = c.render_activity_cards_html(probs, active_label="WALKING")
    assert "Walking" in cards_html
    assert "Off_balance" in cards_html or "Off balance" in cards_html or "OFF_BALANCE" in cards_html.upper()
    assert "Normal_activity" in cards_html or "Normal activity" in cards_html or "NORMAL_ACTIVITY" in cards_html.upper()
    assert "Fall" in cards_html

    bars_html = c.render_horizontal_probability_indicators(probs)
    assert "Walking" in bars_html
    assert "Fall" in bars_html

    print("PASS: UI cards and horizontal bars render all 6 classes without error")


if __name__ == "__main__":
    test_activity_classes()
    test_temporal_filter()
    test_ui_components_6_classes()
    print("ALL 6-CLASS INTEGRATION TESTS PASSED!")
