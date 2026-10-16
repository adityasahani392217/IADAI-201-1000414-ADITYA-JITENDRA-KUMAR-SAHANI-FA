"""
core/temporal_filter.py
=======================
Temporal decision filtering, exponential smoothing, and sliding-window aggregation.
Prevents false-alarm flickering through hysteresis gating and consecutive frame validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.kinematics import ACTIVITY_CLASSES


def resolve_activity_label(probabilities: np.ndarray, fall_threshold: float = 0.60) -> str:
    """Resolve single discrete activity label given class probabilities and fall sensitivity."""
    probs = np.asarray(probabilities, dtype=float)
    fall_idx = ACTIVITY_CLASSES.index("FALL")
    if probs[fall_idx] >= fall_threshold:
        return "FALL"
    # Otherwise select argmax among non-fall postures
    non_fall_indices = [i for i, c in enumerate(ACTIVITY_CLASSES) if c != "FALL"]
    non_fall_probs = probs[non_fall_indices]
    best_idx = non_fall_indices[int(np.argmax(non_fall_probs))]
    return ACTIVITY_CLASSES[best_idx]


def detect_sustained_fall_event(
    fall_probabilities: np.ndarray,
    fall_threshold: float,
    min_consecutive_windows: int = 2
) -> Optional[Tuple[int, int]]:
    """
    Locate the initial interval of consecutive temporal windows meeting or exceeding
    the specified fall threshold: returns (start_idx, end_idx) or None.
    """
    fp = np.asarray(fall_probabilities, dtype=float)
    needed = max(1, min(min_consecutive_windows, len(fp)))
    streak_count = 0
    start_index = 0

    for idx, prob in enumerate(fp):
        if prob >= fall_threshold:
            if streak_count == 0:
                start_index = idx
            streak_count += 1
            if streak_count >= needed:
                end_index = idx
                while end_index + 1 < len(fp) and fp[end_index + 1] >= fall_threshold:
                    end_index += 1
                return start_index, end_index
        else:
            streak_count = 0

    return None


def aggregate_detection_intervals(
    window_probabilities: np.ndarray,
    fall_threshold: float = 0.60,
    min_consecutive_windows: int = 2
) -> Dict[str, Any]:
    """
    Summarize a temporal sequence of window probability distributions into a definitive
    comprehensive video verdict, activity distribution, and peak frame indices.
    """
    matrix = np.asarray(window_probabilities, dtype=float)
    total_windows = len(matrix)
    fall_idx = ACTIVITY_CLASSES.index("FALL")
    num_classes = len(ACTIVITY_CLASSES)
    non_fall_indices = [i for i in range(num_classes) if i != fall_idx]

    top_non_fall_local = np.argmax(matrix[:, non_fall_indices], axis=1)
    top_non_fall = np.array([non_fall_indices[idx] for idx in top_non_fall_local])

    discrete_labels = np.where(matrix[:, fall_idx] >= fall_threshold, fall_idx, top_non_fall)
    activity_votes = np.bincount(discrete_labels, minlength=num_classes) / max(total_windows, 1)

    sustained_fall = detect_sustained_fall_event(matrix[:, fall_idx], fall_threshold, min_consecutive_windows)

    if sustained_fall is not None:
        start_w, end_w = sustained_fall
        fall_segment = matrix[start_w:end_w + 1]
        overall_dist = fall_segment.mean(axis=0)
        peak_offset = int(np.argmax(fall_segment[:, fall_idx]))
        peak_window_idx = start_w + peak_offset
        verdict = "FALL"
        confidence = float(overall_dist[fall_idx])
    else:
        vote_counts = np.bincount(top_non_fall, minlength=num_classes)
        dominant_idx = int(np.argmax(vote_counts))
        verdict = ACTIVITY_CLASSES[dominant_idx]
        matching_rows = matrix[top_non_fall == dominant_idx]
        confidence = float(matching_rows[:, dominant_idx].mean()) if len(matching_rows) > 0 else 0.0
        overall_dist = matrix.mean(axis=0)
        peak_window_idx = int(np.argmax(np.where(top_non_fall == dominant_idx, matrix[:, dominant_idx], -1.0)))

    return {
        "verdict": verdict,
        "confidence": confidence,
        "distribution": overall_dist,
        "sustained_fall_run": sustained_fall,
        "activity_votes": activity_votes,
        "peak_window_idx": peak_window_idx
    }


class TemporalDecisionFilter:
    """
    Online streaming decision smoother applying Exponential Moving Average (EMA)
    and hysteresis state transitions for live video feeds.
    """

    def __init__(self):
        self.ema_distribution: Optional[np.ndarray] = None
        self.fall_streak: int = 0
        self.calm_streak: int = 0
        self.fall_alert_active: bool = False

    def reset(self) -> None:
        """Reset temporal filter memory."""
        self.ema_distribution = None
        self.fall_streak = 0
        self.calm_streak = 0
        self.fall_alert_active = False

    def update(
        self,
        raw_probabilities: np.ndarray,
        smoothing_alpha: float = 0.65,
        fall_threshold: float = 0.60,
        confirmations_needed: int = 2,
        release_frames: int = 6
    ) -> Tuple[str, float, np.ndarray]:

        """
        Ingest current frame predictions, apply EMA smoothing, update hysteresis,
        and return (filtered_label, filtered_confidence, smoothed_distribution).
        """
        raw = np.asarray(raw_probabilities, dtype=float)
        normalized = raw / max(float(raw.sum()), 1e-9)

        if self.ema_distribution is None:
            self.ema_distribution = normalized
        else:
            self.ema_distribution = (smoothing_alpha * normalized) + ((1.0 - smoothing_alpha) * self.ema_distribution)

        smoothed = self.ema_distribution

        fall_idx = ACTIVITY_CLASSES.index("FALL")
        # Evaluate fall condition
        if smoothed[fall_idx] >= fall_threshold:
            self.fall_streak += 1
            self.calm_streak = 0
        else:
            self.fall_streak = 0
            self.calm_streak += 1

        # State transition: activate alert upon sufficient confirmation streak
        if not self.fall_alert_active and self.fall_streak >= confirmations_needed:
            self.fall_alert_active = True

        # State transition: clear alert only after prolonged calm period (hysteresis)
        if self.fall_alert_active and self.calm_streak >= release_frames:
            self.fall_alert_active = False

        if self.fall_alert_active:
            resolved_label = "FALL"
        else:
            non_fall_indices = [i for i, c in enumerate(ACTIVITY_CLASSES) if c != "FALL"]
            best_idx = non_fall_indices[int(np.argmax(smoothed[non_fall_indices]))]
            resolved_label = ACTIVITY_CLASSES[best_idx]

        class_idx = ACTIVITY_CLASSES.index(resolved_label)
        resolved_confidence = float(smoothed[class_idx])

        return resolved_label, resolved_confidence, smoothed.copy()
