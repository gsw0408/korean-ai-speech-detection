"""Option C presentation values: a 0-100 trace index that is not a probability.

Pure functions over an existing decide()/partial result. Nothing here changes scores, thresholds,
outcomes or the partial verdict; thresholds always come from the loaded bundle.
"""

import math

from .decision import classify


DISPLAY_VERSION = "display-v1-20261017"
# Presentation anchors from TRAIN single-clip scores (work/nr/train_clip_scores.jsonl):
# observed minimum about -6.07, AI 99th percentile about 5.51. Not fitted on DEV.
FLOOR = -6.0
CEIL = 5.5
BANDS = {"low": (0, 33), "defer": (34, 66), "warn": (67, 100)}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate(t_low, t_warn):
    values = (t_low, t_warn)
    if not all(_number(value) and math.isfinite(value) for value in values):
        raise ValueError("Invalid display thresholds")
    if not FLOOR < t_low < t_warn < CEIL:
        raise ValueError("Display thresholds outside presentation anchors")


def index(score, t_low, t_warn):
    _validate(t_low, t_warn)
    if not _number(score) or not math.isfinite(score):
        return None
    points = ((FLOOR, 0.0), (t_low, 100 / 3), (t_warn, 200 / 3), (CEIL, 100.0))
    if score <= FLOOR:
        value = 0.0
    elif score >= CEIL:
        value = 100.0
    else:
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if score <= x1:
                value = y0 + (y1 - y0) * (score - x0) / (x1 - x0)
                break
    low, high = BANDS[classify(score, t_low, t_warn)]
    return min(high, max(low, math.floor(value + 0.5)))


def _calibration_range(count, cfg):
    supported, calibrated = cfg.get("supported_windows"), cfg.get("calibration_windows")
    if supported is None or calibrated is None:
        return None  # legacy partial_peak_v1 bundle has no calibrated window range
    if not supported["min"] <= count <= supported["max"]:
        return "not_applicable"
    return "calibrated" if calibrated["min"] <= count <= calibrated["max"] else "extrapolated"


def build_display(result, calibration, partial_cfg):
    t_low, t_warn = calibration["t_low"], calibration["t_warn"]
    display = {"version": DISPLAY_VERSION, "is_probability": False,
               "file_index": index(result["file_score"], t_low, t_warn),
               "slots": [{"start": slot["start"], "end": slot["end"], "index": index(slot["score"], t_low, t_warn)}
                         for slot in result["slots"]],
               "partial": None}
    partial = result.get("partial")
    if partial is not None:
        count = len(result["windows"])
        display["partial"] = {
            "status": partial["status"], "window_count": count,
            "calibration_range": _calibration_range(count, partial_cfg),
            "threshold_index": index(partial["threshold"], t_low, t_warn),
            "segments": [{"start": segment["start"], "end": segment["end"],
                          "peak_index": index(segment["peak"], t_low, t_warn)} for segment in partial["segments"]]}
    return display
