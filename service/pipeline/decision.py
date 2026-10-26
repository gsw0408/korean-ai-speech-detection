"""File mean and indicative two-second slots on the original file clock."""

import math

import numpy as np


OUTCOMES = {
    "warn": "AI 음성 의심 — 추가 확인 필요",
    "low": "AI 음성 증거가 낮음",
    "defer": "판단 유보",
    "cannot": "분석 불가",
}


def classify(score, t_low, t_warn):
    if not math.isfinite(t_low) or not math.isfinite(t_warn) or t_low >= t_warn:
        raise ValueError("Invalid decision thresholds")
    if score is None or not math.isfinite(score):
        return "cannot"
    if score >= t_warn:
        return "warn"
    if score <= t_low:
        return "low"
    return "defer"


def decide(prepared, scores, t_low, t_warn):
    if len(scores) != len(prepared.starts) or not scores or not np.isfinite(scores).all():
        raise ValueError("Invalid window scores")
    file_score = float(np.mean(scores))
    outcome = classify(file_score, t_low, t_warn)
    windows = [{"start": (prepared.trim_start + start) / 16000,
                "end": (prepared.trim_start + start + 64000) / 16000, "score": float(score)}
               for start, score in zip(prepared.starts, scores)]
    slots = []
    for index in range(math.ceil(prepared.duration / 2)):
        start, end = 2 * index, min(2 * index + 2, prepared.duration)
        covering = [window["score"] for window in windows
                    if window["start"] < end and window["end"] > start]
        score = max(covering) if covering else None
        code = classify(score, t_low, t_warn)
        slots.append({"start": start, "end": end, "score": score, "outcome_code": code,
                      "outcome": OUTCOMES[code], "indicative_only": True})
    return {"outcome_code": outcome, "outcome": OUTCOMES[outcome], "file_score": file_score,
            "thresholds": {"t_low": t_low, "t_warn": t_warn}, "slots": slots, "windows": windows,
            "duration": prepared.duration, "timeline_notice": "indicative only"}
