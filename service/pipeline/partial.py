"""Experimental peak-window screen using manifest-bound insertion thresholds."""

import math
import re

import numpy as np


SCHEMA = "partial_peak_v1"
NR_SCHEMA = "partial_v2"
FRAME_CONTRACT = {
    "step_seconds": 0.02, "dtype": "float16", "smoothing_frames": 50,
    "smoothing_alignment": "valid 50-frame mean; segment centers at timeline_start+24.5*0.02",
    "window_frame_clock": "(trim_start + window_start +320*t+200)/16000",
    "quantization": "nearest20 ms using one rounded scalar window offset; average overlapping windows before float16 storage"}


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate_partial_v2(cfg, hashes):
    required = {"schema", "candidate", "parameters", "n_unit", "supported_windows", "calibration_windows",
                "detector_sha256", "preprocessing_sha256s", "selection_sha256", "evaluation_sha256",
                "verdict", "scope_ko", "evaluation_ko", "frame_timeline_contract"}
    if (not isinstance(cfg, dict) or cfg.get("schema") != NR_SCHEMA or set(cfg) != required
            or cfg["candidate"] not in ("W0", "W1", "W2", "W3", "F1", "F2")):
        raise ValueError("Unsupported no-retraining partial contract")
    if cfg["detector_sha256"] != hashes["detector"] or cfg["preprocessing_sha256s"] != {
            key: hashes[key] for key in ("frontend", "audio_windows")}:
        raise ValueError("Partial binding mismatch")
    parameters = cfg["parameters"]
    if (not isinstance(parameters, dict) or set(parameters) != {"b", "c_q", "q"}
            or not all(_finite_number(value) for value in parameters.values())
            or parameters["b"] < 0 or not any(math.isclose(parameters["q"], 0.9 + 0.005 * index, rel_tol=0, abs_tol=1e-12)
                                              for index in range(20))):
        raise ValueError("Invalid no-retraining threshold parameters")
    expected_unit = "duration_seconds/2" if cfg["candidate"].startswith("F") else "4 s window count"
    if (cfg["n_unit"] != expected_unit
            or cfg["supported_windows"] != {"min": 4, "max": 60}
            or cfg["calibration_windows"] != {"min": 12, "max": 56}
            or any(type(value) is not int for key in ("supported_windows", "calibration_windows") for value in cfg[key].values())
            or cfg["frame_timeline_contract"] != FRAME_CONTRACT
            or cfg["verdict"] not in ("PASS-FULL", "PASS-FPR")
            or any(not isinstance(cfg[key], str) or not re.fullmatch(r"[0-9a-f]{64}", cfg[key])
                   for key in ("selection_sha256", "evaluation_sha256"))
            or any(not isinstance(cfg[key], str) or not cfg[key] for key in ("scope_ko", "evaluation_ko"))):
        raise ValueError("Invalid no-retraining provenance or scope")


def _merge_nr(segments, gap=0):
    merged = []
    for start, end, peak in segments:
        if merged and (start <= merged[-1]["end"] if gap == 0 else start - merged[-1]["end"] < gap - 1e-12):
            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["peak"] = max(merged[-1]["peak"], peak)
        else:
            merged.append(dict(start=float(start), end=float(end), peak=float(peak)))
    return merged


def _frame_timeline(frame_scores, starts, trim_start):
    values = np.asarray(frame_scores, dtype=float)
    if (values.shape != (len(starts), 199) or not np.isfinite(values).all()
            or not len(starts) or any(type(start) is not int or start < 0 for start in starts)
            or type(trim_start) is not int or trim_start < 0):
        raise ValueError("Invalid no-retraining frame scores")
    indices = [int(np.floor(start / 320 + 0.5)) + np.arange(199) for start in starts]
    length = max(int(index[-1]) for index in indices) + 1
    total, count = np.zeros(length), np.zeros(length, dtype=np.int32)
    for index, frame in zip(indices, values):
        np.add.at(total, index, frame)
        np.add.at(count, index, 1)
    if not np.all(count > 0):
        raise ValueError("Incomplete no-retraining frame timeline")
    return (total / count).astype(np.float16).astype(float), (trim_start + 200) / 16000


def analyze_partial_v2(windows, cfg, *, duration, frame_scores=None, starts=None, trim_start=None):
    candidate = cfg["candidate"]
    base = dict(scope_ko=cfg["scope_ko"], evaluation_ko=cfg["evaluation_ko"],
                localization="frame_smoothed_1s_indicative" if candidate.startswith("F") else "window_4s_indicative")
    supported = cfg["supported_windows"]
    if not supported["min"] <= len(windows) <= supported["max"]:
        return dict(base, status="not_applicable", statistic=None, threshold=None, segments=[])
    scores = np.asarray([window["score"] for window in windows], dtype=float)
    if not np.isfinite(scores).all() or not _finite_number(duration) or duration <= 0:
        raise ValueError("Invalid no-retraining inputs")
    parameters = cfg["parameters"]
    if candidate.startswith("W"):
        n = len(windows)
        adjusted = scores - np.median(scores) if candidate in ("W2", "W3") else scores
        cut = parameters["c_q"] + parameters["b"] * math.log(n)
        if candidate in ("W1", "W3"):
            statistic = float(np.minimum(adjusted[:-1], adjusted[1:]).max())
            crossings = adjusted >= cut
            pairs = crossings[:-1] & crossings[1:]
            flags = np.zeros(n, dtype=bool)
            flags[:-1] |= pairs
            flags[1:] |= pairs
        else:
            statistic = float(adjusted.max())
            flags = adjusted >= cut
        segments = _merge_nr([(window["start"], min(window["start"] + 4, duration), value)
                              for window, value, flag in zip(windows, adjusted, flags) if flag])
    else:
        timeline, origin = _frame_timeline(frame_scores, starts, trim_start)
        smooth = np.convolve(timeline, np.full(50, 1 / 50), mode="valid")
        if candidate == "F2":
            smooth = smooth - np.median(timeline)
        statistic = float(smooth.max())
        cut = parameters["c_q"] + parameters["b"] * math.log(duration / 2)
        edges = np.diff(np.concatenate(([False], smooth >= cut, [False])).astype(int))
        left, right = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        centers = origin + 24.5 * 0.02
        segments = _merge_nr([(max(0, centers + start * 0.02 - 0.01),
                               min(duration, centers + end * 0.02 - 0.01), float(smooth[start:end].max()))
                              for start, end in zip(left, right) if (end - start) * 0.02 >= 0.5 - 1e-12], gap=0.3)
    return dict(base, status="suspected" if statistic >= cut else "not_found", statistic=statistic,
                threshold=cut, segments=segments)


def validate_partial(cfg, hashes):
    if isinstance(cfg, dict) and cfg.get("schema") == NR_SCHEMA:
        return validate_partial_v2(cfg, hashes)
    if not isinstance(cfg, dict) or cfg.get("schema") != SCHEMA or cfg.get("statistic") != "max_window_score":
        raise ValueError("Unsupported partial contract")
    if cfg.get("detector_sha256") != hashes["detector"] or cfg.get("preprocessing_sha256s") != {
            key: hashes[key] for key in ("frontend", "audio_windows")}:
        raise ValueError("Partial binding mismatch")
    bins = cfg.get("bins")
    if not isinstance(bins, list) or not bins:
        raise ValueError("Invalid partial bins")
    previous = 3
    for band in bins:
        if not isinstance(band, dict):
            raise ValueError("Invalid partial bins")
        low, high, threshold = (band.get(key) for key in ("min_windows", "max_windows", "threshold"))
        try:
            valid_threshold = type(threshold) in (int, float) and math.isfinite(threshold)
        except OverflowError:
            valid_threshold = False
        if (type(low) is not int or type(high) is not int or low != previous + 1 or high < low or high > 60
                or not valid_threshold):
            raise ValueError("Invalid partial bins")
        previous = high
    if previous != 60:
        raise ValueError("Invalid partial bins")


def analyze_partial(windows, cfg, *, duration=None, frame_scores=None, starts=None, trim_start=None):
    if cfg.get("schema") == NR_SCHEMA:
        return analyze_partial_v2(windows, cfg, duration=duration, frame_scores=frame_scores,
                                  starts=starts, trim_start=trim_start)
    base = {"scope_ko": cfg.get("scope_ko"), "evaluation_ko": cfg.get("evaluation_ko"),
            "localization": "window_4s_indicative"}
    band = next((band for band in cfg["bins"] if band["min_windows"] <= len(windows) <= band["max_windows"]), None)
    if band is None:
        return dict(base, status="not_applicable", statistic=None, threshold=None, segments=[])
    threshold, segments = band["threshold"], []
    for window in windows:
        if window["score"] >= threshold:
            if segments and window["start"] <= segments[-1]["end"]:
                segments[-1]["end"] = max(segments[-1]["end"], window["end"])
                segments[-1]["peak"] = max(segments[-1]["peak"], window["score"])
            else:
                segments.append({"start": window["start"], "end": window["end"], "peak": window["score"]})
    peak = max(window["score"] for window in windows)
    return dict(base, status="suspected" if peak >= threshold else "not_found",
                statistic=peak, threshold=threshold, segments=segments)
