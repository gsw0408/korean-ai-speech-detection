import copy
import math
from pathlib import Path

import pytest

from pipeline.bundle import load_bundle
from pipeline.decision import classify
from pipeline.display import BANDS, CEIL, DISPLAY_VERSION, FLOOR, build_display, index

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def v1():
    return load_bundle("bundles/nii-small-v1", root=ROOT, initialize_detector=False)


@pytest.fixture(scope="module")
def partial2():
    return load_bundle("bundles/nii-small-v1-partial2", root=ROOT, initialize_detector=False)


def thresholds(bundle):
    return bundle.calibration["t_low"], bundle.calibration["t_warn"]


def test_anchors_map_exactly(v1):
    t_low, t_warn = thresholds(v1)
    assert [index(score, t_low, t_warn) for score in (FLOOR, t_low, t_warn, CEIL)] == [0, 33, 67, 100]


def test_clipping_below_floor_and_above_ceiling(v1):
    t_low, t_warn = thresholds(v1)
    assert index(FLOOR - 0.5, t_low, t_warn) == index(-1e6, t_low, t_warn) == 0
    assert index(CEIL + 0.5, t_low, t_warn) == index(1e6, t_low, t_warn) == 100


def test_monotonic_over_dense_grid(v1):
    t_low, t_warn = thresholds(v1)
    values = [index(-8 + 16 * step / 20000, t_low, t_warn) for step in range(20001)]
    assert all(isinstance(value, int) and 0 <= value <= 100 for value in values)
    assert all(a <= b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("delta", [-1e-3, -1e-9, 0.0, 1e-9, 1e-3])
@pytest.mark.parametrize("anchor", ["t_low", "t_warn"])
def test_band_clamp_matches_classify(v1, anchor, delta):
    t_low, t_warn = thresholds(v1)
    score = v1.calibration[anchor] + delta
    low, high = BANDS[classify(score, t_low, t_warn)]
    assert low <= index(score, t_low, t_warn) <= high


def test_warn_boundary_is_exact(v1):
    t_low, t_warn = thresholds(v1)
    assert index(t_warn - 1e-9, t_low, t_warn) <= 66
    assert index(t_warn, t_low, t_warn) >= 67
    assert index(t_low, t_low, t_warn) <= 33
    assert index(t_low + 1e-9, t_low, t_warn) >= 34


@pytest.mark.parametrize("score", [None, math.nan, math.inf, -math.inf, "1.0", True])
def test_invalid_scores_give_none(v1, score):
    assert index(score, *thresholds(v1)) is None


@pytest.mark.parametrize("t_low,t_warn", [(-1.0, -2.0), (-1.0, -1.0), (FLOOR, -0.5), (-7.0, -0.5), (-2.0, CEIL), (-2.0, 6.0),
                                          (math.nan, -0.5), (-2.0, math.inf), (None, -0.5)])
def test_invalid_thresholds_raise(t_low, t_warn):
    with pytest.raises(ValueError):
        index(0.0, t_low, t_warn)


# Rows from the Option C prompt; values recomputed by the formula, outcome by classify().
# Discrepancy: the prompt listed -2.132 as low/33, but -2.132 > t_low (-2.13208...) so it is defer/34.
EXAMPLES = [(-6.2, "low", 0), (-5.0, "low", 9), (-3.88, "low", 18), (-3.23, "low", 24), (-2.1321, "low", 33), (-2.132, "defer", 34),
            (-2.0, "defer", 36), (-1.5, "defer", 46), (-1.0, "defer", 56), (-0.6, "defer", 65), (-0.501, "warn", 67),
            (0.0, "warn", 69), (1.0, "warn", 75), (2.0, "warn", 81), (3.86, "warn", 91), (5.0, "warn", 97),
            (5.5, "warn", 100), (5.8, "warn", 100)]


@pytest.mark.parametrize("score,outcome,expected", EXAMPLES)
def test_example_table(v1, score, outcome, expected):
    t_low, t_warn = thresholds(v1)
    assert classify(score, t_low, t_warn) == outcome
    assert index(score, t_low, t_warn) == expected


def result(count, scores=None, partial=None):
    scores = scores or [-3.0] * count
    windows = [{"start": 2.0 * i, "end": 2.0 * i + 4, "score": s} for i, s in enumerate(scores)]
    slots = [{"start": 2 * i, "end": 2 * i + 2, "score": s, "outcome_code": "low", "outcome": "x", "indicative_only": True}
             for i, s in enumerate(scores + [None])]
    out = {"file_score": -3.0, "windows": windows, "slots": slots}
    if partial is not None:
        out["partial"] = partial
    return out


def test_build_display_slots_and_null_partial(v1):
    data = result(5, [-5.0, None, 0.0, 5.8, -2.0])
    data["file_score"] = -1.0
    display = build_display(data, v1.calibration, v1.partial)
    assert display["version"] == DISPLAY_VERSION
    assert display["is_probability"] is False
    assert display["file_index"] == 56
    assert [(s["start"], s["end"]) for s in display["slots"]] == [(s["start"], s["end"]) for s in data["slots"]]
    assert [s["index"] for s in display["slots"]] == [9, None, 69, 100, 36, None]
    assert display["partial"] is None


@pytest.mark.parametrize("count,expected", [(3, "not_applicable"), (4, "extrapolated"), (11, "extrapolated"), (12, "calibrated"),
                                            (56, "calibrated"), (57, "extrapolated"), (60, "extrapolated"), (61, "not_applicable")])
def test_calibration_range_from_partial2_config(partial2, count, expected):
    partial = {"status": "not_found", "statistic": -3.0, "threshold": 1.0, "segments": []}
    display = build_display(result(count, partial=partial), partial2.calibration, partial2.partial)
    assert display["partial"]["window_count"] == count
    assert display["partial"]["calibration_range"] == expected


def test_partial_status_and_segments_are_copied(partial2):
    segments = [{"start": 4.5, "end": 8.5, "peak": 3.0}, {"start": 20.0, "end": 26.0, "peak": 4.2}]
    partial = {"status": "suspected", "statistic": 4.2, "threshold": 2.426, "segments": segments,
               "scope_ko": "s", "evaluation_ko": "e", "localization": "window_4s_indicative"}
    data = result(29, partial=partial)
    before = copy.deepcopy(data)
    display = build_display(data, partial2.calibration, partial2.partial)
    assert data == before
    t_low, t_warn = thresholds(partial2)
    assert display["partial"]["status"] == "suspected"
    assert display["partial"]["threshold_index"] == index(2.426, t_low, t_warn)
    assert [(s["start"], s["end"]) for s in display["partial"]["segments"]] == [(4.5, 8.5), (20.0, 26.0)]
    assert [s["peak_index"] for s in display["partial"]["segments"]] == [index(3.0, t_low, t_warn), index(4.2, t_low, t_warn)]


def test_not_applicable_partial_has_null_threshold_index(partial2):
    partial = {"status": "not_applicable", "statistic": None, "threshold": None, "segments": []}
    display = build_display(result(3, partial=partial), partial2.calibration, partial2.partial)
    assert display["partial"] == {"status": "not_applicable", "window_count": 3, "calibration_range": "not_applicable",
                                  "threshold_index": None, "segments": []}


def test_display_module_has_no_threshold_literals():
    source = (ROOT / "pipeline/display.py").read_text(encoding="utf-8")
    for literal in ("2.132", "0.501", "0.8706", "0.5058", "0.955"):
        assert literal not in source
