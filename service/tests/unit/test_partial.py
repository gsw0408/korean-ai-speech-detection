import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.decision import decide
from pipeline.display import build_display
from pipeline.partial import analyze_partial, validate_partial
from pipeline.preprocess import PreparedAudio
from pipeline.service import _worker


HASHES = {key: str(index) * 64 for index, key in enumerate(("detector", "frontend", "audio_windows"), 1)}


@pytest.fixture
def cfg():
    return {"schema": "partial_peak_v1", "statistic": "max_window_score",
            "detector_sha256": HASHES["detector"],
            "preprocessing_sha256s": {key: HASHES[key] for key in ("frontend", "audio_windows")},
            "bins": [{"min_windows": low, "max_windows": high, "threshold": threshold}
                     for low, high, threshold in ((4, 15, 2.0), (16, 30, 3.0), (31, 60, 4.0))],
            "scope_ko": "unit fixture scope", "evaluation_ko": "unit fixture evaluation"}


def windows(scores):
    return [{"start": 2 * index + 0.5, "end": 2 * index + 4.5, "score": score}
            for index, score in enumerate(scores)]


@pytest.mark.parametrize("count,threshold", [(4, 2.0), (15, 2.0), (16, 3.0), (30, 3.0), (31, 4.0), (60, 4.0)])
def test_bin_boundaries_and_inclusive_threshold(cfg, count, threshold):
    result = analyze_partial(windows([threshold] + [-1] * (count - 1)), cfg)
    assert result["threshold"] == threshold
    assert result["statistic"] == threshold
    assert result["status"] == "suspected"
    assert result["segments"] == [{"start": 0.5, "end": 4.5, "peak": threshold}]
    assert result["localization"] == "window_4s_indicative"
    assert result["scope_ko"] == cfg["scope_ko"]
    assert result["evaluation_ko"] == cfg["evaluation_ko"]


def test_not_found(cfg):
    result = analyze_partial(windows([1.9, -2, -1, 0]), cfg)
    assert result["status"] == "not_found"
    assert result["statistic"] == 1.9
    assert result["threshold"] == 2.0
    assert result["segments"] == []


@pytest.mark.parametrize("count", [0, 1, 3, 61])
def test_not_applicable(cfg, count):
    result = analyze_partial(windows([10] * count), cfg)
    assert result["status"] == "not_applicable"
    assert result["statistic"] is None and result["threshold"] is None
    assert result["segments"] == []


@pytest.mark.parametrize("scores,expected", [
    ([2, 3, 4, -1], [{"start": 0.5, "end": 8.5, "peak": 4}]),
    ([2, -1, 3, -1], [{"start": 0.5, "end": 8.5, "peak": 3}]),
    ([2, -1, -1, 3], [{"start": 0.5, "end": 4.5, "peak": 2}, {"start": 6.5, "end": 10.5, "peak": 3}]),
])
def test_segment_merging_preserves_original_clock(cfg, scores, expected):
    source = windows(scores)
    before = copy.deepcopy((source, cfg))
    assert analyze_partial(source, cfg)["segments"] == expected
    assert (source, cfg) == before


def test_valid_contract(cfg):
    validate_partial(cfg, HASHES)


@pytest.mark.parametrize("field,value", [("schema", "other"), ("statistic", "mean_window_score")])
def test_unsupported_contract(cfg, field, value):
    cfg[field] = value
    with pytest.raises(ValueError, match="Unsupported partial contract"):
        validate_partial(cfg, HASHES)


@pytest.mark.parametrize("field", ["detector", "frontend", "audio_windows"])
def test_binding_mismatch(cfg, field):
    if field == "detector":
        cfg["detector_sha256"] = "0" * 64
    else:
        cfg["preprocessing_sha256s"][field] = "0" * 64
    with pytest.raises(ValueError, match="Partial binding mismatch"):
        validate_partial(cfg, HASHES)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf"), "2", None, True, 10 ** 400])
def test_invalid_threshold(cfg, threshold):
    cfg["bins"][0]["threshold"] = threshold
    with pytest.raises(ValueError, match="Invalid partial bins"):
        validate_partial(cfg, HASHES)


@pytest.mark.parametrize("index,field,value", [
    (0, "min_windows", 3), (0, "max_windows", 3), (0, "min_windows", 4.0),
    (0, "min_windows", True), (1, "min_windows", 17), (1, "min_windows", 15),
    (2, "max_windows", 59), (2, "max_windows", 61),
])
def test_invalid_bin_boundaries(cfg, index, field, value):
    cfg["bins"][index][field] = value
    with pytest.raises(ValueError, match="Invalid partial bins"):
        validate_partial(cfg, HASHES)


@pytest.mark.parametrize("bins", [[], None, {}, [None]])
def test_invalid_bins_structure(cfg, bins):
    cfg["bins"] = bins
    with pytest.raises(ValueError, match="Invalid partial bins"):
        validate_partial(cfg, HASHES)


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_keeps_v1_decision_and_runs_detector_once(cfg, enabled, monkeypatch, tmp_path):
    prepared = PreparedAudio([None] * 4, [0, 32000, 64000, 96000], 10, 0, 160000, 10, 16000)
    scores = [-3, 2.5, 2, -3]
    expected = decide(prepared, scores, -1, 1)
    bundle = SimpleNamespace(detector=Mock(), preprocessing=None,
                             calibration={"t_low": -1, "t_warn": 1}, partial=cfg if enabled else None)
    bundle.detector.score_windows.return_value = scores
    connection = Mock()
    connection.recv.side_effect = [b"synthetic fixture", None]
    monkeypatch.setattr("pipeline.service.load_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr("pipeline.service.prepare", lambda *args, **kwargs: prepared)
    _worker(connection, tmp_path, "bundles/test", 1, "fixture-ffmpeg", tmp_path / "uploads")
    assert connection.send.call_args_list[0].args[0] == {"ready": True}
    actual = connection.send.call_args_list[1].args[0]["result"]
    # Option C: `display` is additive presentation; everything else must equal the v1 decision exactly.
    display = actual.pop("display")
    if enabled:
        assert actual.pop("partial") == analyze_partial(expected["windows"], cfg)
        expected_display = build_display({**expected, "partial": analyze_partial(expected["windows"], cfg)}, bundle.calibration, cfg)
    else:
        expected_display = build_display(expected, bundle.calibration, None)
    assert display == expected_display and display["is_probability"] is False
    assert actual == expected
    bundle.detector.score_windows.assert_called_once_with(prepared.windows)
