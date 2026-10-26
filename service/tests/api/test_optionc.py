"""Option C: additive `display` fields; every pre-existing response field stays byte-identical."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import optionc_harness as h
import pipeline.service as service
from pipeline.display import DISPLAY_VERSION, index

GOLDEN = json.loads((Path(__file__).with_name("optionc_golden_before.json")).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    return h.capture(tmp_path_factory.mktemp("optionc"))


def test_golden_case_sets_match(captured):
    assert set(captured) == set(GOLDEN) == {h.V1, h.PARTIAL2}
    for bundle in GOLDEN:
        assert set(captured[bundle]) == set(GOLDEN[bundle])


@pytest.mark.parametrize("bundle", [h.V1, h.PARTIAL2])
def test_responses_without_display_are_identical_to_golden(captured, bundle):
    for name, before in GOLDEN[bundle].items():
        assert h.canonical(h.strip(captured[bundle][name])) == h.canonical(before), name


def _check_display(body, bundle):
    display = body["display"]
    t_low, t_warn = body["thresholds"]["t_low"], body["thresholds"]["t_warn"]
    assert display["version"] == DISPLAY_VERSION
    assert display["is_probability"] is False
    assert display["file_index"] == index(body["file_score"], t_low, t_warn)
    assert [(s["start"], s["end"]) for s in display["slots"]] == [(s["start"], s["end"]) for s in body["slots"]]
    assert [s["index"] for s in display["slots"]] == [index(s["score"], t_low, t_warn) for s in body["slots"]]
    if bundle == h.V1:
        assert "partial" not in body and display["partial"] is None
        return
    partial, shown = body["partial"], display["partial"]
    assert shown["status"] == partial["status"]
    assert shown["window_count"] == len(body["windows"])
    assert shown["threshold_index"] == index(partial["threshold"], t_low, t_warn)
    assert [(s["start"], s["end"]) for s in shown["segments"]] == [(s["start"], s["end"]) for s in partial["segments"]]
    assert [s["peak_index"] for s in shown["segments"]] == [index(s["peak"], t_low, t_warn) for s in partial["segments"]]


@pytest.mark.parametrize("bundle", [h.V1, h.PARTIAL2])
def test_display_present_on_success_and_absent_on_cannot(captured, bundle):
    for name, record in captured[bundle].items():
        body = record["body"]
        if body["outcome_code"] == "cannot":
            assert "display" not in body, name
        else:
            _check_display(body, bundle)


def test_partial2_states_and_ranges(captured):
    cases = captured[h.PARTIAL2]
    expected = {"low_with_one_peak_n12": ("suspected", 1, "calibrated"), "low_with_several_peaks_n29": ("suspected", 3, "calibrated"),
                "warn_with_peak_n12": ("suspected", 1, "calibrated"), "uniform_warn_n12": ("not_found", 0, "calibrated"),
                "not_applicable_n3": ("not_applicable", 0, "not_applicable"), "count_n4_10s": ("not_found", 0, "extrapolated"),
                "count_n56_114s": ("not_found", 0, "calibrated"), "count_n59_120s": ("not_found", 0, "extrapolated"),
                "tail_peak_n59_120s": ("suspected", 1, "extrapolated")}
    for name, (status, segments, calibration_range) in expected.items():
        shown = cases[name]["body"]["display"]["partial"]
        assert (shown["status"], len(shown["segments"]), shown["calibration_range"]) == (status, segments, calibration_range), name
    assert [cases[f"count_n{n}_{s}s"]["body"]["display"]["partial"]["window_count"] for n, s in
            ((4, 10), (12, 26), (29, 60), (56, 114), (59, 120))] == [4, 12, 29, 56, 59]
    assert cases["warn_with_peak_n12"]["body"]["outcome_code"] == "warn"
    assert cases["low_with_one_peak_n12"]["body"]["outcome_code"] == "low"


def test_build_display_failure_keeps_analysis(tmp_path):
    spec = json.dumps(h.score_cases()["low_with_one_peak_n12"]).encode()
    with patch.object(service, "load_bundle", h.fake_load_bundle), patch.object(service, "prepare", h.fake_prepare):
        with h.client_for(h.PARTIAL2, tmp_path) as client:
            with patch.object(service, "build_display", side_effect=RuntimeError("boom")):
                response = client.post("/api/analyze", files={"file": ("case.wav", spec, "audio/wav")}, data={"analysis_consent": "true"})
    record = {"status": response.status_code, "body": response.json()}
    assert record["status"] == 200 and "display" not in record["body"]
    assert record["body"]["outcome_code"] == "low" and record["body"]["partial"]["status"] == "suspected"
    assert h.canonical(h.strip(record)) == h.canonical(GOLDEN[h.PARTIAL2]["low_with_one_peak_n12"])


def test_config_reports_display_version_and_keeps_keys(tmp_path):
    with patch.object(service, "load_bundle", h.fake_load_bundle), patch.object(service, "prepare", h.fake_prepare):
        with h.client_for(h.PARTIAL2, tmp_path) as client:
            config = client.get("/api/config").json()
    assert config["display_version"] == DISPLAY_VERSION
    assert config["partial_enabled"] is True
    assert {"max_upload_bytes", "max_duration_seconds", "donation_token_seconds", "consent_version", "consent_text",
            "bundle_id", "notice", "partial_enabled", "partial_scope"} < set(config)


def test_logs_do_not_contain_display_or_partial_values(tmp_path, caplog):
    caplog.set_level("DEBUG")
    spec = json.dumps(h.score_cases()["low_with_several_peaks_n29"]).encode()
    with patch.object(service, "load_bundle", h.fake_load_bundle), patch.object(service, "prepare", h.fake_prepare):
        with h.client_for(h.PARTIAL2, tmp_path) as client:
            body = client.post("/api/analyze", files={"file": ("private-recording.wav", spec, "audio/wav")},
                               data={"analysis_consent": "true"}).json()
    assert body["display"]["partial"]["status"] == "suspected"
    metadata = [record.message for record in caplog.records if record.name == "speech.metadata"]
    assert len(metadata) == 1
    assert set(json.loads(metadata[0])) == {"request_id", "duration", "outcome", "latency", "bundle_id"}
    text = "\n".join(record.getMessage() for record in caplog.records)
    for forbidden in ("display", "suspected", "segments", "index", "private-recording", "file_score", "statistic"):
        assert forbidden not in text
