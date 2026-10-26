"""In-process harness that drives the real pipeline.service._worker loop with synthetic window scores.

Each upload body is a JSON case spec, not audio. `prepare` and the detector are replaced so the
unchanged decide()/partial code and the service attachment path run exactly as in production.
Used by the Option C golden identity tests and by work/optionc/capture_golden.py.
"""
import json
import multiprocessing
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import pipeline.service as service
from app.main import create_app
from app.settings import Settings
from pipeline.bundle import load_bundle as real_load_bundle
from pipeline.frozen import audio_windows
from pipeline.preprocess import CannotAnalyze, PreparedAudio

ROOT = Path(__file__).resolve().parents[2]
SR = 16000
V1, PARTIAL2 = "bundles/nii-small-v1", "bundles/nii-small-v1-partial2"


class ScoreDetector:
    """Returns the synthetic scores carried in PreparedAudio.windows."""

    def score_windows(self, windows):
        return [float(value) for value in windows]


def fake_load_bundle(active_bundle, *, root, threads=4, initialize_detector=True):
    bundle = real_load_bundle(active_bundle, root=root, threads=threads, initialize_detector=False)
    bundle.detector = ScoreDetector()
    return bundle


def fake_prepare(body, *, ffmpeg=None, temp_dir=None, modules=None):
    spec = json.loads(body)
    if "reason" in spec:
        raise CannotAnalyze(spec["reason"])
    trimmed = spec["trimmed_samples"]
    starts = audio_windows.starts(trimmed)
    scores = spec["scores"]
    if len(scores) != len(starts):
        raise AssertionError("case spec score count does not match window count")
    duration = (spec.get("trim_start", 0) + trimmed + spec.get("trim_tail", 0)) / SR
    return PreparedAudio(windows=list(scores), starts=starts, duration=duration, trim_start=spec.get("trim_start", 0),
                         trim_end=spec.get("trim_start", 0) + trimmed, active_seconds=trimmed / SR, native_rate=SR)


class ThreadWorker:
    """AnalysisWorker stand-in: same pipe protocol, but _worker runs in a thread."""

    def __init__(self, root, active_bundle, threads, ffmpeg, uploads):
        self.connection, child = multiprocessing.Pipe()
        self.process = None
        self.thread = threading.Thread(target=service._worker, args=(child, str(root), active_bundle, threads, str(ffmpeg), str(uploads)), daemon=True)
        self.thread.start()
        if not self.connection.recv().get("ready"):
            raise RuntimeError("worker startup failed")

    def analyze(self, body, timeout):
        self.connection.send(body)
        if not self.connection.poll(timeout):
            raise CannotAnalyze("analysis_timeout")
        response = self.connection.recv()
        if "reason" in response:
            raise CannotAnalyze(response["reason"])
        return response["result"]

    def close(self):
        try:
            self.connection.send(None)
        except OSError:
            pass
        self.thread.join(timeout=5)
        self.connection.close()


def client_for(bundle, tmp_path):
    settings = replace(Settings(), root=ROOT, active_bundle=bundle, upload_dir=Path(tmp_path) / "uploads",
                       donation_dir=Path(tmp_path) / "donations")
    return TestClient(create_app(settings, worker_factory=ThreadWorker))


def patched():
    return [patch.object(service, "load_bundle", fake_load_bundle), patch.object(service, "prepare", fake_prepare)]


def _wave(n, base, amplitude):
    # Deterministic, non-constant synthetic scores.
    return [round(base + amplitude * ((7 * index) % 5 - 2) / 2, 6) for index in range(n)]


def _seconds(seconds):
    return int(round(seconds * SR))


def _n(trimmed):
    return len(audio_windows.starts(trimmed))


def score_cases():
    """Synthetic score cases; scores are chosen by fixed values, never by thresholds."""
    cases = {}
    t12, t29, t8 = _seconds(26), _seconds(60), _seconds(8)
    cases["uniform_warn_n12"] = dict(trimmed_samples=t12, scores=[0.0] * _n(t12))
    cases["uniform_defer_n12"] = dict(trimmed_samples=t12, scores=[-1.3] * _n(t12))
    cases["uniform_low_n12"] = dict(trimmed_samples=t12, scores=[-4.0] * _n(t12))
    cases["low_with_one_peak_n12"] = dict(trimmed_samples=t12, scores=[-4.0] * 5 + [3.0] + [-4.0] * (_n(t12) - 6))
    multi = [-4.0] * _n(t29)
    multi[3], multi[10], multi[11], multi[20] = 3.5, 3.0, 3.1, 4.2
    cases["low_with_several_peaks_n29"] = dict(trimmed_samples=t29, scores=multi)
    cases["warn_with_peak_n12"] = dict(trimmed_samples=t12, scores=[2.0] * 7 + [4.0] + [2.0] * (_n(t12) - 8))
    cases["not_applicable_n3"] = dict(trimmed_samples=t8, scores=[4.5, 4.0, 5.0])
    for seconds in (10, 26, 60, 114, 120):
        trimmed = _seconds(seconds)
        cases[f"count_n{_n(trimmed)}_{seconds}s"] = dict(trimmed_samples=trimmed, scores=_wave(_n(trimmed), -3.0, 2.5))
    for seconds in (10, 120):
        trimmed = _seconds(seconds)
        scores = [-3.5] * _n(trimmed)
        scores[-1] = 5.0
        cases[f"tail_peak_n{_n(trimmed)}_{seconds}s"] = dict(trimmed_samples=trimmed, scores=scores)
    trimmed = _seconds(26.5)
    cases["tail_window_offset_clock"] = dict(trimmed_samples=trimmed, trim_start=_seconds(3.2), trim_tail=_seconds(0.7),
                                            scores=_wave(_n(trimmed), -1.0, 3.0))
    return cases


WORKER_REASONS = ["decode_failed", "duration_out_of_range", "insufficient_trimmed_audio", "insufficient_active_speech", "invalid_audio"]


def run_cases(client):
    """Return {case_name: {"status": int, "body": dict}} for every golden case on one client."""
    out = {}

    def post(name, body, *, filename="case.wav", consent=True):
        response = client.post("/api/analyze", files={"file": (filename, body, "audio/wav")},
                               data={"analysis_consent": str(consent).lower(), "donation_opt_in": "false"})
        out[name] = {"status": response.status_code, "body": response.json()}

    for name, spec in score_cases().items():
        post(name, json.dumps(spec).encode())
    for reason in WORKER_REASONS:
        post("cannot_" + reason, json.dumps({"reason": reason}).encode())
    trimmed = _seconds(10)
    post("cannot_analysis_failed", json.dumps(dict(trimmed_samples=trimmed, scores=[float("nan")] * _n(trimmed))).encode())
    post("cannot_analysis_consent_required", json.dumps(score_cases()["uniform_warn_n12"]).encode(), consent=False)
    post("cannot_unsupported_format", b"{}", filename="case.txt")
    response = client.post("/api/analyze", content=b"", headers={"content-length": str(30 * 1024 * 1024)})
    out["cannot_file_too_large"] = {"status": response.status_code, "body": response.json()}
    response = client.post("/api/analyze")
    out["cannot_invalid_request"] = {"status": response.status_code, "body": response.json()}
    client.app.state.inflight_uploads = 8
    post("cannot_server_busy", b"{}")
    client.app.state.inflight_uploads = 0
    return out


def strip(record):
    """Drop per-request random and Option C additive fields before identity comparison."""
    body = {key: value for key, value in record["body"].items() if key not in ("request_id", "display")}
    return {"status": record["status"], "body": body}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def capture(tmp_path):
    result = {}
    patches = patched()
    for item in patches:
        item.start()
    try:
        for bundle in (V1, PARTIAL2):
            with client_for(bundle, Path(tmp_path) / bundle.split("/")[-1]) as client:
                result[bundle] = run_cases(client)
    finally:
        for item in patches:
            item.stop()
    return result
