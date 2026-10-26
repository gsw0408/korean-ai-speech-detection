import concurrent.futures
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.settings import Settings
from donations.store import CONSENT_VERSION
from pipeline.decision import decide
from pipeline.preprocess import CannotAnalyze, prepare
from conftest import wav_bytes

ROOT = Path(__file__).resolve().parents[2]


class FakeWorker:
    score = 0
    error = None
    active = peak = calls = 0
    lock = threading.Lock()
    delay = 0

    def __init__(self, root, bundle, threads, ffmpeg, uploads):
        self.ffmpeg, self.uploads = ffmpeg, uploads
        self.calibration = json.loads((root / "bundles/nii-small-v1/calibration.json").read_bytes())

    def analyze(self, body, timeout):
        with self.lock:
            type(self).active += 1
            type(self).calls += 1
            type(self).peak = max(self.active, self.peak)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.error:
                raise CannotAnalyze(self.error)
            audio = prepare(body, ffmpeg=self.ffmpeg, temp_dir=self.uploads)
            return decide(audio, [self.score] * len(audio.windows), self.calibration["t_low"], self.calibration["t_warn"])
        finally:
            with self.lock:
                type(self).active -= 1

    def close(self):
        pass


@pytest.fixture
def client(ffmpeg, tmp_path):
    FakeWorker.error, FakeWorker.delay = None, 0
    FakeWorker.calls = FakeWorker.active = FakeWorker.peak = 0
    settings = replace(Settings(), root=ROOT, upload_dir=tmp_path / "uploads", donation_dir=tmp_path / "donations", ffmpeg=ffmpeg,
                       donations_enabled=True, donation_store_audio=True)
    with TestClient(create_app(settings, worker_factory=FakeWorker)) as test:
        yield test


def upload(client, *, body=None, name="private-recording.wav", opt_in=False, consent=True):
    return client.post("/api/analyze", files={"file": (name, body if body is not None else wav_bytes(), "audio/wav")},
                       data={"analysis_consent": str(consent).lower(), "donation_opt_in": str(opt_in).lower()})


def test_health_config_and_static(client):
    assert client.get("/api/health").json()["status"] == "ok"
    config = client.get("/api/config").json()
    assert config["consent_version"] == CONSENT_VERSION
    assert config["donation_token_seconds"] == 900
    assert config["partial_enabled"] is False
    assert config["partial_scope"] is None
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_worker_partial_result_passes_through(client):
    worker = client.app.state.idle_workers[0]
    original = worker.analyze
    partial = {"status": "suspected", "statistic": 3.5, "threshold": 2.0,
               "segments": [{"start": 1.0, "end": 5.0, "peak": 3.5}],
               "localization": "window_4s_indicative", "scope_ko": "unit fixture scope"}
    baseline = upload(client).json()
    def analyze(body, timeout):
        return {**original(body, timeout), "partial": partial}
    with patch.object(worker, "analyze", side_effect=analyze):
        result = upload(client).json()
    assert result["partial"] == partial
    for field in ("outcome_code", "file_score", "windows", "slots", "hashes", "bundle_id"):
        assert result[field] == baseline[field]


def test_partial_config_reports_loaded_scope(client):
    client.app.state.bundle.partial = {"scope_ko": "unit fixture scope"}
    config = client.get("/api/config").json()
    assert config["partial_enabled"] is True
    assert config["partial_scope"] == "unit fixture scope"


@pytest.mark.parametrize("outcome", ["warn", "low", "defer", "cannot"])
def test_all_outcomes(client, outcome):
    calibration = client.app.state.bundle.calibration
    FakeWorker.score = {"warn": calibration["t_warn"], "low": calibration["t_low"], "defer": (calibration["t_warn"] + calibration["t_low"]) / 2, "cannot": 0}[outcome]
    response = upload(client, body=b"broken" if outcome == "cannot" else None)
    data = response.json()
    assert response.status_code == 200
    assert data["outcome_code"] == outcome
    assert data["request_id"] == response.headers["x-request-id"]
    assert set(data["hashes"]) == {"frontend", "audio_windows", "detector", "calibration", "manifest"}
    assert data["notice"]["development_rates"]["pooled/AI"]["n"] == 999
    assert not list(client.app.state.settings.upload_dir.iterdir())


@pytest.mark.parametrize("reason,options", [("analysis_consent_required", {"consent": False}), ("unsupported_format", {"name": "private.exe"}),
                                           ("decode_failed", {"body": b"bad"}), ("duration_out_of_range", {"body": wav_bytes(seconds=3)}),
                                           ("file_too_large", {"body": b"x" * (25 * 1024 * 1024 + 1)})])
def test_error_model_and_no_opt_in_retention(client, reason, options):
    response = upload(client, opt_in=True, **options)
    assert response.json()["reason"] == reason
    assert response.json()["donation_token"] is None
    assert not client.app.state.donations.pending
    assert not list(client.app.state.settings.upload_dir.iterdir())


def test_schema_and_unknown_endpoint(client):
    assert client.post("/api/analyze").status_code == 422
    assert client.post("/api/donations", json={}).json()["outcome_code"] == "cannot"
    assert client.get("/missing").status_code == 404


def test_body_limit_before_parsing(client):
    response = client.post("/api/analyze", content=b"", headers={"content-length": str(30 * 1024 * 1024)})
    assert response.status_code == 413
    assert response.json()["reason"] == "file_too_large"
    assert client.post("/api/analyze", content=b"", headers={"content-length": "9" * 5000}).status_code == 413


def test_inflight_limit_rejects_without_reading_or_retaining_audio(client):
    client.app.state.inflight_uploads = 8
    response = upload(client, opt_in=True)
    assert response.status_code == 503 and response.json()["reason"] == "server_busy"
    assert FakeWorker.calls == 0 and not client.app.state.donations.pending
    client.app.state.inflight_uploads = 0


def test_chunked_multipart_limit_closes_partial_spools(client):
    opened = []
    original = tempfile.SpooledTemporaryFile
    def spool(*args, **kwargs):
        value = original(*args, **kwargs)
        opened.append(value)
        return value
    def chunks():
        yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="private.wav"\r\nContent-Type: audio/wav\r\n\r\n'
        for _ in range(28):
            yield b"x" * (1024 * 1024)
        yield b"\r\n--boundary--\r\n"
    with patch("starlette.formparsers.SpooledTemporaryFile", side_effect=spool):
        response = client.post("/api/analyze", content=chunks(), headers={"content-type": "multipart/form-data; boundary=boundary"})
    assert response.status_code == 400
    assert response.json()["reason"] == "request_body_too_large"
    assert all(value.closed for value in opened)
    assert FakeWorker.calls == 0


def test_default_off_and_donation_withdrawal(client):
    assert upload(client).json()["donation_token"] is None
    store = client.app.state.donations
    assert not store.pending and not list(store.quarantine.iterdir())
    token = upload(client, opt_in=True).json()["donation_token"]
    assert token and token not in store.pending
    assert not list(store.quarantine.iterdir())
    assert client.post("/api/donations", json={"token": token, "confirm": False, "consent_version": CONSENT_VERSION, "eligibility_confirmed": True}).status_code == 422
    code = client.post("/api/donations", json={"token": token, "confirm": True, "consent_version": CONSENT_VERSION, "eligibility_confirmed": True}).json()["withdrawal_code"]
    assert len(list(store.quarantine.iterdir())) == 1
    assert not store.pending
    assert client.post("/api/donations", json={"token": token, "confirm": True, "consent_version": CONSENT_VERSION, "eligibility_confirmed": True}).status_code == 404
    assert client.delete("/api/donations/" + code).json()["status"] == "withdrawn"
    assert not list(store.quarantine.iterdir())
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM donations").fetchone()[0] == 0
    assert client.delete("/api/donations/" + code).status_code == 404


def test_discard_token(client):
    token = upload(client, opt_in=True).json()["donation_token"]
    assert client.delete("/api/donation-tokens/" + token).status_code == 200
    assert not client.app.state.donations.pending


def test_timeout_and_semaphore(client):
    FakeWorker.error = "analysis_timeout"
    assert upload(client).json()["reason"] == "analysis_timeout"
    FakeWorker.error, FakeWorker.delay = None, .1
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda _: upload(client), range(2)))
    assert all(response.status_code == 200 for response in results)
    assert FakeWorker.peak == 1
    assert not list(client.app.state.settings.upload_dir.iterdir())


def test_semaphore_queue_timeout_returns_busy(ffmpeg, tmp_path):
    FakeWorker.delay, FakeWorker.error = .3, None
    FakeWorker.active = FakeWorker.peak = FakeWorker.calls = 0
    settings = replace(Settings(), root=ROOT, upload_dir=tmp_path / "uploads", donation_dir=tmp_path / "donations", ffmpeg=ffmpeg, queue_timeout_seconds=.01)
    with TestClient(create_app(settings, worker_factory=FakeWorker)) as client:
        with concurrent.futures.ThreadPoolExecutor(2) as executor:
            outcomes = [r.json() for r in executor.map(lambda _: upload(client), range(2))]
        assert sum(r["reason"] == "server_busy" for r in outcomes) == 1
        assert FakeWorker.peak == 1
        assert not list(settings.upload_dir.iterdir())
    FakeWorker.delay = 0


def test_log_scan(client, caplog):
    caplog.set_level("INFO", logger="speech.metadata")
    upload(client)
    messages = [record.message for record in caplog.records if record.name == "speech.metadata"]
    assert len(messages) == 1
    assert set(json.loads(messages[0])) == {"request_id", "duration", "outcome", "latency", "bundle_id"}
    assert "private-recording" not in messages[0] and "file_score" not in messages[0]
