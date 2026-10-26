import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings
from donations.store import CONSENT_VERSION
from test_routes import ROOT, FakeWorker, upload


def make_client(ffmpeg, tmp_path, **overrides):
    FakeWorker.error, FakeWorker.delay = None, 0
    overrides.setdefault("donation_store_audio", True)
    settings = replace(Settings(), root=ROOT, upload_dir=tmp_path / "uploads", donation_dir=tmp_path / "donations",
                       ffmpeg=ffmpeg, **overrides)
    return TestClient(create_app(settings, worker_factory=FakeWorker))


def confirm(client, token):
    return client.post("/api/donations", json={"token": token, "confirm": True, "consent_version": CONSENT_VERSION,
                                               "eligibility_confirmed": True})


def test_default_settings_keep_donations_off():
    assert Settings().donations_enabled is False


def test_disabled_ignores_opt_in_and_refuses_storage_but_withdrawal_still_works(ffmpeg, tmp_path):
    with make_client(ffmpeg, tmp_path, donations_enabled=True) as enabled:
        code = confirm(enabled, upload(enabled, opt_in=True).json()["donation_token"]).json()["withdrawal_code"]
    with make_client(ffmpeg, tmp_path, donations_enabled=False) as client:
        config = client.get("/api/config").json()
        assert config["donations_enabled"] is False and config["donation_retention_days"] == 30
        result = upload(client, opt_in=True).json()
        assert result["outcome_code"] != "cannot" and result["donation_token"] is None
        store = client.app.state.donations
        assert not store.pending
        forged = store.offer(b"audio", "low", "bundle")
        response = confirm(client, forged)
        assert response.status_code == 403 and response.json()["reason"] == "donations_disabled"
        assert len(list(store.quarantine.iterdir())) == 1
        assert client.delete("/api/donations/" + code).json()["status"] == "withdrawn"
        assert not list(store.quarantine.iterdir())


def test_capacity_reached_keeps_analysis_and_reports_reason(ffmpeg, tmp_path):
    with make_client(ffmpeg, tmp_path, donations_enabled=True, donation_max_files=1) as client:
        assert confirm(client, upload(client, opt_in=True).json()["donation_token"]).status_code == 200
        result = upload(client, opt_in=True).json()
        assert result["outcome_code"] != "cannot"
        assert result["donation_token"] is None and result["donation_reason"] == "donation_capacity_reached"
        store = client.app.state.donations
        forged = store.offer(b"audio", "low", "bundle")
        response = confirm(client, forged)
        assert response.status_code == 507 and response.json()["reason"] == "donation_capacity_reached"
        assert len(list(store.quarantine.iterdir())) == 1


def test_restart_persists_and_reconciles(ffmpeg, tmp_path):
    with make_client(ffmpeg, tmp_path, donations_enabled=True) as client:
        code = confirm(client, upload(client, opt_in=True).json()["donation_token"]).json()["withdrawal_code"]
        quarantine = client.app.state.donations.quarantine
    (quarantine / ".tmp-crashed").write_bytes(b"partial")
    with make_client(ffmpeg, tmp_path, donations_enabled=True) as client:
        assert [path.name for path in quarantine.iterdir()] != [] and not (quarantine / ".tmp-crashed").exists()
        assert client.delete("/api/donations/" + code).json()["status"] == "withdrawn"
        assert not list(quarantine.iterdir())


@pytest.mark.parametrize("code", ["' OR 1=1--", "../../etc/passwd", "%2e%2e%2f", "x" * 64, "%00"])
def test_hostile_withdrawal_codes_are_uniform_404(ffmpeg, tmp_path, code):
    with make_client(ffmpeg, tmp_path, donations_enabled=True) as client:
        kept = confirm(client, upload(client, opt_in=True).json()["donation_token"]).json()["withdrawal_code"]
        response = client.delete("/api/donations/" + code)
        assert response.status_code in (404, 405) and kept not in response.text
        store = client.app.state.donations
        with store.connect() as connection:
            assert connection.execute("SELECT COUNT(*) FROM donations").fetchone()[0] == 1


def test_storage_log_has_counts_only(ffmpeg, tmp_path, caplog):
    caplog.set_level("INFO", logger="speech.storage")
    with make_client(ffmpeg, tmp_path, donations_enabled=True) as client:
        token = upload(client, opt_in=True).json()["donation_token"]
        code = confirm(client, token).json()["withdrawal_code"]
        client.delete("/api/donations/" + code)
    messages = [record.message for record in caplog.records if record.name == "speech.storage"]
    events = [json.loads(message)["event"] for message in messages]
    assert events[:3] == ["startup", "donate", "withdraw"]
    for message in messages:
        assert set(json.loads(message)) <= {"event", "files", "bytes", "removed_files", "removed_rows", "expired"}
        assert token not in message and code not in message and "private-recording" not in message


@pytest.mark.parametrize("field,value", [("donation_retention_days", 31), ("donation_retention_days", 0), ("donation_hourly_max", 0),
                                         ("donation_max_files", 0), ("donation_min_free_bytes", -1)])
def test_invalid_storage_settings_are_rejected(field, value):
    with pytest.raises(ValueError):
        create_app(replace(Settings(), root=ROOT, **{field: value}), worker_factory=FakeWorker)


def test_env_flag_parsing(monkeypatch):
    monkeypatch.setenv("DONATIONS_ENABLED", "true")
    assert Settings.from_env().donations_enabled is True
    monkeypatch.setenv("DONATIONS_ENABLED", "yes")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.delenv("DONATIONS_ENABLED")
    assert Settings.from_env().donations_enabled is False


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_schema_and_docs_are_not_exposed(ffmpeg, tmp_path, path):
    with make_client(ffmpeg, tmp_path) as client:
        assert client.get(path).status_code == 404


def test_eligibility_and_current_consent_version_are_required_before_storing(ffmpeg, tmp_path):
    with make_client(ffmpeg, tmp_path, donations_enabled=True) as client:
        token = upload(client, opt_in=True).json()["donation_token"]
        store = client.app.state.donations
        body = {"token": token, "confirm": True, "consent_version": CONSENT_VERSION}
        response = client.post("/api/donations", json=body)
        assert response.status_code == 422 and response.json()["reason"] == "donation_eligibility_required"
        response = client.post("/api/donations", json={**body, "eligibility_confirmed": False})
        assert response.status_code == 422 and response.json()["reason"] == "donation_eligibility_required"
        response = client.post("/api/donations", json={**body, "consent_version": "donation-consent-v2-2026-10-22",
                                                       "eligibility_confirmed": True})
        assert response.status_code == 409 and response.json()["reason"] == "consent_version_changed"
        assert not list(store.quarantine.iterdir()) and store.key(token) in store.pending
        assert confirm(client, token).status_code == 200
        with store.connect() as connection:
            assert connection.execute("SELECT consent_version FROM donations").fetchall() == [(CONSENT_VERSION,)]


def test_hourly_cap_refuses_with_capacity_reason(ffmpeg, tmp_path):
    with make_client(ffmpeg, tmp_path, donations_enabled=True, donation_hourly_max=1) as client:
        assert confirm(client, upload(client, opt_in=True).json()["donation_token"]).status_code == 200
        result = upload(client, opt_in=True).json()
        assert result["donation_token"] is None and result["donation_reason"] == "donation_capacity_reached"


def test_hourly_setting_from_env(monkeypatch):
    assert Settings.from_env().donation_hourly_max == 40
    monkeypatch.setenv("DONATION_HOURLY_MAX", "7")
    assert Settings.from_env().donation_hourly_max == 7


def test_privacy_policy_draft_is_served_with_placeholders_and_review_markers(ffmpeg, tmp_path):
    with make_client(ffmpeg, tmp_path) as client:
        response = client.get("/privacy")
        assert response.status_code == 200 and "default-src 'self'" in response.headers["content-security-policy"]
        page = response.text
        for phrase in ("초안 — 법률 검토 전", "【운영자 이름 또는 소속】", "【이메일 주소】", "[검토 필요]", CONSENT_VERSION,
                       "만 14세 미만은 음성 데이터 제공에 참여할 수 없습니다", "물리적 완전 소거까지 보장하지는 않습니다",
                       "최대 1시간", "백업이나 스냅숏은 만들지 않습니다", "다시 시작할 때 바로 삭제",
                       "음성 파일은 디스크에 저장하지 않습니다"):
            assert phrase in page, phrase
        for removed in ("<script", "영구 삭제", "음성 기부 (선택", "음성 파일 원본", "연구 검토", "기부 음성", "연구용"):
            assert removed not in page, removed


def test_default_demonstration_mode_records_consent_without_keeping_audio(ffmpeg, tmp_path, monkeypatch):
    monkeypatch.delenv("DONATION_STORE_AUDIO", raising=False)
    assert Settings().donation_store_audio is False and Settings.from_env().donation_store_audio is False
    with make_client(ffmpeg, tmp_path, donations_enabled=True, donation_store_audio=False) as client:
        assert client.get("/api/config").json()["donation_store_audio"] is False
        token = upload(client, opt_in=True).json()["donation_token"]
        store = client.app.state.donations
        response = confirm(client, token)
        assert response.status_code == 200 and response.json()["status"] == "consent_recorded"
        code = response.json()["withdrawal_code"]
        assert not list(store.quarantine.iterdir()) and not store.pending
        with store.connect() as connection:
            assert connection.execute("SELECT consent_version, file_sha256 FROM donations").fetchall() == [(CONSENT_VERSION, "")]
    with make_client(ffmpeg, tmp_path, donations_enabled=True, donation_store_audio=False) as client:
        store = client.app.state.donations
        with store.connect() as connection:
            assert connection.execute("SELECT COUNT(*) FROM donations").fetchone()[0] == 1
        assert client.delete("/api/donations/" + code).json()["status"] == "withdrawn"
        with store.connect() as connection:
            assert connection.execute("SELECT COUNT(*) FROM donations").fetchone()[0] == 0
        assert client.delete("/api/donations/" + code).status_code == 404


def test_beta_headers_and_optional_feedback_link(ffmpeg, tmp_path, monkeypatch):
    with make_client(ffmpeg, tmp_path) as client:
        response = client.get("/")
        assert "microphone=()" in response.headers["permissions-policy"]
        assert response.headers["cross-origin-opener-policy"] == "same-origin"
        assert client.get("/api/config").json()["feedback_url"] is None
    with make_client(ffmpeg, tmp_path, feedback_url="https://forms.example/beta") as client:
        assert client.get("/api/config").json()["feedback_url"] == "https://forms.example/beta"
    for bad in ("http://forms.example", "javascript:alert(1)", "https://x.example/\"><script>"):
        with pytest.raises(ValueError):
            create_app(replace(Settings(), root=ROOT, feedback_url=bad), worker_factory=FakeWorker)
    monkeypatch.setenv("FEEDBACK_URL", " https://forms.example/b ")
    assert Settings.from_env().feedback_url == "https://forms.example/b"


def test_head_requests_and_privacy_disclosures(ffmpeg, tmp_path):
    with make_client(ffmpeg, tmp_path) as client:
        for path in ("/", "/privacy"):
            response = client.head(path)
            assert response.status_code == 200 and response.content == b""
        page = client.get("/privacy").text
        for phrase in ("쿠키, 광고·방문 분석 도구, 외부 글꼴이나 외부 스크립트를 사용하지 않습니다",
                       "누르기 전에는 서버로 전송되지 않습니다", "서비스 컨테이너를 교체하거나 프로젝트가 끝나면 삭제"):
            assert phrase in page, phrase


@pytest.mark.parametrize("bundle,partial", [("bundles/nii-small-v1", False), ("bundles/nii-small-v1-partial2", True)])
def test_performance_disclosure_includes_unfavourable_results(ffmpeg, tmp_path, bundle, partial):
    with make_client(ffmpeg, tmp_path, active_bundle=bundle) as client:
        perf = client.get("/api/config").json()["performance"]
    text = json.dumps(perf, ensure_ascii=False)
    for phrase in ("4.70%", "0.982", "8.8%", "서비스 모델 선택(10/10)이 이 결과를 본 뒤", "확률이 아닙니다", "보정된 확률을 제공하지 않습니다"):
        assert phrase in text, phrase
    if partial:
        for phrase in ("9.85%", "5.5% / 26.6% / 45.3% / 77.0%", "입증 실패", "26.0%", "FINAL)으로는 평가하지 않았습니다"):
            assert phrase in text, phrase
    else:
        assert perf["partial"] is None
