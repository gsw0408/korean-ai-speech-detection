import hashlib
import os
import sqlite3
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from donations.store import CONSENT_TEXT, CONSENT_VERSION, CapacityReached, DonationStore

Usage = namedtuple("Usage", "total used free")
NOW = datetime(2026, 10, 22, 3, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def donate(store, body=b"audio"):
    return store.donate(store.offer(body, "low", "bundle"))


def stored(store):
    return sorted(path.name for path in store.quarantine.iterdir())


def rows(store):
    with store.connect() as connection:
        return connection.execute("SELECT file_id, utc_time FROM donations").fetchall()


def test_consent_v4_records_consent_only_without_research_collection_or_audio_retention():
    assert CONSENT_VERSION == "donation-consent-v4-2026-10-22"
    for phrase in ("시연", "기록 항목", "음성 파일, 파일 해시, 이름·연락처·IP 주소는 저장하지 않습니다", "최대 15분",
                   "30일이 지나면 자동으로 삭제", "프로젝트가 끝나면", "백업은 만들지 않습니다", "철회 코드", "즉시 삭제",
                   "코드를 잃어버리면", "동의하지 않을 수 있으며", "음성 분석은 그대로 이용할 수 있습니다"):
        assert phrase in CONSENT_TEXT
    for removed in ("연구 목적", "연구 검토", "음성 파일 원본", "메타데이터", "영구 삭제", "최대 30일"):
        assert removed not in CONSENT_TEXT


def test_file_count_quota(tmp_path):
    store = DonationStore(tmp_path, max_files=2)
    donate(store), donate(store)
    token = store.offer(b"audio", "low", "bundle")
    with pytest.raises(CapacityReached):
        store.donate(token)
    assert len(stored(store)) == 2 and store.key(token) in store.pending


def test_byte_quota(tmp_path):
    store = DonationStore(tmp_path, max_bytes=10)
    donate(store, b"12345678")
    with pytest.raises(CapacityReached):
        donate(store, b"123")
    donate(store, b"12")
    assert store.usage() == (2, 10)


def test_daily_quota_resets_next_utc_day(tmp_path):
    clock = Clock()
    store = DonationStore(tmp_path, daily_max=2, clock=clock)
    donate(store), donate(store)
    with pytest.raises(CapacityReached):
        donate(store)
    clock.now = NOW + timedelta(days=1)
    donate(store)
    assert len(stored(store)) == 3


def test_daily_quota_survives_restart(tmp_path):
    clock = Clock()
    first = DonationStore(tmp_path, daily_max=2, clock=clock)
    donate(first), donate(first)
    second = DonationStore(tmp_path, daily_max=2, clock=clock)
    with pytest.raises(CapacityReached):
        donate(second)


def test_free_space_floor_refuses_before_writing(tmp_path):
    store = DonationStore(tmp_path, min_free_bytes=1000)
    token = store.offer(b"audio", "low", "bundle")
    with patch("donations.store.shutil.disk_usage", return_value=Usage(10_000, 9_000, 1_004)):
        with pytest.raises(CapacityReached):
            store.donate(token)
    assert stored(store) == [] and rows(store) == []
    with patch("donations.store.shutil.disk_usage", return_value=Usage(10_000, 9_000, 1_005)):
        store.donate(token)
    assert len(stored(store)) == 1


def test_atomic_write_leaves_no_partial_file_on_failure(tmp_path):
    store = DonationStore(tmp_path)
    token = store.offer(b"audio", "low", "bundle")
    with patch("donations.store.os.fsync", side_effect=OSError("disk full fixture")):
        with pytest.raises(OSError):
            store.donate(token)
    assert stored(store) == [] and rows(store) == []
    assert store.key(token) in store.pending


def test_failed_ledger_insert_removes_written_file(tmp_path):
    store = DonationStore(tmp_path)
    token = store.offer(b"audio", "low", "bundle")
    real = store.connect

    def failing():
        connection = real()
        class Wrapper:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return connection.__exit__(*args)
            def execute(self, sql, *args):
                if sql.startswith("INSERT"):
                    raise sqlite3.OperationalError("fixture")
                return connection.execute(sql, *args)
        return Wrapper()
    with patch.object(store, "connect", side_effect=failing):
        with pytest.raises(sqlite3.OperationalError):
            store.donate(token)
    assert stored(store) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_stored_file_is_owner_only(tmp_path):
    store = DonationStore(tmp_path)
    donate(store)
    (name,) = stored(store)
    assert (store.quarantine / name).stat().st_mode & 0o777 == 0o600


def test_reconcile_removes_orphans_temporaries_and_missing_rows(tmp_path):
    store = DonationStore(tmp_path)
    kept = donate(store, b"kept")
    lost = donate(store, b"lost")
    with store.connect() as connection:
        lost_file = connection.execute("SELECT file_id FROM donations WHERE code_hash=?", (store.key(lost),)).fetchone()[0]
    (store.quarantine / lost_file).unlink()
    (store.quarantine / "0123abcd").write_bytes(b"orphan")
    (store.quarantine / ".tmp-deadbeef").write_bytes(b"partial")
    assert store.reconcile() == {"removed_files": 2, "removed_rows": 1}
    assert len(stored(store)) == 1 and len(rows(store)) == 1
    store.withdraw(kept)
    assert stored(store) == [] and rows(store) == []


def test_thirty_day_expiry_removes_file_and_record(tmp_path):
    clock = Clock()
    store = DonationStore(tmp_path, retention_days=30, clock=clock)
    old = donate(store, b"old")
    clock.now = NOW + timedelta(days=10)
    donate(store, b"new")
    clock.now = NOW + timedelta(days=30) - timedelta(seconds=1)
    assert store.expire() == 0
    clock.now = NOW + timedelta(days=30)
    assert store.expire() == 1
    assert len(stored(store)) == 1 and len(rows(store)) == 1
    with pytest.raises(KeyError):
        store.withdraw(old)
    assert hashlib.sha256(b"old").hexdigest().encode() not in store.database.read_bytes()


def test_project_end_purge_deletes_everything(tmp_path):
    store = DonationStore(tmp_path)
    codes = [donate(store, bytes([index]) * 10) for index in range(3)]
    pending = store.offer(b"pending", "low", "bundle")
    (store.quarantine / ".tmp-x").write_bytes(b"partial")
    assert store.purge_all() == 3
    assert stored(store) == [] and rows(store) == [] and not store.pending
    for code in codes:
        with pytest.raises(KeyError):
            store.withdraw(code)
    with pytest.raises(KeyError):
        store.donate(pending)
    for index in range(3):
        assert hashlib.sha256(bytes([index]) * 10).hexdigest().encode() not in store.database.read_bytes()


def test_hourly_quota_is_a_sliding_window_that_survives_restart(tmp_path):
    clock = Clock()
    first = DonationStore(tmp_path, hourly_max=2, clock=clock)
    donate(first)
    clock.now = NOW + timedelta(minutes=30)
    donate(first)
    second = DonationStore(tmp_path, hourly_max=2, clock=clock)
    token = second.offer(b"audio", "low", "bundle")
    with pytest.raises(CapacityReached):
        second.donate(token)
    assert len(stored(second)) == 2 and second.key(token) in second.pending
    clock.now = NOW + timedelta(hours=1, seconds=1)
    second.donate(token)
    assert len(stored(second)) == 3
