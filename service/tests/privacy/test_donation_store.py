import hashlib
import sqlite3
from unittest.mock import patch
import pytest
from donations.store import CONSENT_VERSION, DonationStore


def test_expiry_capacity_and_separate_copy(tmp_path):
    store = DonationStore(tmp_path, ttl=1, max_pending_bytes=8)
    body = bytearray(b"audio")
    with patch("donations.store.time.monotonic", return_value=100):
        token = store.offer(body, "low", "fixture")
        body[:] = b"xxxxx"
        assert store.pending[store.key(token)]["body"] == b"audio"
        assert store.offer(b"aaaa", "low", "fixture") is None
    with patch("donations.store.time.monotonic", return_value=102):
        store.purge()
        assert not store.pending
        with pytest.raises(KeyError):
            store.donate(token)
    assert not list(store.quarantine.iterdir())


def test_consent_record_and_complete_withdrawal(tmp_path):
    store = DonationStore(tmp_path)
    code = store.donate(store.offer(b"audio", "defer", "test-bundle"))
    with store.connect() as connection:
        row = connection.execute("SELECT * FROM donations").fetchone()
    assert row[2] == CONSENT_VERSION and row[3].endswith("+00:00")
    assert row[4:] == (hashlib.sha256(b"audio").hexdigest(), "defer", "test-bundle")
    assert code not in row
    store.withdraw(code)
    assert not list(store.quarantine.iterdir())
    with store.connect() as connection:
        assert connection.execute("SELECT * FROM donations").fetchall() == []
    assert hashlib.sha256(b"audio").hexdigest().encode() not in store.database.read_bytes()


def test_failed_ledger_write_removes_partial_file(tmp_path):
    store = DonationStore(tmp_path)
    token = store.offer(b"audio", "low", "bundle")
    with patch.object(store, "connect", side_effect=sqlite3.OperationalError("fixture failure")):
        with pytest.raises(sqlite3.OperationalError):
            store.donate(token)
    assert not list(store.quarantine.iterdir())
    assert store.key(token) in store.pending
