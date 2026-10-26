"""Short-lived memory copies, a bounded quarantine and an SQLite consent/withdrawal ledger."""
import hashlib
import os
import secrets
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

CONSENT_VERSION = "donation-consent-v4-2026-10-22"
# Draft pending legal review (see LEGAL_REVIEW.md); a wording change needs a new version string.
# The text assumes DONATION_STORE_AUDIO=false: confirming records consent only and no audio is kept.
CONSENT_TEXT = ("음성 데이터 제공 및 동의 철회 절차(시연)에 참여하는 데 동의합니다.\n"
                "· 기록 항목: 동의 시각, 분석 결과 구분, 모델 버전, 동의 문구 버전, 철회 코드의 해시값. "
                "음성 파일, 파일 해시, 이름·연락처·IP 주소는 저장하지 않습니다.\n"
                "· 음성 사본: 분석이 끝난 뒤 메모리에만 최대 15분 있으며, 확정하거나 취소하면 바로 삭제됩니다.\n"
                "· 동의 기록: 30일이 지나면 자동으로 삭제하며, 프로젝트가 끝나면 남은 기간과 관계없이 모두 삭제합니다. 백업은 만들지 않습니다.\n"
                "· 철회: 철회 코드로 언제든지 철회할 수 있으며, 철회하면 동의 기록이 즉시 삭제됩니다. "
                "이름이나 연락처를 받지 않으므로 코드를 잃어버리면 개별 철회가 어렵고, 이 경우 동의 기록은 30일이 지나면 자동 삭제됩니다.\n"
                "· 동의하지 않을 수 있으며, 동의하지 않아도 음성 분석은 그대로 이용할 수 있습니다.")
GiB = 1024 ** 3


class CapacityReached(Exception):
    """A storage quota or the free-space floor refuses a new donation."""


class DonationStore:
    def __init__(self, directory, *, ttl=900, max_pending_bytes=128 * 1024 * 1024, max_bytes=4 * GiB,
                 max_files=1000, daily_max=200, hourly_max=40, min_free_bytes=512 * 1024 * 1024, retention_days=30, store_audio=True, clock=None):
        self.directory = Path(directory)
        self.quarantine = self.directory / "quarantine"
        self.quarantine.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "consent.sqlite3"
        self.lock, self.pending = threading.RLock(), {}
        self.ttl, self.max_pending_bytes = ttl, max_pending_bytes
        self.max_bytes, self.max_files, self.daily_max, self.hourly_max = max_bytes, max_files, daily_max, hourly_max
        self.min_free_bytes, self.retention = min_free_bytes, timedelta(days=retention_days)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        # False: demonstration mode; confirmation records consent only and the audio copy is dropped.
        self.store_audio = store_audio
        with self.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS donations (code_hash TEXT PRIMARY KEY, file_id TEXT UNIQUE NOT NULL, consent_version TEXT NOT NULL, utc_time TEXT NOT NULL, file_sha256 TEXT NOT NULL, outcome TEXT NOT NULL, bundle_id TEXT NOT NULL)")
        self.day, self.day_count = None, 0

    def connect(self):
        return sqlite3.connect(self.database)

    @staticmethod
    def key(token):
        return hashlib.sha256(token.encode()).hexdigest()

    def purge(self):
        with self.lock:
            for key in [k for k, v in self.pending.items() if v["expires"] <= time.monotonic()]:
                del self.pending[key]

    def usage(self):
        """Stored donation files and bytes; temporary files are not donations."""
        files = [path for path in self.quarantine.iterdir() if path.is_file() and not path.name.startswith(".tmp-")]
        return len(files), sum(path.stat().st_size for path in files)

    def _today_count(self):
        today = self.clock().date()
        if self.day != today:
            with self.connect() as connection:
                rows = connection.execute("SELECT utc_time FROM donations").fetchall()
            self.day = today
            self.day_count = sum(1 for (stamp,) in rows if datetime.fromisoformat(stamp).date() == today)
        return self.day_count

    def _last_hour_count(self):
        """Donations stored during the last hour, read from the ledger so a restart cannot reset it."""
        since = self.clock() - timedelta(hours=1)
        with self.connect() as connection:
            rows = connection.execute("SELECT utc_time FROM donations").fetchall()
        return sum(1 for (stamp,) in rows if datetime.fromisoformat(stamp) > since)

    def check_capacity(self, size):
        """Raise CapacityReached unless one more donation of `size` bytes fits every limit."""
        with self.lock:
            count, total = self.usage()
            free = shutil.disk_usage(self.quarantine).free
            if (count + 1 > self.max_files or total + size > self.max_bytes
                    or self._today_count() + 1 > self.daily_max or self._last_hour_count() + 1 > self.hourly_max
                    or free - size < self.min_free_bytes):
                raise CapacityReached("donation_capacity_reached")

    def offer(self, body, outcome, bundle_id):
        with self.lock:
            self.purge()
            if sum(len(v["body"]) for v in self.pending.values()) + len(body) > self.max_pending_bytes:
                return None
            token = secrets.token_urlsafe(32)
            self.pending[self.key(token)] = {"body": bytes(bytearray(body)), "expires": time.monotonic() + self.ttl,
                                             "outcome": outcome, "bundle_id": bundle_id}
            return token

    def discard(self, token):
        with self.lock:
            self.pending.pop(self.key(token), None)

    def _write_atomic(self, file_id, body):
        temporary, target = self.quarantine / (".tmp-" + file_id), self.quarantine / file_id
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self._sync_directory()
        return target

    def _sync_directory(self):
        try:
            descriptor = os.open(self.quarantine, os.O_RDONLY)
        except OSError:
            return  # Directory fsync is unavailable on Windows; the Linux service performs it.
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def donate(self, token):
        with self.lock:
            self.purge()
            value = self.pending.get(self.key(token))
            if value is None:
                raise KeyError("expired_or_invalid_token")
            self.check_capacity(len(value["body"]))
            code, file_id = secrets.token_urlsafe(32), secrets.token_hex(16)
            target = None
            try:
                if self.store_audio:
                    target = self._write_atomic(file_id, value["body"])
                digest = hashlib.sha256(value["body"]).hexdigest() if self.store_audio else ""
                with self.connect() as connection:
                    connection.execute("INSERT INTO donations VALUES (?,?,?,?,?,?,?)", (self.key(code), file_id, CONSENT_VERSION,
                                       self.clock().isoformat(), digest, value["outcome"], value["bundle_id"]))
            except Exception:
                if target is not None:
                    target.unlink(missing_ok=True)
                raise
            del self.pending[self.key(token)]
            self._today_count()
            self.day_count += 1
            return code

    def _delete(self, connection, rows):
        for code_hash, file_id in rows:
            (self.quarantine / file_id).unlink(missing_ok=True)
            connection.execute("DELETE FROM donations WHERE code_hash=?", (code_hash,))
        connection.commit()
        connection.execute("VACUUM")

    def withdraw(self, code):
        with self.lock, self.connect() as connection:
            row = connection.execute("SELECT code_hash, file_id FROM donations WHERE code_hash=?", (self.key(code),)).fetchone()
            if row is None:
                raise KeyError("withdrawal_not_found")
            self._delete(connection, [row])

    def expire(self):
        """Delete donations older than the retention period; returns the number removed."""
        cutoff = self.clock() - self.retention
        with self.lock, self.connect() as connection:
            rows = connection.execute("SELECT code_hash, file_id, utc_time FROM donations").fetchall()
            expired = [(code_hash, file_id) for code_hash, file_id, stamp in rows if datetime.fromisoformat(stamp) <= cutoff]
            if expired:
                self._delete(connection, expired)
            return len(expired)

    def reconcile(self):
        """Remove temporary files, files without a ledger row and rows without a file."""
        with self.lock, self.connect() as connection:
            rows = connection.execute("SELECT code_hash, file_id FROM donations").fetchall()
            known = {file_id for _, file_id in rows}
            removed_files = 0
            for path in self.quarantine.iterdir():
                if path.is_file() and (path.name.startswith(".tmp-") or path.name not in known):
                    path.unlink(missing_ok=True)
                    removed_files += 1
            # Without audio storage a consent record has no file by design.
            missing = [(code_hash, file_id) for code_hash, file_id in rows
                       if self.store_audio and not (self.quarantine / file_id).is_file()]
            if missing:
                self._delete(connection, missing)
            return {"removed_files": removed_files, "removed_rows": len(missing)}

    def purge_all(self):
        """Project-end deletion: every stored file, every consent record and every pending copy."""
        with self.lock, self.connect() as connection:
            self.pending.clear()
            rows = connection.execute("SELECT code_hash, file_id FROM donations").fetchall()
            self._delete(connection, rows)
            for path in self.quarantine.iterdir():
                if path.is_file():
                    path.unlink(missing_ok=True)
            return len(rows)

    def close(self):
        with self.lock:
            self.pending.clear()
