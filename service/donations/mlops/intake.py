"""Intake of confirmed contributions from the donation store, and propagation of withdrawals and expiries."""
import shutil
import sqlite3
from pathlib import Path

from . import TRAINING_CONSENT_VERSIONS
from .common import now, read_json, sha256_file, write_json


def _ledger(store_dir):
    database = Path(store_dir) / "consent.sqlite3"
    if not database.exists():
        return {}
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT file_id, consent_version, utc_time, file_sha256 FROM donations").fetchall()
    return {file_id: {"consent_version": version, "donated_at": stamp, "sha256": digest} for file_id, version, stamp, digest in rows}


def intake(store_dir, intake_dir, *, store_audio):
    """Copy confirmed, still-consented contributions into the review area. Returns a summary.

    With store_audio false (the current service policy) there is no audio to take in, so nothing is read.
    The model's outcome stored in the ledger is deliberately not copied: it must never become a label.
    """
    if not store_audio:
        return {"status": "disabled_by_policy", "reason": "DONATION_STORE_AUDIO=false: the service keeps no audio", "added": 0}
    intake_dir = Path(intake_dir)
    added = skipped = 0
    for file_id, row in _ledger(store_dir).items():
        source = Path(store_dir) / "quarantine" / file_id
        target = intake_dir / "items" / file_id
        if (target / "meta.json").exists():
            continue
        if not source.is_file() or not row["sha256"] or sha256_file(source) != row["sha256"]:
            skipped += 1
            continue
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target / "audio")
        write_json(target / "meta.json", {"contribution_id": file_id, "consent_version": row["consent_version"],
                                          "donated_at": row["donated_at"], "sha256": row["sha256"], "taken_in_at": now(),
                                          "training_consent": row["consent_version"] in TRAINING_CONSENT_VERSIONS})
        added += 1
    return {"status": "ok", "added": added, "skipped_integrity": skipped}


def sync_withdrawals(store_dir, intake_dir, datasets_dir=None):
    """Delete intake copies whose ledger row is gone (withdrawn or expired) and supersede dataset versions holding them."""
    ledger = _ledger(store_dir)
    items = Path(intake_dir) / "items"
    removed = []
    if items.is_dir():
        for item in sorted(items.iterdir()):
            if item.name not in ledger:
                shutil.rmtree(item)
                removed.append(item.name)
    superseded = []
    if datasets_dir and removed and Path(datasets_dir).is_dir():
        for manifest_path in sorted(Path(datasets_dir).glob("*/manifest.json")):
            manifest = read_json(manifest_path)
            hit = {entry.get("contribution_id") for entry in manifest["items"]} & set(removed)
            if hit:
                # Scrub identifiers and audio hashes of withdrawn items; keep only split/label counts for the audit trail.
                manifest["items"] = [{"withdrawn": True, "split": entry["split"], "label": entry["label"]}
                                     if entry.get("contribution_id") in hit else entry for entry in manifest["items"]]
                if manifest.get("status") != "superseded":
                    manifest.update(status="superseded", superseded_at=now(), superseded_reason="withdrawal_or_expiry")
                    superseded.append(manifest["version_id"])
                manifest["withdrawn_count"] = sum(bool(entry.get("withdrawn")) for entry in manifest["items"])
                write_json(manifest_path, manifest)
    if removed:
        log = Path(intake_dir) / "withdrawal_log.json"
        history = read_json(log) if log.exists() else []
        # Only counts and time are kept; contribution ids of withdrawn items are not retained.
        history.append({"at": now(), "removed": len(removed), "superseded_versions": superseded})
        write_json(log, history)
    return {"removed": len(removed), "superseded_versions": superseded}
