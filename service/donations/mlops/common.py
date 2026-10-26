import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

PROTECTED = re.compile(r"(^|[^a-z])(calib|calibration|final|sealed)([^a-z]|$)", re.I)


class PolicyError(RuntimeError):
    """A pipeline step was refused by a data-protection or governance rule."""


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return sha256_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


def write_json(path, value):
    """Atomic JSON write with owner-only permissions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".tmp-" + path.name)
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return path


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def refuse_protected(*names):
    """Protected evaluation data (CALIB/FINAL/sealed) never enters this pipeline."""
    for name in names:
        if PROTECTED.search(str(name).replace("\\", "/")):
            raise PolicyError(f"protected evaluation data is not allowed here: {name}")
