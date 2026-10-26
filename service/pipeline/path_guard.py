"""Fail closed before accessing research source files or archive members."""

import hashlib
import ntpath
import re
from pathlib import Path, PurePosixPath


class ProtectedPathError(ValueError):
    pass


_FORBIDDEN = re.compile(r"calibration|calib|final|vl_data|verification", re.I)
_PRIVATE_ROOTS = (
    "c:/users/owner/.codex/tmp/e2fr",
    "c:/users/owner/.codex/tmp/fr",
    "/mnt/c/users/owner/.codex/tmp/e2fr",
    "/mnt/c/users/owner/.codex/tmp/fr",
)
_SOURCE_CONTAINERS = (
    "C:/Users/owner/.codex/tmp/dataset-v2-pre-final/transfer/FAKEGEN_v2_TRAIN_DEV_20261005.zip.bin",
    "D:/AI/학교자료/26년_2학기/오픈소스SW_프로젝트2/claude/artifacts/kaggle_execution_p5_final_20261008/input_uploads/evidence/POST_FREEZE_TRAINING_EVIDENCE.zip.bin",
)


def guard_source_container(path):
    """Owner-approved exceptions apply only to these exact ZIP containers.

    Member access still requires guard_member and an explicit member allowlist.
    This exception does not authorize extracting a container wholesale.
    """
    value = str(path).replace("\\", "/").casefold()
    approved = {source.casefold() for source in _SOURCE_CONTAINERS}
    if value not in approved:
        raise ProtectedPathError("Unapproved source container")
    resolved = Path(path).resolve()
    if str(resolved).replace("\\", "/").casefold() not in approved:
        raise ProtectedPathError("Source container resolves outside allowlist")
    return resolved


def _check_text(value):
    text = str(value).replace("\\", "/").casefold()
    if "\x00" in text or _FORBIDDEN.search(text):
        raise ProtectedPathError("Protected research path")
    if text.startswith(("//?/", "//./")):
        text = text[4:]
        if not re.match(r"^[a-z]:/", text):
            raise ProtectedPathError("Unapproved filesystem namespace")
    elif text.startswith("//"):
        raise ProtectedPathError("Network research paths are not authorized")
    normalized = ntpath.normpath(text).replace("\\", "/").rstrip("/")
    if any(value == root or value.startswith(root + "/")
           for value in (text.rstrip("/"), normalized) for root in _PRIVATE_ROOTS):
        raise ProtectedPathError("Protected private research root")


def guard_path(path):
    """Check both supplied spelling and resolved symlink/junction destination."""
    _check_text(path)
    resolved = Path(path).resolve()
    _check_text(resolved)
    return resolved


def guard_member(member):
    """Validate a ZIP member without opening or extracting its contents."""
    _check_text(member)
    text = str(member).replace("\\", "/")
    parts = PurePosixPath(text).parts
    if not parts or text.startswith("/") or ntpath.splitdrive(text)[0] or ":" in text:
        raise ProtectedPathError("Absolute archive member")
    if any(part in ("..", ".") for part in text.split("/")):
        raise ProtectedPathError("Archive path traversal")
    return PurePosixPath(text)


def read_bytes(path):
    return guard_path(path).read_bytes()


def sha256_file(path):
    digest = hashlib.sha256()
    with guard_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
