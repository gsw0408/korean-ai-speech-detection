import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

# Runs inside the service container (non-root user, tmpfs uploads); skipped on a developer machine.
pytestmark = pytest.mark.skipif(not hasattr(os, "getuid") or not Path("/proc/mounts").exists(), reason="container-only test")


def test_non_root_tmpfs_and_frozen_assets():
    assert os.getuid() == 10001
    assert any(line.split()[1:3] == ["/tmp/uploads", "tmpfs"] for line in Path("/proc/mounts").read_text().splitlines())
    root = Path(__file__).resolve().parents[2]
    audit = json.loads((root / "REUSE_AUDIT.json").read_bytes())
    for name in ("frontend", "audio_windows"):
        assert hashlib.sha256((root / "pipeline/frozen" / (name + ".py")).read_bytes()).hexdigest() == audit["inputs"][name]["sha256"]
    assert hashlib.sha256((root / "models/wav2vec2-small-antideepfake.onnx").read_bytes()).hexdigest() == audit["inputs"]["model"]["sha256"]
    subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    assert not (root / "calibration/DEVELOPMENT_REPORT.json").exists()
    assert all(set(value) == {"sha256"} for value in audit["inputs"].values())
