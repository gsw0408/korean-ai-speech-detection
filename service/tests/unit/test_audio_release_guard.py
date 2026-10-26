import json
import os
import subprocess
from pathlib import Path

import pytest

from app.settings import Settings


ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "ops"


@pytest.mark.parametrize("value", ["true", "false", "TRUE", "False", " false", "false ", "1", "yes", ""])
def test_audio_storage_flag_requires_exact_boolean(monkeypatch, value):
    monkeypatch.setenv("DONATION_STORE_AUDIO", value)
    if value in {"true", "false"}:
        assert Settings.from_env().donation_store_audio is (value == "true")
    else:
        with pytest.raises(ValueError, match="DONATION_STORE_AUDIO"):
            Settings.from_env()


@pytest.mark.skipif(not OPS.exists(), reason="ops is host-only; verified separately with a read-only ops mount")
def test_public_overlay_passes_no_audio_default():
    assert "DONATION_STORE_AUDIO: ${DONATION_STORE_AUDIO:-false}" in (OPS / "docker-compose.public.yml").read_text()


@pytest.mark.skipif(not OPS.exists() or os.name == "nt", reason="requires POSIX shell and host-only ops")
@pytest.mark.parametrize("setting,env_line,compose_value,override,refused", [
    (True, "DONATION_STORE_AUDIO=false", "false", False, False),
    (False, "DONATION_STORE_AUDIO=false", "false", False, True),
    (True, "", "false", False, True),
    (True, "DONATION_STORE_AUDIO=true", "false", False, True),
    (True, "DONATION_STORE_AUDIO=FALSE", "false", False, True),
    (True, "DONATION_STORE_AUDIO=false\nDONATION_STORE_AUDIO=true", "false", False, True),
    (True, "DONATION_STORE_AUDIO=false", "true", False, True),
    (True, "DONATION_STORE_AUDIO=false", None, False, True),
    (False, "DONATION_STORE_AUDIO=true", "true", True, False),
])
def test_switch_guard_refuses_before_any_compose_up(tmp_path, setting, env_line, compose_value, override, refused):
    root = tmp_path / "nii"
    target = root / "release-fixture"
    target.mkdir(parents=True)
    (root / "CURRENT_RELEASE").write_text(str(target))
    (target / ".env").write_text("IMAGE=fixture\n" + env_line + "\n")
    (target / "docker-compose.public.yml").write_text("services: {}\n")
    config = {"services": {
        "service": {"environment": {"DONATION_STORE_AUDIO": compose_value}, "volumes": [{"source": "/srv/nii-donations", "target": "/data/donations"}]},
        "proxy": {"volumes": [{"source": "/opt/nii/proxy", "target": "/etc/nii-proxy"}, {"source": "/opt/nii/proxy/Caddyfile", "target": "/etc/caddy/Caddyfile"}]},
    }}
    (tmp_path / "config.json").write_text(json.dumps(config))
    script = (OPS / "switch_release.sh").read_text().replace("/opt/nii/release-", str(root) + "/release-").replace("/opt/nii/CURRENT_RELEASE", str(root / "CURRENT_RELEASE")).replace("DON=/srv/nii-donations", "DON=" + str(tmp_path / "empty-donations"))
    (tmp_path / "switch.sh").write_text(script)
    commands = tmp_path / "bin"
    commands.mkdir()
    docker = commands / "docker"
    docker.write_text('''#!/bin/sh
case "$*" in
  *Settings.__dataclass_fields__*) exit "$MISSING_SETTING";;
  *"config --format json"*) cat "$FIXTURE_CONFIG";;
  *"up -d"*) echo up >> "$UP_CALLS"; exit 49;;
  inspect*) echo fixture-start;;
esac
''')
    docker.chmod(0o755)
    calls = tmp_path / "up-calls"
    env = {**os.environ, "PATH": str(commands) + ":" + os.environ["PATH"], "MISSING_SETTING": "0" if setting else "1",
           "FIXTURE_CONFIG": str(tmp_path / "config.json"), "UP_CALLS": str(calls)}
    result = subprocess.run(["sh", str(tmp_path / "switch.sh"), "fixture", *(["--allow-audio-storage"] if override else [])], env=env, capture_output=True, text=True)
    if refused:
        assert result.returncode == 6, result.stdout + result.stderr
        assert "REFUSED: audio storage" in result.stdout
        assert not calls.exists()
    else:
        # The fixture deliberately fails compose up; reaching it proves all preflight checks passed.
        assert result.returncode == 4, result.stdout + result.stderr
        assert calls.exists()
        if override:
            assert "WRITTEN OWNER APPROVAL REQUIRED" in result.stderr
    assert (root / "CURRENT_RELEASE").read_text() == str(target)
