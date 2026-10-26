import os
import shutil
import subprocess
from pathlib import Path


def test_deploy_rollback_uses_only_stubbed_local_commands(tmp_path):
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(root / "scripts/deploy.sh", tmp_path / "scripts/deploy.sh")
    (tmp_path / ".deployed-image").write_text("private.example/repo:previous\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    aws = bin_dir / "aws"
    aws.write_text("#!/bin/sh\nprintf 'test-only-placeholder\\n'\n")
    docker = bin_dir / "docker"
    docker.write_text('''#!/bin/sh
printf '%s %s\n' "${IMAGE:-none}" "$*" >> "$CALLS"
case "$1" in
login) cat >/dev/null; exit 0;;
pull) exit 0;;
compose) case "$*" in *"up -d"*) case "${IMAGE:-}" in *:candidate) exit 1;; esac;; esac; exit 0;;
esac
exit 0
''')
    aws.chmod(0o755)
    docker.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir) + ":" + os.environ["PATH"], "IMAGE_REPOSITORY": "private.example/repo", "CALLS": str(tmp_path / "calls")}
    result = subprocess.run(["sh", str(tmp_path / "scripts/deploy.sh"), "candidate"], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    calls = (tmp_path / "calls").read_text()
    assert "private.example/repo:candidate compose" in calls
    assert "private.example/repo:previous compose" in calls
    assert (tmp_path / ".deployed-image").read_text().strip() == "private.example/repo:previous"
