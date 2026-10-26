"""Package source, licensed model and reports; exclude all audio/work data."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("app", "pipeline", "donations", "static", "bundles", "calibration", "tests", "scripts", "evidence")
FILES = ("Dockerfile", ".dockerignore", ".gitignore", "docker-compose.yml", "docker-compose.prod.yml", "Caddyfile", "pytest.ini",
         "requirements.txt", "requirements-dev.txt", "README.md", "RUNBOOK_AWS.md", "MODEL_ATTRIBUTION.md", "REUSE_AUDIT.json",
         "PARITY_REPORT.json", "BENCHMARK.json", "VERIFICATION_REPORT.md", "UI_CHECKLIST.md", "RUNTIME_AUDIT.json", "models/wav2vec2-small-antideepfake.onnx")


def main():
    targets = [ROOT / value for value in FILES]
    for folder in FOLDERS:
        targets.extend(path for path in (ROOT / folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    target = ROOT / "NII_SMALL_WEBSERVICE.zip"
    temporary = target.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for source in sorted(set(targets)):
            archive.write(source, "webservice/" + source.relative_to(ROOT).as_posix())
    temporary.replace(target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    report = {"archive": target.name, "sha256": digest.hexdigest(), "members": len(set(targets)), "research_audio_included": False, "work_directory_included": False}
    (ROOT / "PACKAGE_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
