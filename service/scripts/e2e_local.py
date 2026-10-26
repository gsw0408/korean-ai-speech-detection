"""Live HTTP tests with real NII inference and permitted DEV fixtures."""
import hashlib
import json
import os
import subprocess
import sys
import time
import io
from pathlib import Path
import httpx
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]


def main():
    calibration = json.loads((ROOT / "bundles/nii-small-v1/calibration.json").read_bytes())
    report = json.loads((ROOT / "work/DEVELOPMENT_REPORT.json").read_bytes())
    reference = json.loads((ROOT / "work/nii.json").read_bytes())["records"]
    rows = {row["sample_id"]: row for row in json.loads((ROOT / "work/dev_manifest.json").read_bytes())}
    eligible = [r for r in reference if r["sample_id"] not in report["service_rejections"]]
    midpoint = (calibration["t_low"] + calibration["t_warn"]) / 2
    selected = {"low": min(eligible, key=lambda r: r["mean_negative_bonafide_logit"]),
                "warn": max(eligible, key=lambda r: r["mean_negative_bonafide_logit"]),
                "defer": min(eligible, key=lambda r: abs(r["mean_negative_bonafide_logit"] - midpoint))}
    folder = ROOT / "work/e2e-fixtures"
    folder.mkdir(exist_ok=True)
    evidence = {"status": "running", "cases": [], "donation_withdrawal": False}
    with httpx.Client(base_url=os.environ.get("BASE_URL", "http://127.0.0.1:8000"), timeout=180) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        config = client.get("/api/config").json()
        for outcome, row in selected.items():
            wav = folder / (outcome + ".wav")
            wav.write_bytes((ROOT / "work/dev" / rows[row["sample_id"]]["audio_path"]).read_bytes())
            for extension in ("wav", "mp3", "flac"):
                path = folder / (outcome + "." + extension)
                started = time.perf_counter()
                candidates = [row]
                if outcome == "defer" and extension == "mp3":
                    candidates = sorted([r for r in eligible if calibration["t_low"] < r["mean_negative_bonafide_logit"] < calibration["t_warn"]], key=lambda r: abs(r["mean_negative_bonafide_logit"] - midpoint))[:30]
                attempts = []
                for candidate in candidates:
                    source = wav
                    if candidate is not row:
                        source = folder / "codec-candidate.wav"
                        source.write_bytes((ROOT / "work/dev" / rows[candidate["sample_id"]]["audio_path"]).read_bytes())
                    if extension != "wav":
                        subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-nostdin", "-loglevel", "error", "-y", "-i", str(source), str(path)], check=True)
                    result = client.post("/api/analyze", files={"file": (path.name, path.read_bytes())}, data={"analysis_consent": "true"}).json()
                    attempts.append({"sample_id": candidate["sample_id"], "actual": result["outcome_code"]})
                    if result["outcome_code"] == outcome:
                        break
                evidence["cases"].append({"format": extension, "expected": outcome, "actual": result["outcome_code"], "latency_seconds": time.perf_counter() - started,
                                          "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_sample_id": candidate["sample_id"], "fixture_selection_attempts": attempts, "timeline_slots": len(result["slots"]), "request_id": result["request_id"]})
                (ROOT / "work/E2E_REPORT.json").write_text(json.dumps(evidence, indent=2) + "\n")
                assert result["outcome_code"] == outcome, evidence["cases"][-1]
                assert result["windows"] and result["slots"] and result["donation_token"] is None
                assert all(slot["indicative_only"] and slot["start"] < slot["end"] <= result["duration"] for slot in result["slots"])
        invalid = client.post("/api/analyze", files={"file": ("corrupt.wav", b"corrupt")}, data={"analysis_consent": "true"}).json()
        assert invalid["outcome_code"] == "cannot" and invalid["file_score"] is None
        evidence["cases"].append({"expected": "cannot", "actual": invalid["outcome_code"], "reason": invalid["reason"]})
        output = io.BytesIO()
        wavfile.write(output, 16000, np.zeros(80000, np.int16))
        silence = folder / "cannot.wav"
        silence.write_bytes(output.getvalue())
        for extension in ("wav", "mp3", "flac"):
            path = folder / ("cannot." + extension)
            if extension != "wav":
                subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-nostdin", "-loglevel", "error", "-y", "-i", str(silence), str(path)], check=True)
            result = client.post("/api/analyze", files={"file": (path.name, path.read_bytes())}, data={"analysis_consent": "true"}).json()
            assert result["outcome_code"] == "cannot" and result["file_score"] is None
            evidence["cases"].append({"format": extension, "expected": "cannot", "actual": result["outcome_code"], "reason": result["reason"]})
        donation = client.post("/api/analyze", files={"file": ("opt-in.wav", (folder / "low.wav").read_bytes())}, data={"analysis_consent": "true", "donation_opt_in": "true"}).json()
        token = donation["donation_token"]
        assert token
        code = client.post("/api/donations", json={"token": token, "consent_version": config["consent_version"], "confirm": True}).json()["withdrawal_code"]
        assert client.delete("/api/donations/" + code).json()["status"] == "withdrawn"
        evidence["donation_withdrawal"] = True
    evidence["status"] = "pass"
    (ROOT / "work/E2E_REPORT.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"status": evidence["status"], "cases": len(evidence["cases"]), "donation_withdrawal": True}))


if __name__ == "__main__":
    main()
