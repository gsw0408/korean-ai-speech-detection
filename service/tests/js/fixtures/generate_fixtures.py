"""Regenerate the Option C UI fixtures from the real service path with synthetic window scores.

Run from webservice/:  .venv\\Scripts\\python.exe -B tests\\js\\fixtures\\generate_fixtures.py
Uses tests/api/optionc_harness.py (real FastAPI app + real pipeline.service._worker; only prepare() and the
detector are replaced). No audio, no DEV/CALIB/FINAL data. request_id is replaced by a constant.
"""
import copy
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "tests/api")]
import optionc_harness as h  # noqa: E402

SR = 16000


def n_windows(seconds):
    return len(h.audio_windows.starts(int(seconds * SR)))


def spec(seconds, scores):
    assert len(scores) == n_windows(seconds), (seconds, len(scores), n_windows(seconds))
    return {"trimmed_samples": int(seconds * SR), "scores": scores}


def peaks(seconds, base, at):
    scores = [base] * n_windows(seconds)
    for index, value in at.items():
        scores[index] = value
    return scores


# Fixed synthetic scores (never derived from thresholds). Window counts: 10 s=4, 26 s=12, 60 s=29, 120 s=59.
CASES = {
    h.V1: {
        "v1_warn": spec(26, [1.5] * 12),
        "v1_low": spec(26, peaks(26, -4.0, {5: 3.0})),
    },
    h.PARTIAL2: {
        "p2_suspected_one_segment": spec(26, peaks(26, -2.6, {6: 3.0})),
        "p2_suspected_several": spec(60, peaks(60, -4.0, {3: 3.5, 10: 3.0, 11: 3.1, 15: 1.0, 16: 1.2, 20: 4.2, 25: 2.6})),
        "p2_low_suspected": spec(26, peaks(26, -4.0, {5: 3.0})),
        "p2_defer_suspected": spec(26, peaks(26, -1.6, {2: 0.4, 8: 2.5})),
        "p2_warn_suspected": spec(26, peaks(26, 2.0, {7: 4.0})),
        "p2_not_found": spec(26, peaks(26, -3.0, {5: 1.5})),
        "p2_not_applicable": spec(8, [4.5, 4.0, 5.0]),
        "p2_extrapolated_n4_suspected": spec(10, [-3.5, -3.5, -3.5, 5.0]),
        "p2_extrapolated_n59_not_found": spec(120, peaks(120, -3.0, {30: 2.5})),
    },
}


def main():
    out_dir = HERE
    patches = h.patched()
    for item in patches:
        item.start()
    written = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for bundle, cases in CASES.items():
                with h.client_for(bundle, Path(tmp) / bundle.split("/")[-1]) as client:
                    for name, case in cases.items():
                        response = client.post("/api/analyze", files={"file": ("case.wav", json.dumps(case).encode(), "audio/wav")},
                                               data={"analysis_consent": "true"})
                        assert response.status_code == 200, name
                        body = response.json()
                        body["request_id"] = "fixture"
                        written[name] = body
    finally:
        for item in patches:
            item.stop()
    unknown = copy.deepcopy(written["p2_not_found"])
    unknown["partial"]["status"] = "inconclusive"
    unknown["display"]["partial"]["status"] = "inconclusive"
    written["p2_unknown_status"] = unknown
    degraded = copy.deepcopy(written["p2_suspected_several"])
    del degraded["display"]
    written["p2_display_missing_suspected"] = degraded
    for name, body in written.items():
        (out_dir / f"{name}.json").write_text(json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        partial = body.get("partial") or {}
        print(f"{name:34s} outcome={body['outcome_code']:6s} partial={partial.get('status')!s:15s} "
              f"segments={len(partial.get('segments', []))} display={'display' in body}")


if __name__ == "__main__":
    main()
