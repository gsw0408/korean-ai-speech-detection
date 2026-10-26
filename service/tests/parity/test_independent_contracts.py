import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pytest
from pipeline.detector import NiiOnnxDetector
from pipeline.preprocess import CannotAnalyze, decode_canonical, prepare, scoring_windows

ROOT = Path(__file__).resolve().parents[2]
# The DEV audio set is not distributed with this repository; these contracts run only where it is present.
pytestmark = pytest.mark.skipif(not (ROOT / "work/dev_manifest.json").exists(), reason="private DEV data not available")
# Private source identities are supplied through the gitignored DEV fixture.
@pytest.fixture(scope="module")
def rejected_case_ids():
    fixture = json.loads((ROOT / "work/dev_contract_cases.json").read_bytes())
    assert fixture["schema"] == "dev_contract_cases_v1"
    rejected = fixture["rejected_sample_ids"]
    assert len(rejected) == len(set(rejected)) == 4
    return rejected


@pytest.fixture(scope="module")
def dev():
    rows = {r["sample_id"]: r for r in json.loads((ROOT / "work/dev_manifest.json").read_bytes())}
    reference = json.loads((ROOT / "work/nii.json").read_bytes())
    expected = {r["sample_id"]: r for r in reference["records"]}
    detector = NiiOnnxDetector(ROOT / "models/wav2vec2-small-antideepfake.onnx", reference["checkpoint_sha256"])
    return rows, expected, detector


@pytest.mark.parametrize("case_index", range(4), ids=[f"case-{i + 1}" for i in range(4)])
def test_rejected_service_inputs_still_have_reproducible_research_scores(
        dev, ffmpeg, tmp_path, rejected_case_ids, case_index):
    rows, expected, detector = dev
    sample_id = rejected_case_ids[case_index]
    body = (ROOT / "work/dev" / rows[sample_id]["audio_path"]).read_bytes()
    with pytest.raises(CannotAnalyze, match="insufficient_active_speech"):
        prepare(body, ffmpeg=ffmpeg, temp_dir=tmp_path)
    assert not list(tmp_path.iterdir())
    numerical = scoring_windows(*decode_canonical(body, ffmpeg=ffmpeg, temp_dir=tmp_path))
    record = expected[sample_id]
    assert numerical.starts == record["window_starts"]
    assert [hashlib.sha256(w.tobytes()).hexdigest() for w in numerical.windows] == record["window_input_sha256s"]
    scores = detector.score_windows(numerical.windows)
    assert np.max(np.abs(np.asarray(scores) - record["window_scores"])) <= 1e-5
    assert abs(float(np.mean(scores)) - record["mean_negative_bonafide_logit"]) <= 1e-5


def test_46_original_accepted_files_remain_accepted(dev, ffmpeg, tmp_path, rejected_case_ids):
    rows, expected, detector = dev
    groups = defaultdict(list)
    for row in rows.values():
        groups[(row["engine"], row["label_fake"])].append(row)
    original_50 = [row for key in sorted(groups) for row in sorted(groups[key], key=lambda row: row["sha256"])[:5]]
    passed = [row for row in original_50 if row["sample_id"] not in rejected_case_ids]
    assert len(passed) == 46
    for row in passed:
        sample_id = row["sample_id"]
        audio = prepare((ROOT / "work/dev" / rows[sample_id]["audio_path"]).read_bytes(), ffmpeg=ffmpeg, temp_dir=tmp_path)
        assert [hashlib.sha256(w.tobytes()).hexdigest() for w in audio.windows] == expected[sample_id]["window_input_sha256s"]
    assert not list(tmp_path.iterdir())
