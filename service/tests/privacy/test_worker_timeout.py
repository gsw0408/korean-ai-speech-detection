from pathlib import Path
import pytest
from pipeline.service import AnalysisWorker
from pipeline.preprocess import CannotAnalyze
from conftest import wav_bytes


def test_real_timeout_kills_inference_cleans_files_and_recovers(ffmpeg, tmp_path):
    root = Path(__file__).resolve().parents[2]
    worker = AnalysisWorker(root, "bundles/nii-small-v1", 4, ffmpeg, tmp_path)
    try:
        original_pid = worker.process.pid
        with pytest.raises(CannotAnalyze, match="analysis_timeout"):
            worker.analyze(wav_bytes(seconds=60), .000001)
        assert not list(tmp_path.iterdir())
        assert worker.process.pid != original_pid and worker.process.is_alive()
        result = worker.analyze(wav_bytes(), 60)
        assert result["outcome_code"] != "cannot"
        assert not list(tmp_path.iterdir())
    finally:
        worker.close()
