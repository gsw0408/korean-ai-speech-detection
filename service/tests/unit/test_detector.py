from unittest.mock import Mock

import numpy as np
import pytest

from pipeline.detector import NiiOnnxDetector


def fake_detector():
    detector = NiiOnnxDetector.__new__(NiiOnnxDetector)
    detector.input_name = "audio"
    detector.session = Mock()
    detector.session.run.side_effect = lambda _, inputs: [np.tile([7, -3], (len(inputs["audio"]), 1)).astype(np.float32)]
    return detector


def test_negative_class_one_and_batch_eight():
    detector = fake_detector()
    assert detector.score_windows([np.ones(64000)] * 19) == [3] * 19
    assert [call.args[1]["audio"].shape for call in detector.session.run.call_args_list] == [(8, 64000), (8, 64000), (3, 64000)]


@pytest.mark.parametrize("output", [np.zeros((1, 3)), np.full((1, 2), np.nan)])
def test_bad_logits_rejected(output):
    detector = fake_detector()
    detector.session.run.side_effect = None
    detector.session.run.return_value = [output]
    with pytest.raises(ValueError, match="logits"):
        detector.score_windows([np.ones(64000)])
