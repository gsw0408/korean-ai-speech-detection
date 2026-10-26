"""NII scoring matches evaluator.nii_predictor and evaluator.score_file."""

from typing import Protocol

import numpy as np
import onnxruntime as ort

from .path_guard import sha256_file


class Detector(Protocol):
    def score_windows(self, windows: list) -> list:
        ...


class NiiOnnxDetector:
    def __init__(self, model, expected_sha256, *, threads=4):
        if sha256_file(model) != expected_sha256:
            raise ValueError("Detector SHA-256 mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        signature = self.session.get_inputs()[0]
        if signature.shape[-1] != 64000:
            raise ValueError("NII window signature mismatch")
        self.input_name = signature.name

    def score_windows(self, windows):
        if not windows:
            raise ValueError("No detector windows")
        logits = []
        for start in range(0, len(windows), 8):
            batch = np.stack(windows[start:start + 8]).astype(np.float32)
            if batch.shape[1:] != (64000,) or not np.isfinite(batch).all():
                raise ValueError("Invalid detector input")
            logits.extend(self.session.run(None, {self.input_name: batch})[0])
        logits = np.asarray(logits)
        if logits.shape != (len(windows), 2) or not np.isfinite(logits).all():
            raise ValueError("Invalid detector logits")
        return (-logits[:, 1]).tolist()
