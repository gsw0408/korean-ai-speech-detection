"""Boundary probabilities on held-out eval composites: official (zero-shot) BAM and fine-tuned BAM (GPU)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path("/opt/exp")
sys.path.insert(0, str(ROOT / "code" / "bam_cfprf"))
from candidate_inference import BAMBackend  # noqa: E402


def main():
    items = [json.loads(l) for l in (ROOT / "ft_data" / "index.jsonl").open(encoding="utf-8")]
    items = [i for i in items if i["split"] == "eval"]
    backend = BAMBackend(ROOT / "code" / "BAM", ROOT / "assets" / "bam_checkpoint.zip", ROOT / "assets" / "WavLM-Large.pt",
                         Path("/tmp/bam_scratch_eval"), device="cuda")
    backend.load()
    zero = {k: v.detach().clone() for k, v in backend.model.state_dict().items()}
    tuned = torch.load(ROOT / "out" / "bam_ft_final.pt", map_location="cuda")
    started = time.time()
    for name, state in (("bam0", zero), ("bamft", tuned)):
        backend.model.load_state_dict(state, strict=True)
        backend.model.eval()
        out = ROOT / "out" / name
        out.mkdir(parents=True, exist_ok=True)
        for item in items:
            x, _ = sf.read(str(ROOT / "ft_data" / "eval" / f"{item['file_id']}.wav"), dtype="float32")
            r = backend.score_chunk(x)
            np.savez(out / f"{item['file_id']}.npz", boundary=r["boundary"].astype(np.float32),
                     logits=r["logits"].astype(np.float32), hop=backend.hop_samples)
        print(name, "done", round(time.time() - started), flush=True)


if __name__ == "__main__":
    main()
