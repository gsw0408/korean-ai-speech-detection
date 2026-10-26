"""Demo calibration step 1 (bam venv, GPU): windowed fine-tuned BAM boundary probabilities on TRAIN composites only."""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/opt/exp/demo")
import demo_core as dc  # noqa: E402

out = dc.ROOT / "out" / "demo_calib"
out.mkdir(parents=True, exist_ok=True)
items = [json.loads(l) for l in (dc.ROOT / "ft_data" / "index.jsonl").open(encoding="utf-8")]
items = [i for i in items if i["split"] == "train"]
b = dc.Boundary()
for k, it in enumerate(items):
    x, _ = sf.read(str(dc.ROOT / "ft_data" / "train" / f"{it['file_id']}.wav"), dtype="float32")
    np.save(out / f"{it['file_id']}.npy", b.prob(x).astype(np.float32))
    if k % 500 == 0:
        print(k, len(items), flush=True)
print("calib boundaries done")
