"""CPU cost of boundary inference on a 120 s file with 4 threads (service-like profile), one model per process.

usage: cpu_bench.py bdr|bam   -> prints JSON {model, warm_seconds[], peak_rss_bytes}
The 120 s input is concatenated held-out eval audio; boundary inference only (NII scoring is measured separately).
"""
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path("/opt/exp")
torch.set_num_threads(4)
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def audio120():
    items = [json.loads(l) for l in (ROOT / "ft_data" / "index.jsonl").open(encoding="utf-8")]
    parts, n = [], 0
    for it in items:
        if it["split"] == "eval" and it["kind"] == "none":
            x, _ = sf.read(str(ROOT / "ft_data" / "eval" / f"{it['file_id']}.wav"), dtype="float32")
            parts.append(x)
            n += len(x)
            if n >= 120 * 16000:
                break
    return np.concatenate(parts)[:120 * 16000]


def main():
    which = sys.argv[1]
    x = audio120()
    if which == "bdr":
        sys.argv = [sys.argv[0], "infer"]
        sys.path.insert(0, str(ROOT / "code"))
        import bdr
        label, wavroot = bdr.write_labels()
        hp = bdr.build(label, wavroot)
        state = torch.load(bdr.WORK / "bdr_ft_final.pt", map_location="cpu")
        for m in ("wav2vec", "front", "back"):
            hp[m].load_state_dict(state[m])
        mods = [hp[m].eval() for m in ("wav2vec", "front", "back")]
        win, shift = 20800, 10240
        starts = list(range(0, len(x) - win + 1, shift))

        def run():
            with torch.no_grad():
                for i in range(0, len(starts), 32):
                    b = torch.from_numpy(np.stack([x[s:s + win] for s in starts[i:i + 32]]))
                    mods[2](mods[1](mods[0](b)))
    else:
        sys.path.insert(0, str(ROOT / "code" / "bam_cfprf"))
        from candidate_inference import BAMBackend
        be = BAMBackend(ROOT / "code" / "BAM", ROOT / "assets" / "bam_checkpoint.zip", ROOT / "assets" / "WavLM-Large.pt",
                        Path("/tmp/bam_bench"), device="cpu")
        be.load()
        be.model.load_state_dict(torch.load(ROOT / "out" / "bam_ft_final.pt", map_location="cpu"))

        def run():
            be.score_chunk(x)
    times = []
    for _ in range(3):
        t = time.time()
        run()
        times.append(time.time() - t)
    print(json.dumps(dict(model=which, warm_seconds=times, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          threads=torch.get_num_threads(), audio_seconds=len(x) / 16000)))


if __name__ == "__main__":
    main()
