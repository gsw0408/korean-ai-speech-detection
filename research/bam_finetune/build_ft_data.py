"""Build Korean edit-boundary fine-tuning and held-out evaluation composites (WSL BAM prefix; Dataset v2 TRAIN only).

Goal: BAM must mark every edit edge whether the insert is AI or human; NII judges segments later.
Split: TRAIN real-host components are hashed into FIT (about 80%) and HOLD (about 20%). FIT builds the training set,
HOLD builds the evaluation set (different seed). No DEV/CALIB/FINAL access.
Outputs int16 16 kHz WAVs after the service front-end, plus JSONL with edit edges and AI intervals (samples).
"""
import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bam_cause_20261024"))
import build_factorial as bf  # noqa: E402

SR = bf.SR


def hold(component):
    return hashlib.sha256(f"{component}|ft-split-20261024".encode()).digest()[0] < 52  # about 20 %


def load_pairs():
    rows = [json.loads(l) for l in bf.KO_MANIFEST.open(encoding="utf-8")]
    pairs, by_rec = defaultdict(dict), defaultdict(list)
    for r in rows:
        if r.get("allocation_role") != "TRAIN":
            raise ValueError("STOP: non-TRAIN row")
        pairs[r["pair_id"]]["fake" if r["label_fake"] else "real"] = r
        if not r["label_fake"]:
            by_rec[r["recording"]].append(r)
    return pairs, by_rec


def write(out, split, fid, x, edges, ai, meta, index):
    y = np.clip(np.rint(np.asarray(x) * 32768), -32768, 32767).astype(np.int16)
    sf.write(str(out / split / f"{fid}.wav"), y, SR, subtype="PCM_16")
    index.append(dict(meta, file_id=fid, split=split, edges=[int(e) for e in edges], ai=[[int(a), int(b)] for a, b in ai],
                      samples=int(len(y))))


def make(front, style, host, sib, insert, seed):
    audio, truth = bf.build(style, host, sib, insert, front, seed)
    x = np.clip(np.rint(audio * 32768), -32768, 32767) / 32768.0
    t0, _, _ = front.edge_trim(x)
    return front.frontend(x), (truth[0] - t0, truth[1] - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval-per-engine", type=int, default=20)
    args = ap.parse_args()
    for split in ("train", "eval"):
        (args.out / split).mkdir(parents=True, exist_ok=True)
    front = bf.load_frontend()
    pairs, by_rec = load_pairs()
    ids = sorted(pairs)
    fit = [p for p in ids if not hold(pairs[p]["real"]["component_id"])]
    held = [p for p in ids if hold(pairs[p]["real"]["component_id"])]
    index, cache = [], {}

    def clip(row):
        if row["sample_id"] not in cache:
            cache[row["sample_id"]] = bf.trim(front, bf.read16(bf.KO_DATA / row["audio_path"]))
        return cache[row["sample_id"]]

    def sibling_of(pid, rng):
        real = pairs[pid]["real"]
        sib = [r for r in by_rec[real["recording"]] if r["sample_id"] != real["sample_id"]]
        return clip(rng.choice(sib)) if sib else None

    # training set: for every FIT pair one AI-insert and one human-insert composite with random style/length
    rng = random.Random(20261024)
    for i, pid in enumerate(fit):
        host, fake = clip(pairs[pid]["real"]), clip(pairs[pid]["fake"])
        sib = sibling_of(pid, rng)
        donor = clip(pairs[fit[(i + 7) % len(fit)]]["real"])
        meta = dict(pair_id=pid, component=pairs[pid]["real"]["component_id"], engine=pairs[pid]["fake"]["engine"])
        for kind, src in (("attack", fake), ("control", donor)):
            style = rng.choice(("pause", "mid", "ps") if sib is not None else ("mid",))
            L = rng.choice((0.5, 1.0, 2.0, 3.0))
            n = int(L * SR)
            if len(host) < n + 2 * SR or len(src) < n:
                continue
            x, truth = make(front, style, host, sib, bf.centre(src, n), bf.stable_seed(pid, kind, "fit"))
            write(args.out, "train", f"{pid}__{style}__{L}__{kind}", x, truth, [truth] if kind == "attack" else [],
                  dict(meta, style=style, length=L, kind=kind), index)
        if rng.random() < 0.15:
            write(args.out, "train", f"{pid}__none", front.frontend(host), [], [], dict(meta, style=None, length=None, kind="none"), index)

    # held-out evaluation set: balanced engines, full factorial of style x length x kind plus unedited host
    by_engine = defaultdict(list)
    for pid in held:
        by_engine[pairs[pid]["fake"]["engine"]].append(pid)
    erng = random.Random(20261026)
    chosen = []
    for e in sorted(by_engine):
        ids_e = by_engine[e][:]
        erng.shuffle(ids_e)
        chosen.extend(ids_e[:args.eval_per_engine])
    for i, pid in enumerate(chosen):
        host, fake = clip(pairs[pid]["real"]), clip(pairs[pid]["fake"])
        sib = sibling_of(pid, erng)
        donor = clip(pairs[chosen[(i + 1) % len(chosen)]]["real"])
        meta = dict(pair_id=pid, component=pairs[pid]["real"]["component_id"], engine=pairs[pid]["fake"]["engine"])
        write(args.out, "eval", f"{pid}__none", front.frontend(host), [], [], dict(meta, style=None, length=None, kind="none"), index)
        for L in (1.0, 2.0):
            n = int(L * SR)
            if len(host) < n + 2 * SR or min(len(fake), len(donor)) < n:
                continue
            for style in ("pause", "mid", "ps"):
                if style != "mid" and sib is None:
                    continue
                for kind, src in (("attack", fake), ("control", donor)):
                    x, truth = make(front, style, host, sib, bf.centre(src, n), bf.stable_seed(pid, L, "hold"))
                    write(args.out, "eval", f"{pid}__{style}__{L}__{kind}", x, truth, [truth] if kind == "attack" else [],
                          dict(meta, style=style, length=L, kind=kind), index)
    with (args.out / "index.jsonl").open("w", encoding="utf-8") as h:
        for item in index:
            h.write(json.dumps(item) + "\n")
    comps = lambda s: len({i["component"] for i in index if i["split"] == s})
    print(json.dumps(dict(train=sum(i["split"] == "train" for i in index), eval=sum(i["split"] == "eval" for i in index),
                          train_hours=sum(i["samples"] for i in index if i["split"] == "train") / SR / 3600,
                          eval_hours=sum(i["samples"] for i in index if i["split"] == "eval") / SR / 3600,
                          train_components=comps("train"), eval_components=comps("eval"))))


if __name__ == "__main__":
    main()
