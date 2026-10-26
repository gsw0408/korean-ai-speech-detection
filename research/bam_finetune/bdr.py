"""BDR (Cai et al., speechbrain_PartialFake) fine-tuning on Korean edit composites + boundary inference (GPU).

Uses the repository's own model modules (wav2vec2-base-960h -> MelResNet -> Transformer) and its training data
pipeline (segment change points + online concatenation of unedited speech) without the MUSAN/RIR augmentations.
Every edit edge (AI or human insert) is a change point. Fixed recipe chosen before evaluation:
  pretrained BDR checkpoint available -> Adam 2e-5 constant, 30 epochs; otherwise Adam 1e-4 + Noam(1600), 40 epochs.
Final epoch used. Inference: 1.30 s windows, 0.64 s shift, boundary prob = 1 - sigmoid(output), averaged per 20 ms.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path("/opt/speechbrain_PartialFake/ADD_2023")
sys.path.insert(0, str(REPO))
import speechbrain as sb  # noqa: E402
from hyperpyyaml import load_hyperpyyaml  # noqa: E402

ROOT = Path("/opt/exp")
WORK = ROOT / "bdr_work"
YAML = """
seed: 1994
__set_seed: !apply:torch.manual_seed [!ref <seed>]
feature: wav2vec
sample_rate: 16000
embedding_size: 128
wav_length: 1.28
subsampling: 2
time_resolution: !ref 0.01*<subsampling>
positive_ratio: 0.2
concat_ratio: 0.4
neighborhood_size: 2
add_train: {label}
train_wav_folder: {wavroot}
wav2vec: !new:speechbrain.lobes.models.huggingface_wav2vec.HuggingFaceWav2Vec2
   source: {w2v}
   output_norm: True
   freeze: False
   save_path: {work}/w2v_save
front: !new:speechbrain.nnet.containers.Sequential
    melnet: !new:speechbrain.lobes.models.ResNet.MelResNet.MelResNet
        in_planes: 768
        planes: 512
        embed_dim: !ref <embedding_size>
        res_blocks: 12
    linear: !new:torch.nn.Linear
        in_features: !ref <embedding_size>
        out_features: !ref <embedding_size>
    norm: !new:speechbrain.nnet.normalization.BatchNorm1d
        input_size: !ref <embedding_size>
back: !new:speechbrain.lobes.models.diarization.EEND.EEND_TransformerEncoder
    num_layers: 2
    nhead: 4
    d_ffn: 1024
    d_model: !ref <embedding_size>
    dropout: 0.5
    n_spk: 1
"""


def write_labels():
    WORK.mkdir(parents=True, exist_ok=True)
    wavroot = WORK / "train_root"
    (wavroot).mkdir(exist_ok=True)
    if not (wavroot / "wav").exists():
        os.symlink(ROOT / "ft_data" / "train", wavroot / "wav")
    items = [json.loads(l) for l in (ROOT / "ft_data" / "index.jsonl").open(encoding="utf-8")]
    lines = []
    for it in items:
        if it["split"] != "train":
            continue
        end = it["samples"] / 16000
        if it["edges"]:
            s, e = it["edges"][0] / 16000, it["edges"][1] / 16000
            mid = "F" if it["kind"] == "attack" else "T"
            info = f"0.0000-{s:.4f}-T/{s:.4f}-{e:.4f}-{mid}/{e:.4f}-{end:.4f}-T"
            lab = 0 if it["kind"] == "attack" else 1
        else:
            info, lab = f"0.0000-{end:.4f}-T", 1
        lines.append(f"{it['file_id']} {info} {lab}")
    label = WORK / "label.txt"
    label.write_text("\n".join(lines) + "\n")
    return label, wavroot


def build(label, wavroot):
    text = YAML.format(label=label, wavroot=wavroot, w2v=ROOT / "assets" / "wav2vec2-base-960h", work=WORK)
    return load_hyperpyyaml(text)


def load_pretrained(hp):
    files = {m: glob.glob(str(ROOT / "assets" / "bdr_ckpt" / "**" / f"{m}.ckpt"), recursive=True) for m in ("wav2vec", "front", "back")}
    if not all(files.values()):
        return None
    report = {}
    for m, paths in files.items():
        state = torch.load(sorted(paths)[0], map_location="cpu")
        missing, unexpected = hp[m].load_state_dict(state, strict=False)
        report[m] = dict(path=sorted(paths)[0], missing=len(missing), unexpected=len(unexpected))
    return report


def train(args):
    label, wavroot = write_labels()
    hp = build(label, wavroot)
    pre = load_pretrained(hp)
    from dataset import train_dataio_prep_bdr_dc
    data = train_dataio_prep_bdr_dc(hp)
    loader = sb.dataio.dataloader.make_dataloader(data, batch_size=64, shuffle=True, num_workers=4)
    modules = torch.nn.ModuleDict(dict(wav2vec=hp["wav2vec"], front=hp["front"], back=hp["back"])).cuda().train()
    epochs, lr = (30, 2e-5) if pre else (40, 1e-4)
    opt = torch.optim.Adam(modules.parameters(), lr=lr, weight_decay=2e-6)
    noam = None if pre else sb.nnet.schedulers.NoamScheduler(lr_initial=lr, n_warmup_steps=1600)
    bce = torch.nn.BCEWithLogitsLoss()
    log, started, step = [], time.time(), 0
    for epoch in range(1, epochs + 1):
        losses = []
        for batch in loader:
            sig, _ = batch.sig
            target, _ = batch.target
            sig, target = sig.cuda(), target.cuda()
            out = modules["back"](modules["front"](modules["wav2vec"](sig)))
            loss = bce(out, 1 - target[:, :-1, :])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modules.parameters(), 5.0)
            opt.step()
            if noam:
                noam(opt)
            losses.append(loss.item())
            step += 1
        log.append(dict(epoch=epoch, loss=float(np.mean(losses)), steps=step, seconds=time.time() - started))
        print(json.dumps(log[-1]), flush=True)
    torch.save({k: v.state_dict() for k, v in modules.items()}, WORK / "bdr_ft_final.pt")
    (ROOT / "out" / "BDR_TRAIN_LOG.json").write_text(json.dumps(dict(pretrained=pre, epochs=epochs, lr=lr, log=log,
                                                                     train_utts=len(data)), indent=1, default=str))


def infer(args):
    label, wavroot = write_labels()
    hp = build(label, wavroot)
    state = torch.load(WORK / "bdr_ft_final.pt", map_location="cpu")
    for m in ("wav2vec", "front", "back"):
        hp[m].load_state_dict(state[m])
    modules = [hp[m].cuda().eval() for m in ("wav2vec", "front", "back")]
    win, shift, hop = 20800, 10240, 320
    items = [json.loads(l) for l in (ROOT / "ft_data" / "index.jsonl").open(encoding="utf-8")]
    out_dir = ROOT / "out" / "bdr"
    out_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        if it["split"] != "eval":
            continue
        x, _ = sf.read(str(ROOT / "ft_data" / "eval" / f"{it['file_id']}.wav"), dtype="float32")
        if len(x) < win:
            x = np.pad(x, (0, win - len(x)))
        starts = list(range(0, len(x) - win + 1, shift))
        if starts[-1] != len(x) - win:
            starts.append(len(x) - win)
        batch = torch.from_numpy(np.stack([x[s:s + win] for s in starts])).cuda()
        with torch.no_grad():
            o = modules[2](modules[1](modules[0](batch))).squeeze(-1)
        p = (1 - torch.sigmoid(o)).cpu().numpy()
        n = len(x) // hop + 1
        acc, cnt = np.zeros(n), np.zeros(n)
        for s, row in zip(starts, p):
            f0 = s // hop
            k = min(len(row), n - f0)
            acc[f0:f0 + k] += row[:k]
            cnt[f0:f0 + k] += 1
        prob = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0).astype(np.float32)
        np.savez(out_dir / f"{it['file_id']}.npz", boundary=prob, hop=hop)
    print("bdr infer done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "infer"])
    a = ap.parse_args()
    train(a) if a.stage == "train" else infer(a)
