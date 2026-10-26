"""Fine-tune official BAM on Korean edit-boundary composites (GPU instance).

Boundary target = every edit edge (AI or human insert). Spoof target = AI-insert frames only (BAM convention 1 = bona fide).
Fixed recipe chosen before evaluation: Adam lr 1e-5, 6 epochs, batch 8, 4 s crops on the 160 ms grid
(80 % of crops contain an edge), boundary loss weight 1.0, fp16 autocast. The final epoch is used; no checkpoint
selection on the held-out set.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path("/opt/exp")
sys.path.insert(0, str(ROOT / "code" / "bam_cfprf"))
from candidate_inference import BAMBackend  # noqa: E402

HOP, CROP_FRAMES = 2560, 25
CROP = HOP * CROP_FRAMES


def labels(item, n_frames):
    spoof = np.ones(n_frames, np.float32)
    for a, b in item["ai"]:
        for k in range(n_frames):
            lo, hi = k * HOP, (k + 1) * HOP
            if max(0, min(hi, b) - max(lo, a)) > HOP / 2:
                spoof[k] = 0.0
    bound = np.zeros(n_frames, np.float32)
    if item["edges"]:
        s, e = item["edges"]
        for k in (s // HOP, (e - 1) // HOP):
            if 0 <= k < n_frames:
                bound[k] = 1.0
    return spoof, bound


class Data(torch.utils.data.Dataset):
    def __init__(self, items, data_dir, seed):
        self.items, self.dir, self.rng = items, data_dir, random.Random(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        x, _ = sf.read(str(self.dir / f"{item['file_id']}.wav"), dtype="float32")
        n_frames = -(-len(x) // HOP)
        x = np.pad(x, (0, n_frames * HOP - len(x)))
        spoof, bound = labels(item, n_frames)
        if n_frames <= CROP_FRAMES:
            f0 = 0
        else:
            edges = [k for k in np.flatnonzero(bound)]
            if edges and self.rng.random() < 0.8:
                k = self.rng.choice(edges)
                f0 = min(max(0, k - self.rng.randint(2, CROP_FRAMES - 3)), n_frames - CROP_FRAMES)
            else:
                f0 = self.rng.randint(0, n_frames - CROP_FRAMES)
        xs = x[f0 * HOP:(f0 + CROP_FRAMES) * HOP]
        sp, bd = spoof[f0:f0 + CROP_FRAMES], bound[f0:f0 + CROP_FRAMES]
        mask = np.ones(CROP_FRAMES, np.float32)
        valid = len(sp)
        xs = np.pad(xs, (0, CROP - len(xs)))
        sp, bd = np.pad(sp, (0, CROP_FRAMES - valid), constant_values=1), np.pad(bd, (0, CROP_FRAMES - valid))
        mask[valid:] = 0
        return torch.from_numpy(xs), torch.from_numpy(sp), torch.from_numpy(bd), torch.from_numpy(mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bweight", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(20261024)
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    items = [json.loads(l) for l in (ROOT / "ft_data" / "index.jsonl").open(encoding="utf-8")]
    items = [i for i in items if i["split"] == "train"]
    if args.limit:
        items = items[:args.limit]
    backend = BAMBackend(ROOT / "code" / "BAM", ROOT / "assets" / "bam_checkpoint.zip", ROOT / "assets" / "WavLM-Large.pt",
                         Path("/tmp/bam_scratch"), device="cuda")
    backend.load()
    model = backend.model
    from models.loss import MaskCrossEnrtopyLoss, BalanceBCELoss
    ce, bce = MaskCrossEnrtopyLoss(), BalanceBCELoss()
    model.train()
    loader = torch.utils.data.DataLoader(Data(items, ROOT / "ft_data" / "train", 20261024), batch_size=args.batch,
                                         shuffle=True, num_workers=4, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    log, started = [], time.time()
    for epoch in range(1, args.epochs + 1):
        totals = []
        for step, (x, sp, bd, mask) in enumerate(loader):
            x, sp, bd, mask = (t.cuda(non_blocking=True) for t in (x, sp, bd, mask))
            with torch.cuda.amp.autocast():
                output, boundary = model(x)
            output, boundary = output.float(), boundary.float().clamp(1e-6, 1 - 1e-6)
            loss_b = bce(boundary, bd, mask=mask)
            loss_s = ce(output.transpose(-1, -2), sp.long(), mask=mask)
            loss = loss_s + args.bweight * loss_b
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            totals.append([loss.item(), loss_s.item(), loss_b.item()])
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.4f} spoof {loss_s.item():.4f} "
                      f"boundary {loss_b.item():.4f} {time.time() - started:.0f}s", flush=True)
        mean = np.mean(totals, axis=0).tolist()
        log.append(dict(epoch=epoch, loss=mean[0], spoof=mean[1], boundary=mean[2], seconds=time.time() - started))
        print(json.dumps(log[-1]), flush=True)
    torch.save(model.state_dict(), out / "bam_ft_final.pt")
    (out / "TRAIN_LOG.json").write_text(json.dumps(dict(args=vars(args), files=len(items), log=log,
                                                        identity=backend.identity), indent=1, default=str))
    print("saved", time.time() - started)


if __name__ == "__main__":
    main()
