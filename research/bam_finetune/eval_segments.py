"""Segment held-out composites with several segmenters, judge every segment with unchanged NII, report metrics.

Segmenters: oracle edges, fixed 4 s windows (current service style), energy VAD, official BAM boundary (bam0) and
fine-tuned BAM boundary (bamft) at fixed thresholds. File score = max NII segment score (negative bona fide logit).
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = Path("/opt/exp")
SR, MIN_SEG = 16000, 0.5
TOLS = (0.1, 0.3, 0.5)
# boundary-score families and thresholds; PRIMARY thresholds were fixed before any result was seen
FAMILIES = {"bam0": (0.1, 0.3, 0.5), "bamft": (0.1, 0.3, 0.5), "bdr": (0.1, 0.3, 0.5),
            "trace_last": (1.5, 2.0, 2.5), "trace_mid": (1.5, 2.0, 2.5)}
PRIMARY = {"bam0": 0.5, "bamft": 0.5, "bdr": 0.5, "trace_last": 2.0, "trace_mid": 2.0}
K2_RECALL, K2_FALSE = 0.70, 2.0


def frame_db(x, win=400, hop=160):
    n = 1 + max(0, (len(x) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    f = x[np.minimum(idx, len(x) - 1)].astype(np.float64)
    return 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)


def vad_boundaries(x, min_pause=0.12):
    db = frame_db(x)
    floor, top = np.percentile(db, 10), np.percentile(db, 95)
    speech = db > floor + 0.3 * (top - floor)
    k, out, i, n = int(min_pause * SR / 160), [], 0, len(speech)
    while i < n:
        if not speech[i]:
            j = i
            while j < n and not speech[j]:
                j += 1
            if j - i >= k and i > 0 and j < n:
                out.extend([i * 160 + 200, j * 160 + 200])
            i = j
        else:
            i += 1
    return out


def peaks(prob, hop, threshold, min_gap=0.3):
    prob = np.asarray(prob, float)
    cand = [k for k in range(len(prob)) if prob[k] > threshold
            and (k == 0 or prob[k] >= prob[k - 1]) and (k == len(prob) - 1 or prob[k] >= prob[k + 1])]
    picked = []
    for k in sorted(cand, key=lambda k: -prob[k]):
        if all(abs(k - p) * hop >= min_gap * SR for p in picked):
            picked.append(k)
    return sorted(int(k * hop + hop // 2) for k in picked)


def segments_from(bounds, total):
    pts = sorted({0, total, *[int(b) for b in bounds if 0 < b < total]})
    merged = [pts[0]]
    for p in pts[1:]:
        if p - merged[-1] >= int(0.3 * SR) or p == total:
            merged.append(p)
    out = []
    for a, b in zip(merged[:-1], merged[1:]):
        if out and b - a < MIN_SEG * SR:
            out[-1][1] = b
        else:
            out.append([a, b])
    if len(out) > 1 and out[0][1] - out[0][0] < MIN_SEG * SR:
        out[1][0] = out[0][0]
        out.pop(0)
    return [tuple(s) for s in out]


def fixed_windows(total, win=64000, hop=32000):
    if total <= win:
        return [(0, total)]
    starts = list(range(0, total - win + 1, hop))
    if starts[-1] != total - win:
        starts.append(total - win)
    return [(s, s + win) for s in starts]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def main():
    items = [json.loads(l) for l in (ROOT / "ft_data" / "index.jsonl").open(encoding="utf-8")]
    items = [i for i in items if i["split"] == "eval"]
    opts = ort.SessionOptions()
    sess = ort.InferenceSession(str(ROOT / "assets" / "nii_frames_dyn.onnx"), sess_options=opts,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    provider = sess.get_providers()[0]
    cache = {}

    def nii(x, a, b):
        key = (id(x), a, b)
        if key not in cache:
            cache[key] = float(-sess.run(["logits"], {"wav": x[None, a:b].astype(np.float32)})[0][0, 1])
        return cache[key]

    records, started = [], time.time()
    for n_item, item in enumerate(items):
        x, _ = sf.read(str(ROOT / "ft_data" / "eval" / f"{item['file_id']}.wav"), dtype="float32")
        total, edges = len(x), item["edges"]
        variants = {"oracle": edges, "vad": vad_boundaries(x)}
        for name, thrs in FAMILIES.items():
            if not (ROOT / "out" / name).is_dir():
                continue
            d = np.load(ROOT / "out" / name / f"{item['file_id']}.npz")
            for thr in thrs:
                variants[f"{name}>{thr}"] = peaks(d["boundary"], int(d["hop"]), thr)
        seg_sets = {k: segments_from(v, total) for k, v in variants.items()}
        seg_sets["fixed4s"] = fixed_windows(total)
        cache.clear()
        for name, segs in seg_sets.items():
            segs = [s for s in segs if s[1] - s[0] >= MIN_SEG * SR] or [(0, total)]
            scores = [nii(x, a, b) for a, b in segs]
            top = int(np.argmax(scores))
            bounds = variants.get(name, [b for s in segs for b in s])
            rec = dict(file_id=item["file_id"], pair_id=item["pair_id"], component=item["component"], kind=item["kind"],
                       style=item["style"], length=item["length"], segmenter=name, file_score=scores[top],
                       n_segments=len(segs), duration=total / SR,
                       n_bounds=len([b for b in bounds if 0 < b < total]),
                       top_iou=iou(segs[top], edges) if edges else None)
            for tol in TOLS:
                rec[f"rec_{tol}"] = float(np.mean([any(abs(e - b) <= tol * SR for b in bounds) for e in edges])) if edges else None
            rec["false_bounds"] = sum(all(abs(b - e) > 0.5 * SR for e in edges) for b in bounds if 0 < b < total)
            records.append(rec)
        if n_item % 100 == 0:
            print(n_item, len(items), round(time.time() - started), provider, flush=True)
    summary = []
    groups = defaultdict(list)
    for r in records:
        groups[(r["style"], r["length"], r["segmenter"])].append(r)
    rng = np.random.default_rng(20261026)
    for (style, length, seg), rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if style is None:
            summary.append(dict(style=None, segmenter=seg, n=len(rows),
                                false_per_10s=float(np.mean([10 * r["false_bounds"] / r["duration"] for r in rows]))))
            continue
        att = {r["pair_id"]: r for r in rows if r["kind"] == "attack"}
        con = {r["pair_id"]: r for r in rows if r["kind"] == "control"}
        ids = sorted(set(att) & set(con))
        pos, neg = [att[i]["file_score"] for i in ids], [con[i]["file_score"] for i in ids]
        comps = defaultdict(list)
        for j, i in enumerate(ids):
            comps[att[i]["component"]].append(j)
        keys = sorted(comps)
        boot = []
        for _ in range(1000):
            pick = rng.choice(len(keys), len(keys), replace=True)
            idx = [j for k in pick for j in comps[keys[k]]]
            boot.append(auc([pos[j] for j in idx], [neg[j] for j in idx]))
        thr = np.quantile(neg, 0.9, interpolation="higher")
        everything = [r for r in rows if r["kind"] != "none"]
        summary.append(dict(style=style, length=length, segmenter=seg, n_pairs=len(ids), n_components=len(keys),
                            auc=auc(pos, neg), auc_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                            tpr_at_fpr10=float(np.mean(np.asarray(pos) > thr)),
                            rec_0_3=float(np.mean([r["rec_0.3"] for r in everything])),
                            rec_0_1=float(np.mean([r["rec_0.1"] for r in everything])),
                            false_per_10s=float(np.mean([10 * r["false_bounds"] / r["duration"] for r in everything])),
                            top_iou_attack=float(np.mean([att[i]["top_iou"] for i in ids]))))
    # pre-registered criteria: pooled over styles, per insert length, paired vs fixed4s by component bootstrap
    criteria = {}
    segs_all = sorted({r["segmenter"] for r in records})
    by = defaultdict(dict)
    for r in records:
        if r["kind"] in ("attack", "control"):
            by[(r["segmenter"], r["length"])][(r["pair_id"], r["style"], r["kind"])] = r
    for seg in segs_all:
        out = dict(segmenter=seg)
        k1 = True
        for L in (1.0, 2.0):
            cand, base = by[(seg, L)], by[("fixed4s", L)]
            units = sorted({(p, s) for (p, s, k) in cand if (p, s, "attack") in cand and (p, s, "control") in cand
                            and (p, s, "attack") in base and (p, s, "control") in base})
            comp_of = {u: cand[(u[0], u[1], "attack")]["component"] for u in units}
            def pooled(sel, table):
                return auc([table[(p, s, "attack")]["file_score"] for p, s in sel],
                           [table[(p, s, "control")]["file_score"] for p, s in sel])
            comps = defaultdict(list)
            for u in units:
                comps[comp_of[u]].append(u)
            keys = sorted(comps)
            rng2 = np.random.default_rng(20261027)
            diffs = []
            for _ in range(2000):
                pick = rng2.choice(len(keys), len(keys), replace=True)
                sel = [u for k in pick for u in comps[keys[k]]]
                diffs.append(pooled(sel, cand) - pooled(sel, base))
            lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
            out[f"auc_{L}"] = pooled(units, cand)
            out[f"auc_fixed4s_{L}"] = pooled(units, base)
            out[f"diff_ci95_{L}"] = [lo, hi]
            k1 = k1 and lo > 0
        edited = [r for r in records if r["segmenter"] == seg and r["kind"] in ("attack", "control")]
        none = [r for r in records if r["segmenter"] == seg and r["kind"] == "none"]
        # non-inferiority vs the confirmed fine-tuned BAM (fixed before BDR results): diff lower bound > -0.05
        ni = True
        for L in (1.0, 2.0):
            cand, ref = by[(seg, L)], by.get(("bamft>0.5", L), {})
            units = sorted({(p, s) for (p, s, k) in cand if all(t in cand and t in ref for t in ((p, s, "attack"), (p, s, "control")))})
            if not units:
                ni = False
                continue
            comps = defaultdict(list)
            for u in units:
                comps[cand[(u[0], u[1], "attack")]["component"]].append(u)
            keys = sorted(comps)
            rng3 = np.random.default_rng(20261028)
            f = lambda sel, t: auc([t[(p, s, "attack")]["file_score"] for p, s in sel], [t[(p, s, "control")]["file_score"] for p, s in sel])
            diffs = []
            for _ in range(2000):
                pick = rng3.choice(len(keys), len(keys), replace=True)
                sel = [u for k in pick for u in comps[keys[k]]]
                diffs.append(f(sel, cand) - f(sel, ref))
            out[f"diff_vs_bamft_{L}"] = f(units, cand) - f(units, ref)
            out[f"diff_vs_bamft_ci95_{L}"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
            ni = ni and out[f"diff_vs_bamft_ci95_{L}"][0] > -0.05
        out["NONINF_vs_bamft"] = bool(ni)
        out["edge_recall_0.3"] = float(np.mean([r["rec_0.3"] for r in edited]))
        out["false_per_10s_unedited"] = float(np.mean([10 * r["false_bounds"] / r["duration"] for r in none])) if none else None
        k2 = out["edge_recall_0.3"] >= K2_RECALL and (out["false_per_10s_unedited"] or 0) <= K2_FALSE
        family = seg.split(">")[0]
        out["primary"] = family in PRIMARY and seg == f"{family}>{PRIMARY[family]}"
        out.update(K1=bool(k1), K2=bool(k2), PASS=bool(k1 and k2))
        criteria[seg] = out
    (ROOT / "out" / "EVAL_RESULT.json").write_text(json.dumps(dict(provider=provider, files=len(items),
                                                                   wall_seconds=time.time() - started, summary=summary,
                                                                   criteria=criteria), indent=1))
    print("CRITERIA", json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
                                      if kk in ("auc_1.0", "auc_fixed4s_1.0", "diff_ci95_1.0", "auc_2.0", "auc_fixed4s_2.0",
                                                "diff_ci95_2.0", "edge_recall_0.3", "false_per_10s_unedited", "primary", "PASS",
                                                "diff_vs_bamft_1.0", "diff_vs_bamft_ci95_1.0", "diff_vs_bamft_2.0",
                                                "diff_vs_bamft_ci95_2.0", "NONINF_vs_bamft")}
                                  for k, v in criteria.items()}))
    with (ROOT / "out" / "eval_records.jsonl").open("w") as h:
        for r in records:
            h.write(json.dumps(r) + "\n")
    print("eval done", round(time.time() - started), provider)


if __name__ == "__main__":
    main()
