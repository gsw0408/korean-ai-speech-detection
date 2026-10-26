"""Experimental demo: edit-boundary segmentation (fine-tuned BAM) + per-segment NII judgement.

Research demonstration only. Uploaded audio is processed in a temporary directory and never stored.
Access requires the token in DEMO_TOKEN (query ?k=... or header X-Demo-Key).
"""
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

import demo_core as dc

TOKEN = os.environ["DEMO_TOKEN"]
CAL = json.loads((dc.ROOT / "demo" / "demo_calibration.json").read_text())
WHOLE = json.loads((dc.ROOT / "demo" / "demo_whole_prob.json").read_text())   # calib_whole_prob.py
MAX_BYTES = 25 * 1024 * 1024
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
LOCK = threading.Lock()
BOUNDARY = dc.Boundary(device="cuda")
NII = dc.Nii()
PAGE = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
V1_LABEL = {"warn": "AI 음성 의심 — 추가 확인 필요", "defer": "판단 유보", "low": "AI 음성 증거가 낮음"}


def check(request: Request):
    if request.query_params.get("k") != TOKEN and request.headers.get("x-demo-key") != TOKEN:
        raise HTTPException(status_code=404)


def platt(score, cal):
    """TRAIN-calibrated (Platt, equal AI/human prior) probability of AI speech, in percent."""
    z = max(-50.0, min(50.0, cal["coef"] * score + cal["intercept"]))
    return round(100 / (1 + math.exp(-z)), 1)


@app.get("/", response_class=HTMLResponse)
def page(request: Request):
    check(request)
    return PAGE


@app.post("/api/analyze")
def analyze(request: Request, file: UploadFile = File(...)):
    # Plain def: FastAPI runs it in a worker thread, so the blocking decode/BAM/NII work does not stall the event loop.
    check(request)
    data = file.file.read(MAX_BYTES + 1)
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="파일이 비었거나 25MB를 넘습니다.")
    started = time.time()
    too_short = JSONResponse({"error": "4~120초 길이의 음성만 분석합니다(앞뒤 무음 제외 4초 이상)."}, status_code=422)
    try:
        x, duration, trim0 = dc.decode(data)
    except dc.TooShort:
        return too_short
    except Exception:
        return JSONResponse({"error": "오디오를 읽을 수 없습니다(WAV/MP3/FLAC)."}, status_code=422)
    finally:
        del data
    if not 4 <= duration <= 120:
        return too_short
    with LOCK:
        t0 = time.time()
        prob = BOUNDARY.prob(x)
        t_bam = time.time() - t0
        bounds = dc.peaks(prob)
        segs = [g for g in dc.segments(bounds, len(x)) if g[1] - g[0] >= dc.MIN_SEG * dc.SR] or [(0, len(x))]
        t0 = time.time()
        scores = [NII.score(x[a:b]) for a, b in segs]
        v1_score, v1_code = NII.file_v1(x)
        t_nii = time.time() - t0
    off = trim0 / dc.SR   # front-end trim offset -> original file clock
    seg_out = [dict(start=round(off + a / dc.SR, 2), end=round(off + b / dc.SR, 2), score=round(s, 3),
                    ai_prob=platt(s, CAL["prob"]), flagged=bool(s >= CAL["t_seg"])) for (a, b), s in zip(segs, scores)]
    top = max(scores)
    return {
        "duration": round(duration, 2),
        "whole_file": {"code": v1_code, "label": V1_LABEL[v1_code], "score": round(v1_score, 3),
                       "ai_prob": platt(v1_score, WHOLE)},
        "partial": {"suspected": bool(top >= CAL["t_file"]), "max_ai_prob": max(s["ai_prob"] for s in seg_out),
                    "flagged_segments": sum(s["flagged"] for s in seg_out)},
        "segments": seg_out,
        "boundaries": [round(off + b / dc.SR, 2) for b in bounds],
        "timing": {"boundary_s": round(t_bam, 2), "nii_s": round(t_nii, 2), "total_s": round(time.time() - started, 2)},
        "calibration": {"scope": CAL["scope"], "t_seg": CAL["t_seg"], "t_file": CAL["t_file"],
                        "prob": {k: CAL["prob"][k] for k in ("method", "features", "prior", "cv")},
                        "whole_prob": {k: WHOLE[k] for k in ("method", "prior", "cv")}},
    }
