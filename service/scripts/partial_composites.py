"""Deterministic insertion composites from TRAIN/DEV clips; no generation."""

import hashlib
import io
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.io import wavfile


SR, GAP, XF, MAX_S = 16000, 4800, 160, 119.0
ROLES = {"train": "TRAIN", "dev": "DEV_ASSESSMENT"}
SAME = {"train": (), "dev": ("xtts", "openvoice", "seedvc")}
GENERIC = {"train": (), "dev": ("supertonic", "melo")}


def _split(split):
    if split not in ROLES:
        raise ValueError("STOP: only TRAIN/DEV clips are authorized")


def _root(root):
    root = Path(root).resolve()
    if any("calib" in part.lower() or "final" in part.lower() for part in root.parts):
        raise ValueError("STOP: protected dataset root")
    return root


def _audio_path(root, row):
    split = row.get("split")
    _split(split)
    if row.get("allocation_role") != ROLES[split]:
        raise ValueError("STOP: unauthorized allocation role")
    raw = row["audio_path"]
    supplied = PurePosixPath(raw)
    if ("\\" in raw or supplied.is_absolute() or ".." in supplied.parts
            or supplied.parts[:2] != ("audio", split)
            or any("calib" in part.lower() or "final" in part.lower()
                   for part in supplied.parts)):
        raise ValueError("STOP: unauthorized audio path")
    root = _root(root)
    path = (root / Path(*supplied.parts)).resolve()
    if not path.is_relative_to(root / "audio" / split):
        raise ValueError("STOP: audio path escapes the authorized split")
    return path


def load_rows(root, split):
    _split(split)
    root = _root(root)
    path = (root / "manifests" / f"{split}_required.jsonl").resolve()
    if not path.is_relative_to(root / "manifests"):
        raise ValueError("STOP: manifest path escapes the dataset")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ids = set()
    for row in rows:
        if row.get("split") != split or row.get("allocation_role") != ROLES[split]:
            raise ValueError("STOP: unexpected split or allocation role")
        _audio_path(root, row)
        if row["sample_id"] in ids or row["label_fake"] not in (0, 1):
            raise ValueError("STOP: duplicate ID or invalid label")
        ids.add(row["sample_id"])
    return rows


def clip_seconds(row):
    return (row["bytes"] - 44) / 2 / SR


def read(root, row):
    data = _audio_path(root, row).read_bytes()
    if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("STOP: audio hash or byte count mismatch")
    rate, samples = wavfile.read(io.BytesIO(data))
    if rate != SR or samples.dtype != np.int16 or samples.ndim != 1:
        raise ValueError("Non-canonical TRAIN/DEV clip")
    return samples.astype(np.float64) / 32768.0


def recipes(root, split, lengths, per_len=2, seed=20261012, attacks=True):
    rows = load_rows(root, split)
    rng = random.Random(f"{seed}|{split}")
    real, fakes, speakers = defaultdict(list), defaultdict(list), defaultdict(set)
    for row in rows:
        (fakes if row["label_fake"] else real)[row["recording"]].append(row)
        if not row["label_fake"]:
            speakers[row["recording"]].add(row["speaker_id"])
    output = []
    for recording in sorted(real):
        clips = sorted(real[recording], key=lambda row: row["derivation"]["start_frame"])
        durations = [clip_seconds(clip) + GAP / SR for clip in clips]
        same = sorted((fake for fake in fakes[recording]
                       if fake["engine"] in SAME[split]
                       and fake["voice"] in speakers[recording]),
                      key=lambda fake: fake["sample_id"])
        generic = sorted((fake for fake in fakes[recording]
                          if fake["engine"] in GENERIC[split]),
                         key=lambda fake: fake["sample_id"])
        for length in lengths:
            feasible = [index for index in range(len(clips))
                        if sum(durations[index:]) >= length]
            for index in range(per_len if feasible else 0):
                start, sequence, total = rng.choice(feasible), [], 0.0
                for clip, duration in zip(clips[start:], durations[start:]):
                    sequence.append(clip["sample_id"])
                    total += duration
                    if total >= length:
                        break
                if total > MAX_S or len(sequence) < 3:
                    continue
                base = f"{split}-{recording}-L{length}-{index}"
                output.append(dict(id=base, base=base, split=split, recording=recording,
                                   kind="real", seq=sequence, insert=None))
                if not (attacks and same):
                    continue
                fake = rng.choice(same)
                plans = [(position, "same_full", fake, None)
                         for position in ("begin", "middle", "end")]
                plans.append(("middle", "same_2s", fake, 2.0))
                if generic:
                    plans.append(("middle", "generic_full", rng.choice(generic), None))
                for position, kind, fake, crop in plans:
                    if total + (crop or clip_seconds(fake)) + GAP / SR > MAX_S:
                        continue
                    at = {"begin": 1, "middle": len(sequence) // 2,
                          "end": len(sequence) - 1}[position]
                    output.append(dict(id=f"{base}-{kind}-{position}", base=base,
                                       split=split, recording=recording, kind="attack",
                                       seq=sequence, insert=dict(at=at, fake=fake["sample_id"],
                                       engine=fake["engine"], crop=crop, type=kind,
                                       position=position)))
    return output


def _active_db(samples, frontend):
    start, end, _ = frontend.edge_trim(samples)
    return frontend.active_level_dbfs(samples[start:end]) if end > start else -200.0


def _room_tone(samples):
    # Quietest 0.3 s of the preceding REAL clip, sampled every 10 ms.
    if len(samples) < GAP:
        raise ValueError("REAL clip is too short for the common pause")
    cumulative = np.concatenate([[0.0], np.cumsum(samples * samples)])
    index = int(np.argmin((cumulative[GAP:] - cumulative[:-GAP])[::160])) * 160
    return samples[index:index + GAP].copy()


def render(root, recipe, by_id, frontend):
    split = recipe["split"]
    _split(split)
    insertion, parts, truth_part, last_real = recipe["insert"], [], None, None
    sequence = [(sample_id, False) for sample_id in recipe["seq"]]
    if not sequence:
        raise ValueError("Composite has no REAL clips")
    if insertion:
        if not 1 <= insertion["at"] < len(sequence):
            raise ValueError("Insertion requires preceding and following REAL clips")
        sequence.insert(insertion["at"], (insertion["fake"], True))
    for sample_id, is_fake in sequence:
        row = by_id[sample_id]
        if (row["split"] != split or row["recording"] != recipe["recording"]
                or bool(row["label_fake"]) != is_fake):
            raise ValueError("STOP: composite crosses split, recording, or label")
        samples = read(root, row)
        if is_fake:
            if insertion["crop"]:
                count = int(insertion["crop"] * SR)
                if count < XF:
                    raise ValueError("Fake crop is shorter than the crossfade")
                offset = max(0, (len(samples) - count) // 2)
                samples = samples[offset:offset + count]
            samples = samples * 10 ** ((_active_db(last_real, frontend)
                                        - _active_db(samples, frontend)) / 20)
        if parts:
            parts.append(_room_tone(last_real))
        if is_fake:
            truth_part = len(parts)
        else:
            last_real = samples
        if len(samples) < XF:
            raise ValueError("Clip is shorter than the common crossfade")
        parts.append(samples)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, XF))
    audio, starts = parts[0].copy(), [0]
    for part in parts[1:]:
        start = len(audio) - XF
        audio[start:] = audio[start:] * (1 - ramp) + part[:XF] * ramp
        audio = np.concatenate([audio, part[XF:]])
        starts.append(start)
    if len(audio) > MAX_S * SR or not np.isfinite(audio).all():
        raise ValueError("Composite exceeds the duration or finite-sample contract")
    truth = None if truth_part is None else [starts[truth_part] / SR,
             (starts[truth_part] + len(parts[truth_part])) / SR]
    output = io.BytesIO()
    wavfile.write(output, SR, np.clip(np.rint(audio * 32768), -32768, 32767).astype(np.int16))
    return output.getvalue(), truth


def mp3(wav_bytes, ffmpeg="ffmpeg"):
    return subprocess.run([str(ffmpeg), "-nostdin", "-loglevel", "error", "-f", "wav",
                           "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-c:a",
                           "libmp3lame", "-b:a", "64k", "-f", "mp3", "pipe:1"],
                          input=wav_bytes, capture_output=True, check=True, timeout=60).stdout
