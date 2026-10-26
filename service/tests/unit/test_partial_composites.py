"""Synthetic fixtures verify joins, central crops, and dataset scope guards."""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from scripts import partial_composites as pc


def _frontend():
    return SimpleNamespace(edge_trim=lambda samples: (0, len(samples), None),
                           active_level_dbfs=lambda samples:
                           20 * np.log10(np.sqrt(np.mean(samples * samples))))


def _write(root, sample_id, samples, *, fake=False, split="dev", engine="human"):
    output = io.BytesIO()
    wavfile.write(output, pc.SR, np.rint(samples * 32768).astype(np.int16))
    body = output.getvalue()
    relative = f"audio/{split}/{sample_id}.wav"
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    return dict(sample_id=sample_id, recording="fixture-recording", split=split,
                allocation_role=pc.ROLES[split], audio_path=relative, bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest(), label_fake=int(fake),
                speaker_id="fixture-voice", voice="fixture-voice", engine=engine,
                derivation={"start_frame": int(sample_id.rsplit("-", 1)[-1])})


@pytest.fixture
def clips(tmp_path):
    seconds = np.arange(4 * pc.SR) / pc.SR
    real = 0.3 * np.sin(2 * np.pi * 173 * seconds)
    fake = 0.075 * np.sin(2 * np.pi * 211 * seconds)
    rows = [_write(tmp_path, f"fixture-real-{index}", real) for index in range(3)]
    rows.append(_write(tmp_path, "fixture-fake-3", fake, fake=True, engine="xtts"))
    rows.append(_write(tmp_path, "fixture-generic-4", fake, fake=True, engine="melo"))
    manifest = tmp_path / "manifests" / "dev_required.jsonl"
    manifest.parent.mkdir()
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return tmp_path, rows


def test_recipe_determinism_and_common_controls(clips):
    root, rows = clips
    first = pc.recipes(root, "dev", lengths=(10,), per_len=2)
    assert first == pc.recipes(root, "dev", lengths=(10,), per_len=2)
    assert len(first) == 12
    for base in {recipe["base"] for recipe in first}:
        family = [recipe for recipe in first if recipe["base"] == base]
        control = next(recipe for recipe in family if recipe["kind"] == "real")
        assert all(recipe["seq"] == control["seq"] for recipe in family)
        assert {recipe["insert"]["type"] for recipe in family if recipe["insert"]} == {
            "same_full", "same_2s", "generic_full"}
    by_id = {row["sample_id"]: row for row in rows}
    assert pc.render(root, first[0], by_id, _frontend()) == pc.render(root, first[0], by_id, _frontend())


def test_crossfades_pause_and_truth(clips):
    root, rows = clips
    by_id = {row["sample_id"]: row for row in rows}
    control = pc.recipes(root, "dev", lengths=(10,), per_len=1)[0]
    body, truth = pc.render(root, control, by_id, _frontend())
    rate, actual = wavfile.read(io.BytesIO(body))
    parts = []
    previous = None
    for sample_id in control["seq"]:
        samples = pc.read(root, by_id[sample_id])
        if previous is not None:
            parts.append(pc._room_tone(previous))
        parts.append(samples)
        previous = samples
    expected = parts[0].copy()
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, pc.XF))
    for part in parts[1:]:
        expected[-pc.XF:] = expected[-pc.XF:] * (1 - ramp) + part[:pc.XF] * ramp
        expected = np.concatenate((expected, part[pc.XF:]))
    assert rate == pc.SR and truth is None
    assert len(actual) == sum(len(part) for part in parts) - pc.XF * (len(parts) - 1)
    np.testing.assert_array_equal(actual, np.rint(expected * 32768).astype(np.int16))


def test_center_crop_loudness_and_fake_clock(clips):
    root, rows = clips
    by_id = {row["sample_id"]: row for row in rows}
    recipe = next(recipe for recipe in pc.recipes(root, "dev", lengths=(10,), per_len=1)
                  if recipe["insert"] and recipe["insert"]["type"] == "same_2s")
    body, truth = pc.render(root, recipe, by_id, _frontend())
    _, pcm = wavfile.read(io.BytesIO(body))
    real = pc.read(root, by_id[recipe["seq"][0]])
    fake = pc.read(root, by_id[recipe["insert"]["fake"]])
    cropped = fake[pc.SR:3 * pc.SR]
    expected = cropped * 10 ** ((pc._active_db(real, _frontend())
                                - pc._active_db(cropped, _frontend())) / 20)
    start, end = [round(time * pc.SR) for time in truth]
    assert end - start == 2 * pc.SR
    assert start == len(real) + pc.GAP - 2 * pc.XF
    np.testing.assert_array_equal(pcm[start + pc.XF:end - pc.XF],
                                  np.rint(expected[pc.XF:-pc.XF] * 32768).astype(np.int16))


def test_hash_and_byte_count_are_verified(clips):
    root, rows = clips
    for changed in (dict(rows[0], sha256="0" * 64), dict(rows[0], bytes=rows[0]["bytes"] + 1)):
        with pytest.raises(ValueError, match="hash or byte count"):
            pc.read(root, changed)


@pytest.mark.parametrize("split", ["calib", "final", "../dev", "DEV"])
def test_protected_splits_rejected_before_read(tmp_path, split):
    with pytest.raises(ValueError, match="TRAIN/DEV"):
        pc.load_rows(tmp_path, split)


@pytest.mark.parametrize("relative", ["../outside.wav", "audio/train/fixture.wav",
                                     "audio/dev/../calib/fixture.wav", "audio/dev/final/fixture.wav",
                                     "C:/fixture.wav", "audio\\dev\\fixture.wav"])
def test_audio_paths_fail_closed(clips, relative):
    root, rows = clips
    with pytest.raises(ValueError, match="unauthorized audio path"):
        pc.read(root, dict(rows[0], audio_path=relative))


def test_allocation_and_cross_recording_are_rejected(clips):
    root, rows = clips
    with pytest.raises(ValueError, match="allocation role"):
        pc.read(root, dict(rows[0], allocation_role="CALIB"))
    recipe = pc.recipes(root, "dev", lengths=(10,), per_len=1)[0]
    by_id = {row["sample_id"]: row for row in rows}
    by_id[recipe["seq"][0]] = dict(by_id[recipe["seq"][0]], recording="different-fixture")
    with pytest.raises(ValueError, match="crosses split, recording, or label"):
        pc.render(root, recipe, by_id, _frontend())


def test_manifest_role_rejected(clips):
    root, rows = clips
    rows[0]["allocation_role"] = "CALIB"
    (root / "manifests" / "dev_required.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected split or allocation role"):
        pc.load_rows(root, "dev")


def test_dataset_root_scope_rejected_before_read(tmp_path):
    for directory in ("CALIB", "FINAL", "final_v2"):
        with pytest.raises(ValueError, match="protected dataset root"):
            pc.load_rows(tmp_path / directory, "dev")
