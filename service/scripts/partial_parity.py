"""Rebuild DEV composites locally and check frozen CPU/GPU service parity."""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np


WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS))

from pipeline.bundle import load_bundle
from pipeline.decision import decide
from pipeline.partial import analyze_partial
from pipeline.preprocess import prepare
from scripts import partial_composites as pc


TOLERANCE = 1e-3


def _work_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(WS / "work"):
        raise ValueError("Parity outputs and run inputs must stay in webservice/work")
    return path


def _write_json(path, value):
    path = _work_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _rows(path):
    return [json.loads(line) for line in _work_path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _run_dir(supplied=None):
    if supplied is None:
        receipt = WS / "work" / "partial_mvp" / "PULLED_RUN.json"
        if receipt.is_file():
            supplied = json.loads(receipt.read_text(encoding="utf-8"))["directory"]
    if supplied:
        supplied = Path(supplied)
        if not supplied.is_absolute():
            supplied = WS / supplied
        directory = _work_path(supplied)
        if not all((directory / name).is_file() for name in
                   ("recipes.jsonl", "window_scores.jsonl", "OUTPUT_MANIFEST.json")):
            raise ValueError("Pulled run is incomplete")
        return directory
    candidates = [path.parent for path in (WS / "work" / "partial_mvp").rglob("recipes.jsonl")
                  if (path.parent / "window_scores.jsonl").is_file()
                  and (path.parent / "OUTPUT_MANIFEST.json").is_file()]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one verified pulled run; supply --run-dir")
    return _work_path(candidates[0])


def _verify_run(directory):
    manifest = json.loads((directory / "OUTPUT_MANIFEST.json").read_text(encoding="utf-8"))
    required = {"recipes.jsonl", "window_scores.jsonl", "partial_thresholds.json", "partial_eval.json"}
    if not required.issubset(manifest):
        raise ValueError("Run manifest omits required evidence")
    for relative, digest in manifest.items():
        supplied = Path(relative)
        if supplied.is_absolute() or len(supplied.parts) != 1 or "\\" in relative:
            raise ValueError("Run manifest has an unsafe member")
        member = _work_path(directory / supplied)
        if member.parent != directory:
            raise ValueError("Run manifest member escapes the run")
        if hashlib.sha256(member.read_bytes()).hexdigest() != digest:
            raise ValueError("Pulled run SHA-256 mismatch")
    return hashlib.sha256((directory / "OUTPUT_MANIFEST.json").read_bytes()).hexdigest()


def select_composites(records, count=40):
    """Use an even stride within sorted REAL and attack IDs, twenty of each."""
    if count < 2 or count % 2:
        raise ValueError("Parity count must be a positive even count")
    selected = []
    for kind in ("real", "attack"):
        group = sorted((record for record in records if record["kind"] == kind),
                       key=lambda record: record["id"])
        take = count // 2
        if len(group) < take:
            raise ValueError("Insufficient DEV clean REAL/attack parity cases")
        stride = max(1, len(group) // take)
        selected.extend(group[::stride][:take])
    return sorted(selected, key=lambda record: record["id"])


def _response(bundle, prepared, scores):
    result = decide(prepared, scores, bundle.calibration["t_low"], bundle.calibration["t_warn"])
    if bundle.partial is not None:
        result["partial"] = analyze_partial(result["windows"], bundle.partial)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--run-dir")
    arguments = parser.parse_args(argv)
    directory = _run_dir(arguments.run_dir)
    output = _work_path(WS / "work" / "partial_mvp")
    report = dict(status="RUNNING", tolerance=TOLERANCE, selected_count=40,
                  output_manifest_sha256=_verify_run(directory), cases=[],
                  selection="Even stride over sorted IDs, 20 REAL and 20 attack DEV clean composites",
                  response_check="Frozen v1 fields use the same CPU scores; partial analysis adds one key")
    destination = output / "LOCAL_PARITY.json"
    _write_json(destination, report)
    try:
        recipes = {}
        for recipe in _rows(directory / "recipes.jsonl"):
            pc._split(recipe["split"])
            if recipe["id"] in recipes:
                raise ValueError("Duplicate composite recipe")
            recipes[recipe["id"]] = recipe
        records = _rows(directory / "window_scores.jsonl")
        keys = set()
        for record in records:
            pc._split(record["split"])
            key = (record["id"], record["cond"])
            if key in keys:
                raise ValueError("Duplicate composite score record")
            keys.add(key)
        dev = [record for record in records if record["split"] == "dev"
               and record["cond"] == "clean" and record["status"] == "ok"]
        selected = select_composites(dev)
        by_id = {row["sample_id"]: row for row in pc.load_rows(arguments.dataset_root, "dev")}
        partial = load_bundle("bundles/nii-small-v1-partial", root=WS)
        baseline = load_bundle("bundles/nii-small-v1", root=WS, initialize_detector=False)
        if baseline.partial is not None or partial.partial is None:
            raise ValueError("Wrong baseline or experimental bundle")
        for key in ("detector", "frontend", "audio_windows", "calibration"):
            if baseline.hashes[key] != partial.hashes[key]:
                raise ValueError("Experimental bundle changed a v1 binding")
        if baseline.calibration != partial.calibration:
            raise ValueError("Experimental bundle changed v1 decisions")
        thresholds = json.loads((directory / "partial_thresholds.json").read_text(encoding="utf-8"))
        if partial.partial != thresholds:
            raise ValueError("Experimental bundle differs from the verified run")
        report["bundle_hashes"] = {"v1": baseline.hashes, "v1_partial": partial.hashes}
        ffmpeg = os.environ.get("FFMPEG") or imageio_ffmpeg.get_ffmpeg_exe()
        for record in selected:
            recipe = recipes[record["id"]]
            if any(record[key] != recipe[key] for key in ("split", "recording", "kind")):
                raise ValueError("Recipe and score identities disagree")
            body, truth = pc.render(arguments.dataset_root, recipe, by_id, partial.preprocessing[0])
            if truth != record["truth"]:
                raise ValueError("Composite insertion clock differs from Kaggle")
            prepared = prepare(body, ffmpeg=ffmpeg, temp_dir=output / "uploads",
                               modules=partial.preprocessing)
            cpu = partial.detector.score_windows(prepared.windows)
            gpu = record["scores"]
            if (len(cpu) != record["n"] or len(cpu) != len(gpu)
                    or not np.isfinite(gpu).all() or not np.isfinite(cpu).all()):
                raise ValueError("CPU/GPU window shape or finiteness mismatch")
            delta = float(np.max(np.abs(np.subtract(cpu, gpu))))
            if delta > TOLERANCE:
                raise ValueError("CPU/GPU score tolerance exceeded")
            local_starts = [(prepared.trim_start + start) / pc.SR for start in prepared.starts]
            if (not np.allclose(local_starts, record["starts"], rtol=0, atol=1e-10)
                    or not math.isclose(prepared.duration, record["duration"], rel_tol=0, abs_tol=1e-10)):
                raise ValueError("CPU/GPU preprocessing clock mismatch")
            before = _response(baseline, prepared, cpu)
            response = _response(partial, prepared, cpu)
            saved_before = copy.deepcopy(before)
            if set(response) != set(before) | {"partial"}:
                raise ValueError("Experimental response added or removed v1 fields")
            if {key: value for key, value in response.items() if key != "partial"} != saved_before:
                raise ValueError("Experimental response changed v1 fields")
            gpu_response = _response(partial, prepared, gpu)["partial"]
            screen = response["partial"]
            boundary = (gpu_response["threshold"] is not None
                        and abs(gpu_response["statistic"] - gpu_response["threshold"]) <= TOLERANCE)
            same_status = screen["status"] == gpu_response["status"]
            if not same_status and not boundary:
                raise ValueError("CPU/GPU partial status differs away from threshold")
            report["cases"].append(dict(id=record["id"], kind=record["kind"],
                                       n_windows=len(cpu), max_abs=delta,
                                       composite_sha256=hashlib.sha256(body).hexdigest(),
                                       v1_fields_identical=True, local_status=screen["status"],
                                       gpu_status=gpu_response["status"],
                                       boundary_exemption=not same_status and boundary))
            _write_json(destination, report)
            print(f"CPU parity {len(report['cases'])}/40", flush=True)
        e2e = _work_path(output / "e2e")
        e2e.mkdir(parents=True, exist_ok=True)
        report["e2e"] = {}
        for kind, filename in (("real", "real.wav"), ("same_full", "attack.wav")):
            chosen = next(record for record in dev if record["kind"] == "real") if kind == "real" else next(
                record for record in dev if record.get("type") == "same_full")
            body, _ = pc.render(arguments.dataset_root, recipes[chosen["id"]], by_id, partial.preprocessing[0])
            (e2e / filename).write_bytes(body)
            report["e2e"][filename] = dict(id=chosen["id"], sha256=hashlib.sha256(body).hexdigest())
        report.update(status="PASS", verified_count=len(report["cases"]),
                      max_abs=max(case["max_abs"] for case in report["cases"]),
                      boundary_exemptions=sum(case["boundary_exemption"] for case in report["cases"]))
        _write_json(destination, report)
        print(f"LOCAL_PARITY PASS: 40 DEV clean composites, max_abs={report['max_abs']:.9g}")
    except Exception as exc:
        report.update(status="FAIL", failure_type=type(exc).__name__, failure=str(exc))
        _write_json(destination, report)
        raise


if __name__ == "__main__":
    main()
