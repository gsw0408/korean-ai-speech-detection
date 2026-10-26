"""Run the pinned BAM repository's complete PartialSpoof evaluation on Kaggle.

No network access or dependency installation occurs here. ``--data`` is the
official database root containing eval/con_wav and segment_labels. Outputs are
compact resumable evidence under --out; model binaries and source stay in /tmp.
"""

import argparse
import contextlib
import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import sys
import time
import traceback
import zipfile


COMMIT = "55f3fb9e3b4dd6281597b86d7712fb23454179f6"
# LF-normalized source hashes permit a Windows checkout or a Linux git archive.
SOURCE_HASHES = {
    "requirements.txt": "154e3213c43865f19c5b8e250040858541c3670ed22cd36bc26aa3acddec631b",
    "train.py": "18ba70b78c0efe7890e8e5b4f846e468aa6f81c21be9a096f57e98bf977980b1",
    "config/bam_wavlm.yaml": "246acdd36a7d17f765b0589802cbaa32e73b9e1a00c7fee368b7898569a2c9f3",
    "config/bam_xlsr.yaml": "34b1d3a1e67b4aa4fd2f0bc98ca57dd0b6ca797e1dc0fa4fd7451940eba39aa8",
    "dataset/base_dataset.py": "f1e7bf349e2e799070badc878eaebdd0f4351d3ed53e53a9735621ba9a497d34",
    "dataset/partialspoof.py": "7a2451ba0534bf0ba669541fa0f91d5e1b1e89a8a70e0c3064f42e924883c042",
    "dataset/ps_preprocess.py": "09130b187c0c713b1019192943ca4d630888750cf5d46fed87e01407c544886e",
    "dataset/resample.py": "3054f0192985691bc1337a12d61ba0ecf5f20b933d796d34f00feb95c1326c70",
    "models/bam.py": "278fd8de63340da73fcd0c434e6982bc16250dcf25c6c6c1953a7cdfe5167b14",
    "models/loss.py": "f004aefe2d461912790b855e75e888d042385e742f065fbfe3faebf885bd16a5",
    "models/modules/attention.py": "f0b1cd70723a47b004a43b1816ea8a55a5011abbe1f0de323924d175e4144830",
    "models/modules/gap.py": "283da679b69a4b59773db1e074b5cef3aa7dd7e7b17adb91c4c0e7d5c0d2a410",
    "models/modules/resnet.py": "46170f7268d466b8a7613dd3f6a65f68ec4479b65f02508b24356570a779252d",
    "utils/__init__.py": "a46ee79a633570c27317ca6eb00fa7e9a062af99726e0523b7f344c7314939b1",
    "utils/analyze.py": "25ee4dc5b6f56e5029ca8c1bd95d1ca7675880d488f1dc3f5693b7d55cf4ca01",
    "utils/collect.py": "e8eb802c19723f48a3e0ff12584eddf5be1f513477b05c457cb353f2a71942d8",
    "utils/display.py": "2b8a7c3d381449bd78b15485e66d6fbf7b7f32f7c807b10b26fe0e859dd0e01c",
    "utils/metric.py": "d2f693b9e42e83134c4c5efd77f215a8ce024b115cb992c57f47316ed38eeacf",
    "utils/tool.py": "d65c713ff0df461268c681ad4403dedd0e72062cd3c7fe4bc7dc6f9442e867af",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class ParityFailure(RuntimeError):
    pass


def pack_records(records):
    import numpy as np
    if not records:
        raise ValueError("Cannot persist an empty inference partition")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate inference IDs")
    offsets = [0]
    for record in records:
        count = len(record["labels"])
        if count < 1 or any(len(record[key]) != count for key in ("logits", "boundary", "boundary_labels")):
            raise ValueError("Native inference returned truncated or misaligned frames")
        offsets.append(offsets[-1] + count)
    arrays = {key: np.concatenate([np.asarray(record[key], dtype=np.float32) for record in records])
              for key in ("logits", "labels", "boundary", "boundary_labels")}
    arrays.update(ids=np.asarray(ids), offsets=np.asarray(offsets, dtype=np.int64),
                  wall_seconds=np.asarray([r["wall_seconds"] for r in records], dtype=np.float64),
                  audio_seconds=np.asarray([r["audio_seconds"] for r in records], dtype=np.float64))
    validate_arrays(arrays)
    return arrays


def validate_arrays(arrays):
    import numpy as np
    ids, offsets = arrays["ids"], arrays["offsets"]
    count = len(ids)
    if count == 0 or len(set(ids.tolist())) != count or offsets.shape != (count + 1,):
        raise ValueError("Invalid compact output IDs/offsets")
    if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("Invalid compact frame offsets")
    frames = int(offsets[-1])
    if arrays["logits"].shape != (frames, 2):
        raise ValueError("Native BAM logits must have two classes")
    for key in ("labels", "boundary", "boundary_labels"):
        if arrays[key].shape != (frames,):
            raise ValueError("Compact output frame lengths differ")
    for key in ("wall_seconds", "audio_seconds"):
        if arrays[key].shape != (count,) or np.any(arrays[key] < 0):
            raise ValueError("Invalid per-utterance timing/duration")
    if any(not np.isfinite(value).all() for key, value in arrays.items() if key != "ids"):
        raise ValueError("Non-finite native inference output")


def pool_outputs(paths):
    import numpy as np
    blocks = []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            block = {key: saved[key] for key in saved.files}
        validate_arrays(block)
        blocks.append(block)
    if not blocks:
        raise ValueError("No complete inference partitions")
    arrays = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0] if key != "offsets"}
    arrays["offsets"] = np.concatenate(([0], np.cumsum(np.concatenate([np.diff(block["offsets"]) for block in blocks]))))
    validate_arrays(arrays)
    return arrays


def native_metrics(arrays, torch, compute_eer, fscore):
    """The same tensors, score direction and native functions as upstream epoch end."""
    result = {}
    for prefix, scores, labels, hard in (
        ("", arrays["logits"][:, 1], arrays["labels"], arrays["logits"].argmax(axis=-1)),
        ("boundary_", arrays["boundary"], arrays["boundary_labels"], (arrays["boundary"] > 0.5).astype("int64")),
    ):
        eer, _ = compute_eer(torch.from_numpy(scores), torch.from_numpy(labels))
        accuracy, precision, recall, f1 = fscore(torch.from_numpy(hard), torch.from_numpy(labels))
        for key, value in zip(("eer", "accuracy", "precision", "recall", "f1"), (eer, accuracy, precision, recall, f1)):
            result[prefix + key] = float(value)
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Native pooled metric is non-finite")
    return result


def compare_parity(unsharded, sharded, first_metrics, second_metrics, tolerance=1e-6):
    import numpy as np
    differences = {}
    passed = set(first_metrics) == set(second_metrics)
    for key in ("ids", "offsets", "labels", "boundary_labels"):
        equal = np.array_equal(unsharded[key], sharded[key])
        differences[key + "_equal"] = bool(equal)
        passed = passed and equal
    for key in ("logits", "boundary"):
        same_shape = unsharded[key].shape == sharded[key].shape
        delta = float(np.max(np.abs(unsharded[key] - sharded[key]))) if same_shape else None
        differences[key + "_max_abs_delta"] = delta
        passed = passed and same_shape and delta <= tolerance
    metric_delta = max((abs(first_metrics[key] - second_metrics[key]) for key in first_metrics), default=0.0) if set(first_metrics) == set(second_metrics) else None
    passed = passed and metric_delta is not None and metric_delta <= tolerance
    return {"passed": bool(passed), "tolerance": tolerance, "files": len(unsharded["ids"]),
            "raw_comparison": differences, "native_metric_max_abs_delta": metric_delta,
            "unsharded_metrics": first_metrics, "two_shard_pooled_metrics": second_metrics}


def record_parity(root, parity):
    from p1_shards import atomic_json
    atomic_json(root / "PARITY.json", parity)
    if not parity["passed"]:
        atomic_json(root / "HARD_STOP_PARITY.json", dict(parity, hard_stop="P1_SHARD_PARITY_FAILURE"))
        raise ParityFailure("BAM shard parity failed at 1e-6 tolerance")


def run_partition(store, index, ids, infer):
    """Commit only complete raw partitions; integrity-verified checkpoints skip inference."""
    from p1_shards import atomic_json
    import numpy as np
    existing = store.completed(index, ids)
    npz = store.root / ("shard_%03d.npz" % index)
    if existing is not None:
        return npz, existing
    arrays = infer(ids)
    if arrays["ids"].tolist() != list(ids):
        raise ValueError("Inference order differs from deterministic shard IDs")
    temporary = npz.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, npz)
    metadata = store.root / ("shard_%03d.json" % index)
    extra = {"files": len(ids), "frames": int(arrays["offsets"][-1]),
             "wall_seconds": float(arrays["wall_seconds"].sum()), "audio_seconds": float(arrays["audio_seconds"].sum()),
             "timing_definition": "Native evaluation_run_model wall time including synchronized collector conversion; data-loader waits excluded"}
    atomic_json(metadata, dict(extra, ids=list(ids)))
    return npz, store.commit(index, ids, [npz, metadata], extra=extra)


def output_manifest(root, identity):
    from p1_shards import atomic_json
    entries = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != root / "MANIFEST.json" and path.suffix not in (".part", ".tmp", ".partial"):
            entries[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    atomic_json(root / "MANIFEST.json", {"schema_version": 1, "identity": identity, "files": entries})


def select_checkpoint(source, scratch):
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            members = [entry for entry in archive.infolist() if entry.filename.endswith(".ckpt")]
            preferred = [entry for entry in members if Path(entry.filename).name == "model.ckpt"]
            members = preferred or members
            if len(members) != 1:
                raise ValueError("Checkpoint ZIP must contain one unambiguous model.ckpt")
            target = scratch / "model.ckpt"
            with archive.open(members[0]) as incoming, target.open("wb") as destination:
                shutil.copyfileobj(incoming, destination, length=8 * 1024 * 1024)
            return target
    if source.is_file() and source.suffix == ".ckpt":
        return source
    if source.is_dir():
        candidates = sorted(source.rglob("*.ckpt"))
        preferred = [path for path in candidates if path.name == "model.ckpt"]
        candidates = preferred or candidates
        if len(candidates) == 1:
            return candidates[0]
    raise ValueError("Supply an official checkpoint ZIP, .ckpt, or an unambiguous extracted directory")


def evaluate(options, result, started, scratch, stop):
    from p1_shards import ShardStore, atomic_json, shard_ids
    source_hashes = {}
    for relative, expected in SOURCE_HASHES.items():
        raw = (options.repo / relative).read_bytes()
        actual = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError("Pinned BAM source mismatch: " + relative)
        source_hashes[relative] = actual
    result["source_files_lf_sha256"] = source_hashes
    runtime_repo = scratch / "repo"
    shutil.copytree(options.repo, runtime_repo, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    os.chdir(runtime_repo)
    sys.path.insert(0, str(runtime_repo))
    import lightning as L
    import numpy as np
    import soundfile as sf
    import torch
    import train as upstream
    from dataset.partialspoof import PartialSpoofDataModule
    from dataset.ps_preprocess import get_boundary_labels
    if not torch.cuda.is_available():
        raise RuntimeError("P1 requires an available Kaggle GPU")
    result["environment"] = {"python": sys.version, "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    for package in ("torch", "torchaudio", "lightning", "pytorch-lightning", "s3prl", "numpy", "scipy", "scikit-learn", "librosa", "soundfile"):
        result["environment"][package] = importlib.metadata.version(package)
    checkpoint = select_checkpoint(options.checkpoint, scratch)
    result["artifacts"] = {"checkpoint_sha256": sha256(checkpoint), "frontend_sha256": sha256(options.frontend)}
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        saved = torch.load(checkpoint, map_location="cpu")
    hyperparameters = saved["hyper_parameters"]
    config = hyperparameters["config"]
    config_values = dict(config if isinstance(config, dict) else vars(config))
    if config_values.get("ssl_name") != "wavlm_local" or config_values.get("embed_dim") != 1024:
        raise ValueError("Official WavLM-Large configuration absent from checkpoint")
    saved_args = hyperparameters.get("args")
    saved_arg_values = dict(saved_args if isinstance(saved_args, dict) else vars(saved_args)) if saved_args is not None else {}
    if float(saved_arg_values.get("resolution", 0.16)) != 0.16:
        raise ValueError("Checkpoint resolution differs from 160 ms")
    result["checkpoint_saved_config"] = config_values.copy()
    config_values["ssl_ckpt"] = str(options.frontend)
    config = upstream.Attribution_Config(**config_values)
    del saved, hyperparameters
    gc.collect()
    audio_root = options.data / "eval" / "con_wav"
    label_file = options.data / "segment_labels" / "eval_seglab_0.16.npy"
    audio_files = sorted(audio_root.rglob("*.wav"))
    if not audio_files or not label_file.is_file():
        raise ValueError("--data lacks official eval/con_wav or 160 ms labels")
    labels = np.load(label_file, allow_pickle=True).item()
    paths = {path.stem: path for path in audio_files}
    if len(paths) != len(audio_files) or set(paths) != set(labels) or len(paths) != 71237:
        raise ValueError("Complete official eval requires exactly 71,237 unique matching audio/label IDs")
    ids = sorted(paths)
    durations, headers = {}, []
    for identifier in ids:
        info = sf.info(paths[identifier])
        if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16" or info.frames <= 0:
            raise ValueError("Expected official mono 16 kHz PCM16 WAV: " + identifier)
        durations[identifier] = info.frames / 16000
        headers.append((identifier, info.frames, info.samplerate, info.channels, info.subtype))
    digest_json = lambda value: hashlib.sha256(json.dumps(value, separators=(",", ":")).encode("utf-8")).hexdigest()
    expanded_manifest = Path(__file__).resolve().with_name("PARTIALSPOOF_EXPANDED_SHA256.json")
    identity = {"model": "BAM", "commit": COMMIT, "source_files": source_hashes,
                "checkpoint_sha256": result["artifacts"]["checkpoint_sha256"], "frontend_sha256": result["artifacts"]["frontend_sha256"],
                "labels_sha256": sha256(label_file), "ids_sha256": digest_json(ids), "headers_sha256": digest_json(headers),
                "expanded_manifest_sha256": sha256(expanded_manifest) if expanded_manifest.is_file() else None,
                "frame_hop_ms": 160, "shard_count": options.shard_count, "output_format": 1}
    result["identity"] = identity
    label_root = scratch / "labels"
    label_root.mkdir()
    get_boundary_labels(str(options.data), "eval", str(label_root), resolution=0.16)
    args = argparse.Namespace(dataset="dataset.partialspoof.PartialSpoofDataModule", model="models.bam.BAM",
        train_root="unused", dev_root="unused", eval_root=str(audio_root), label_root=str(label_root), max_epochs=50,
        batch_size=8, num_workers=options.num_workers, base_lr=1e-5, weight_decay=1e-4, samplerate=16000,
        resolution=0.16, input_maxlength=None, input_minlength=None, label_maxlength=25, pad_mode="label", gpu=[0],
        test_only=True, exp_name="bam_wavlm", validate_interval=3, checkpoint=str(checkpoint), continue_train=False)
    L.seed_everything(42, workers=True)
    model = upstream.LightingModelWrapper.load_from_checkpoint(str(checkpoint), map_location="cpu", args=args, config=config, strict=True)
    model.eval().to(torch.device("cuda", 0))
    dataset = PartialSpoofDataModule(args).get_dataset("eval")
    indices = {identifier: index for index, identifier in enumerate(dataset.utt_ids)}
    processed_audio = [0.0]
    remaining_audio = [sum(durations.values())]
    stage = ["parity"]
    last_advisory = [time.monotonic()]
    def infer(selected):
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [indices[item] for item in selected]),
                                            batch_size=1, shuffle=False, num_workers=options.num_workers)
        records = []
        with torch.no_grad():
            for batch in loader:
                identifier = batch[0][0]
                transferred = tuple(value.to(torch.device("cuda", 0)) if isinstance(value, torch.Tensor) else value for value in batch)
                before = time.monotonic()
                model.evaluation_run_model(transferred, type="test")
                records.append({"id": identifier, "logits": model.test_step_outputs[0], "labels": model.test_pred_labels[0],
                                "boundary": model.b_test_step_outputs[0], "boundary_labels": model.b_test_pred_labels[0],
                                "wall_seconds": time.monotonic() - before, "audio_seconds": durations[identifier]})
                for collector in (model.test_step_outputs, model.test_pred_labels, model.b_test_step_outputs,
                                  model.b_test_pred_labels, model.utt_id_list):
                    collector.clear()
                processed_audio[0] += durations[identifier]
                remaining_audio[0] = max(0.0, remaining_audio[0] - durations[identifier])
                now = time.monotonic()
                if now - last_advisory[0] >= 600:
                    rate = (now - started) / processed_audio[0]
                    print(json.dumps({"phase": "eta_advisory", "elapsed_seconds": now - started,
                                      "stage": stage[0], "wall_seconds_per_audio_second": rate,
                                      "remaining_audio_seconds": remaining_audio[0], "eta_seconds": rate * remaining_audio[0],
                                      "projections_are_advisory_only": True}), flush=True)
                    last_advisory[0] = now
        return pack_records(records)
    metric = lambda arrays: native_metrics(arrays, torch, upstream.compute_eer, upstream.computer_precision_recall_fscore)
    parity_ids = ids[:200]
    parity_paths = []
    for name, count in (("unsharded", 1), ("two_shards", 2)):
        parity_identity = dict(identity, scope="parity_first_200", partitions=count, parity_ids_sha256=digest_json(parity_ids))
        store = ShardStore(options.out / "parity" / name, parity_identity,
                           [root / "parity" / name for root in options.resume_root])
        parity_paths.append([run_partition(store, index, selected, infer)[0]
                             for index, selected in enumerate(shard_ids(parity_ids, count))])
    unsharded, sharded = (pool_outputs(paths) for paths in parity_paths)
    parity = compare_parity(unsharded, sharded, metric(unsharded), metric(sharded))
    result["parity"] = parity
    record_parity(options.out, parity)
    if options.parity_only:
        result.update(status="parity_passed", execution_status="parity_passed")
        return
    store = ShardStore(options.out / "shards", identity, [root / "shards" for root in options.resume_root])
    stage[0] = "full"
    remaining_audio[0] = sum(durations.values())
    partitions = shard_ids(ids, options.shard_count)
    full_paths, completed_ids = [], []
    for index, selected in enumerate(partitions):
        existing = store.completed(index, selected)
        if existing is None and (stop["requested"] or time.monotonic() - started >= options.max_seconds):
            break
        path, metadata = run_partition(store, index, selected, infer)
        if existing is not None:
            remaining_audio[0] = max(0.0, remaining_audio[0] - sum(durations[item] for item in selected))
        full_paths.append(path)
        completed_ids.extend(selected)
        result["coverage"] = {"expected_files": len(ids), "scored_files": len(completed_ids), "completed_shards": len(full_paths),
                              "total_shards": len(partitions), "complete": len(completed_ids) == len(ids)}
        result.update(status="partial", execution_status="partial")
        atomic_json(options.out / "result.json", result)
        output_manifest(options.out, identity)
        print(json.dumps({"phase": "shard_completed", "model": "BAM", "shard": index, "files": len(selected),
                          "completed_files": len(completed_ids), "total_files": len(ids)}), flush=True)
    if completed_ids != ids:
        result.update(status="partial", execution_status="partial", metrics=None,
                      resume_command="Repeat the same CLI with --resume-root " + str(options.out))
        result.setdefault("coverage", {"expected_files": len(ids), "scored_files": len(completed_ids),
                                        "completed_shards": len(full_paths), "total_shards": len(partitions), "complete": False})
        return
    pooled = pool_outputs(full_paths)
    if pooled["ids"].tolist() != ids or len(pooled["labels"]) != sum(len(labels[item]) for item in ids):
        raise ValueError("Pooled complete-eval coverage is not exact")
    measured = metric(pooled)
    result["raw_metrics"] = measured
    result["metrics"] = {"eer_percent": measured["eer"] * 100, "f1_percent": measured["f1"] * 100,
                         "f1_positive_class": "bona fide (upstream label 1)", "map": "not provided by native BAM evaluator"}
    result["full_metric_computations"] = 1
    result["coverage"].update(expected_frames=sum(len(labels[item]) for item in ids), scored_frames=len(pooled["labels"]), skipped_files=0)
    result["reproduced"] = abs(measured["eer"] * 100 - 3.58) <= 1.0
    result.update(status="passed" if result["reproduced"] else "failed", execution_status="completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "data", "checkpoint", "frontend", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=40500, help="Soft deadline checked only between complete shards")
    parser.add_argument("--shard-count", type=int, default=32)
    parser.add_argument("--resume-root", type=Path, action="append", default=[], help="Previous BAM output root; may be repeated")
    parser.add_argument("--parity-only", action="store_true")
    options = parser.parse_args()
    if not 16 <= options.shard_count <= 64 or not math.isfinite(options.max_seconds) or options.max_seconds <= 0:
        parser.error("Require 16-64 shards and a finite positive soft deadline")
    for name in ("repo", "data", "checkpoint", "frontend", "out"):
        setattr(options, name, getattr(options, name).resolve())
    options.resume_root = [root.resolve() for root in options.resume_root]
    options.out.mkdir(parents=True, exist_ok=True)
    stop = {"requested": False}
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda signum, frame: stop.update(requested=True))
    started = time.monotonic()
    result = {"schema_version": 2, "model": "BAM", "label": "PartialSpoof reproduction", "pinned_commit": COMMIT,
              "frame_hop_ms": 160, "published_eer_percent": 3.58, "published_f1_percent": 96.09,
              "tolerance_eer_points": 1.0, "status": "failed", "execution_status": "failed", "reproduced": False, "metrics": None,
              "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    scratch_parent = Path("/tmp/bc_p1")
    scratch_parent.mkdir(parents=True, exist_ok=True)
    original_directory = Path.cwd()
    raw_log = options.out / "raw.log"
    with raw_log.open("a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                with tempfile.TemporaryDirectory(prefix="bam_", dir=str(scratch_parent)) as temporary:
                    evaluate(options, result, started, Path(temporary), stop)
            except Exception as error:
                result.update(status="failed", execution_status="failed", metrics=None)
                result["error"] = {"type": type(error).__name__, "message": str(error)}
                traceback.print_exc()
            finally:
                os.chdir(original_directory)
    result["elapsed_seconds"] = time.monotonic() - started
    result["raw_log"] = {"path": str(raw_log), "sha256": sha256(raw_log)}
    write_json(options.out / "result.json", result)
    output_manifest(options.out, result.get("identity", {"model": "BAM", "commit": COMMIT}))
    print(json.dumps({"model": "BAM", "status": result["status"], "result": str(options.out / "result.json")}), flush=True)
    return 0 if result["execution_status"] in ("completed", "partial", "parity_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
