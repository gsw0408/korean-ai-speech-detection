"""HTTP upload-to-result CPU latency, startup, and process-tree peak RSS."""
import argparse
import io
import json
import os
import platform
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import psutil
from scipy.io import wavfile
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import create_app
from app.settings import Settings
from pipeline.frozen import frontend
from pipeline.preprocess import _active_seconds

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    rows = json.loads((ROOT / "work/dev_manifest.json").read_bytes())
    for row in rows:
        if row["label_fake"]:
            continue
        samples, _ = frontend.canonical((ROOT / "work/dev" / row["audio_path"]).read_bytes())
        start, end, _ = frontend.edge_trim(samples)
        clip = samples[start:start + 64000]
        if len(clip) == 64000 and frontend.edge_trim(clip)[:2] == (0, 64000) and _active_seconds(clip) == 4:
            break
    else:
        raise RuntimeError("No valid four-second REAL DEV crop for benchmark")
    bodies = {}
    for seconds in (4, 60, 120):
        output = io.BytesIO()
        wavfile.write(output, 16000, (np.tile(clip, seconds // 4) * 32768).astype(np.int16))
        bodies[seconds] = output.getvalue()
    return bodies, {"source_sample_id": row["sample_id"], "source_sha256": row["sha256"], "crop_start": start,
                    "method": "Repeat an unchanged four-second REAL DEV PCM crop; no external or protected audio."}


def request(client, body):
    result = client.post("/api/analyze", files={"file": ("benchmark.wav", body, "audio/wav")}, data={"analysis_consent": "true"}).json()
    if result["outcome_code"] == "cannot":
        raise RuntimeError("Benchmark request failed: " + str(result["reason"]))


def stats(values):
    return {"samples": len(values), "p50_seconds": float(np.percentile(values, 50)), "p95_seconds": float(np.percentile(values, 95)), "raw_seconds": values}


def main(repetitions):
    bodies, source = fixtures()
    destination = ROOT / "work/BENCHMARK.json"
    report = {"status": "running", "runtime": {"platform": platform.platform(), "python": platform.python_version(), "logical_cpus": os.cpu_count()},
              "source": source, "thread_trials": {}, "measurements": {}, "peak_rss_bytes": 0,
              "scope": "HTTP TestClient upload and response, multiprocessing worker IPC, decode, validation, full inference and decision. Cold includes app/model startup; warm reuses worker. Peak is parent plus child RSS, conservatively summing shared pages."}
    for file, key in (("/sys/fs/cgroup/cpu.max", "cgroup_cpu_max"), ("/sys/fs/cgroup/memory.max", "cgroup_memory_max")):
        if Path(file).exists():
            report["runtime"][key] = Path(file).read_text().strip()
    done = threading.Event()

    def memory_monitor():
        process = psutil.Process()
        while not done.wait(.02):
            try:
                total = process.memory_info().rss + sum(child.memory_info().rss for child in process.children(recursive=True) if child.is_running())
                report["peak_rss_bytes"] = max(report["peak_rss_bytes"], total)
            except psutil.Error:
                pass
    monitor = threading.Thread(target=memory_monitor, daemon=True)
    monitor.start()

    def save():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)

    try:
        with tempfile.TemporaryDirectory(dir=ROOT / "work", prefix="benchmark-") as directory:
            base = replace(Settings.from_env(), root=ROOT, upload_dir=Path(directory) / "uploads", donation_dir=Path(directory) / "donations")
            for threads in (1, 2, 4):
                with TestClient(create_app(replace(base, onnx_threads=threads))) as client:
                    request(client, bodies[4])
                    values = []
                    for _ in range(3):
                        started = time.perf_counter()
                        request(client, bodies[120])
                        values.append(time.perf_counter() - started)
                report["thread_trials"][str(threads)] = stats(values)
                save()
                print("Thread trial", threads, report["thread_trials"][str(threads)]["p95_seconds"], flush=True)
            chosen = min((1, 2, 4), key=lambda value: report["thread_trials"][str(value)]["p95_seconds"])
            report["selected_onnx_threads"], report["selected_concurrency"] = chosen, 1
            report["concurrency_reason"] = "One worker bounds model memory and avoids oversubscribing the four-CPU budget; semaphore contention is tested separately."
            for seconds, body in bodies.items():
                cold, warm = [], []
                report["measurements"][str(seconds)] = {"status": "running", "cold_raw_seconds": cold, "warm_raw_seconds": warm}
                save()
                for _ in range(repetitions):
                    started = time.perf_counter()
                    with TestClient(create_app(replace(base, onnx_threads=chosen))) as client:
                        request(client, body)
                        cold.append(time.perf_counter() - started)
                    save()
                with TestClient(create_app(replace(base, onnx_threads=chosen))) as client:
                    request(client, body)
                    for _ in range(repetitions):
                        started = time.perf_counter()
                        request(client, body)
                        warm.append(time.perf_counter() - started)
                        save()
                report["measurements"][str(seconds)] = {"cold": stats(cold), "warm": stats(warm)}
                save()
                print("Measured", seconds, report["measurements"][str(seconds)], flush=True)
        report["status"] = "complete"
        report["latency_target_met"] = report["measurements"]["120"]["warm"]["p95_seconds"] <= 30
        report["rss_target_met"] = report["peak_rss_bytes"] <= 2_500_000_000
    except Exception as exc:
        report["status"], report["error"] = "failed", str(exc)
        raise
    finally:
        done.set()
        monitor.join()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    main(parser.parse_args().repetitions)
