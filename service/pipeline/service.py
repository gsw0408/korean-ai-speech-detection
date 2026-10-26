"""Persistent, terminable inference workers; service validity is mandatory."""
import multiprocessing as mp
import shutil
import uuid
from pathlib import Path
import psutil
from .bundle import load_bundle
from .decision import decide
from .display import DISPLAY_VERSION, build_display
from .partial import NR_SCHEMA, analyze_partial
from .preprocess import CannotAnalyze, prepare
from .path_guard import guard_path


def _worker(connection, root, active_bundle, threads, ffmpeg, uploads):
    try:
        bundle = load_bundle(active_bundle, root=Path(root), threads=threads)
        connection.send({"ready": True})
    except Exception:
        connection.send({"ready": False})
        return
    while True:
        try:
            body = connection.recv()
        except EOFError:
            return
        if body is None:
            return
        try:
            prepared = prepare(body, ffmpeg=ffmpeg, temp_dir=uploads, modules=bundle.preprocessing)
            frame_scores = None
            frame_method = (bundle.partial is not None and bundle.partial.get("schema") == NR_SCHEMA
                            and bundle.partial["candidate"].startswith("F"))
            if frame_method:
                scores, frame_scores = bundle.detector.score_windows_with_frames(prepared.windows)
            else:
                scores = bundle.detector.score_windows(prepared.windows)
            result = decide(prepared, scores, bundle.calibration["t_low"], bundle.calibration["t_warn"])
            if bundle.partial is not None:
                if bundle.partial.get("schema") == NR_SCHEMA:
                    result["partial"] = analyze_partial(result["windows"], bundle.partial, duration=prepared.duration,
                        frame_scores=frame_scores, starts=prepared.starts, trim_start=prepared.trim_start)
                else:
                    result["partial"] = analyze_partial(result["windows"], bundle.partial)
            try:
                result["display"] = build_display(result, bundle.calibration, bundle.partial)
            except Exception:
                pass  # Presentation only: the analysis still succeeds, without a display key.
            response = {"result": result}
        except CannotAnalyze as exc:
            response = {"reason": exc.reason}
        except Exception:
            response = {"reason": "analysis_failed"}
        finally:
            body = prepared = scores = frame_scores = result = None
            Path(uploads).rmdir() if Path(uploads).exists() else None
        connection.send(response)
        response = None


class AnalysisWorker:
    def __init__(self, root, active_bundle, threads, ffmpeg, uploads):
        self.upload_root = guard_path(uploads)
        self.uploads = self.upload_root / ("worker-" + uuid.uuid4().hex)
        self.arguments = (str(root), active_bundle, threads, str(ffmpeg), str(self.uploads))
        self.process = self.connection = None
        self.start()

    def start(self):
        parent, child = mp.get_context("spawn").Pipe()
        self.connection = parent
        self.process = mp.get_context("spawn").Process(target=_worker, args=(child, *self.arguments), daemon=True)
        self.process.start()
        child.close()
        if not parent.poll(45) or not parent.recv().get("ready"):
            self.close()
            raise RuntimeError("Inference worker startup failed")

    def analyze(self, body, timeout):
        try:
            self.connection.send(body)
            if not self.connection.poll(timeout):
                self.close()
                self.start()
                raise CannotAnalyze("analysis_timeout")
            response = self.connection.recv()
            if "reason" in response:
                raise CannotAnalyze(response["reason"])
            return response["result"]
        except (BrokenPipeError, EOFError, OSError):
            self.close()
            self.start()
            raise CannotAnalyze("analysis_failed")

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                try:
                    for child in psutil.Process(self.process.pid).children(recursive=True):
                        child.kill()
                except psutil.Error:
                    pass
                self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        if self.connection is not None:
            self.connection.close()
        # This path is created exclusively for this worker, never user supplied.
        if self.uploads.exists():
            target = self.uploads.resolve()
            if target.parent != self.upload_root:
                raise RuntimeError("Worker cleanup path escapes upload root")
            shutil.rmtree(target)
