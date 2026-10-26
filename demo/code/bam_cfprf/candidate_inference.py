"""Label-free native candidate inference on one canonical, normalized clock.

Imports are light until a backend loads. Assets must be pinned by the external
driver. Native source is verified here and copied to the supplied scratch path;
Linux experiment callers keep that path under /tmp. No dataset loader is used.
"""

from argparse import Namespace
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np

from research_common import SR, aggregate_frames, max_rolling_mean


WINDOW = 64000
ORIENTATION = "spoof score = negative native class-1 (bona fide) raw logit"


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " requires integer sample coordinates")
    return int(value)


def _one(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError("Expected one-dimensional convolution attributes")
        value = value[0]
    return _integer(value, "convolution attribute")


def convolution_geometry(layers):
    """Derive valid feature count, receptive field and left origin from attrs.

    This describes the convolutional feature extractor only. Transformer,
    normalization, graph and proposal context may use the whole input chunk.
    """
    specs, hop, receptive, origin = [], 1, 1, 0
    for layer in layers:
        kernel, stride, padding, dilation = (_one(getattr(layer, key)) for key in
                                             ("kernel_size", "stride", "padding", "dilation"))
        if min(kernel, stride, dilation) <= 0 or padding < 0:
            raise ValueError("Invalid feature extractor convolution")
        specs.append(dict(kernel=kernel, stride=stride, padding=padding, dilation=dilation))
        origin -= padding * hop
        receptive += (kernel - 1) * dilation * hop
        hop *= stride
    if not specs:
        raise ValueError("No feature extractor convolutions were found")
    return {"layers": specs, "hop_samples": hop, "receptive_field_samples": receptive,
            "origin_sample": origin, "center_offset_samples": origin + (receptive - 1) / 2,
            "context": "full chunk; convolutional receptive field is not the final neural context"}


def feature_count(samples, geometry):
    count = _integer(samples, "input length")
    for layer in geometry["layers"]:
        count = (count + 2 * layer["padding"] - layer["dilation"] * (layer["kernel"] - 1) - 1) // layer["stride"] + 1
        if count <= 0:
            return 0
    return count


def inspect_ssl_geometry(module, torch):
    candidates = [(name, child) for name, child in module.named_modules()
                  if name.split(".")[-1] == "feature_extractor"]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one loaded SSL feature_extractor; found " + str([name for name, _ in candidates]))
    name, extractor = candidates[0]
    layers = [child for child in extractor.modules() if isinstance(child, torch.nn.Conv1d)]
    geometry = convolution_geometry(layers)
    if (geometry["hop_samples"], geometry["receptive_field_samples"], geometry["origin_sample"]) != (320, 400, 0):
        raise ValueError("Loaded SSL geometry differs from the verified native 20 ms clock")
    geometry["module"] = name
    return geometry


def native_supports(frames, hop_samples, valid_samples):
    frames, hop, valid = (_integer(value, name) for value, name in
                          ((frames, "frames"), (hop_samples, "hop"), (valid_samples, "valid_samples")))
    if frames < 0 or hop <= 0 or valid <= 0:
        raise ValueError("Invalid native decision bins")
    left = np.arange(frames, dtype=np.int64) * hop
    supports = np.column_stack((left, left + hop))
    # Partially real BAM bins retain only their real support. No new CF bins
    # are generated to cover the native valid-convolution tail.
    supports = np.minimum(supports, valid)
    return supports, supports[:, 1] > supports[:, 0]


def _array(value):
    return value.detach().cpu().numpy().copy()


def _torch_load(torch, path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _source_import(repo, module):
    # BAM and CFPRF both use top-level models/libs packages. Separate model
    # worker processes prevent Python's module cache from mixing repositories.
    for name in ("models", "libs", "utils", "train", "evaluate_CFPRF"):
        loaded = sys.modules.get(name)
        location = getattr(loaded, "__file__", None)
        if location:
            try:
                Path(location).resolve().relative_to(repo.resolve())
            except ValueError:
                raise RuntimeError("Use one isolated model worker process: conflicting import " + name)
    sys.path.insert(0, str(repo))
    return importlib.import_module(module)


class BAMBackend:
    name, hop_samples = "BAM", 2560
    cpu_capability = {"supported": True, "condition": "native PyTorch CPU inference; unbenchmarked"}

    def __init__(self, repo, checkpoint, frontend, scratch, device="cuda"):
        self.repo, self.checkpoint, self.frontend, self.scratch = map(Path, (repo, checkpoint, frontend, scratch))
        self.device_name, self.model, self.geometry, self.identity = device, None, None, None

    def load(self):
        if self.model is not None:
            return
        from p1_bam import COMMIT, SOURCE_HASHES, select_checkpoint
        runtime = self.scratch / "bam_native"
        runtime.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for relative, expected in SOURCE_HASHES.items():
            content = (self.repo / relative).read_bytes()
            actual = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
            if actual != expected:
                raise ValueError("Pinned BAM source mismatch: " + relative)
            target = runtime / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            hashes[relative] = actual
        import torch
        upstream = _source_import(runtime, "train")
        checkpoint = select_checkpoint(self.checkpoint, self.scratch)
        saved = _torch_load(torch, checkpoint)
        hyper = saved["hyper_parameters"]
        config = hyper["config"]
        values = dict(config if isinstance(config, dict) else vars(config))
        if values.get("ssl_name") != "wavlm_local" or values.get("embed_dim") != 1024:
            raise ValueError("Official WavLM-Large saved configuration is required")
        saved_args = hyper.get("args")
        args = dict(saved_args if isinstance(saved_args, dict) else vars(saved_args)) if saved_args is not None else {}
        if float(args.get("resolution", 0.16)) != 0.16:
            raise ValueError("Expected the native 160 ms BAM checkpoint")
        args.update(model="models.bam.BAM", resolution=0.16, samplerate=SR)
        values["ssl_ckpt"] = str(self.frontend.resolve())
        del saved, hyper
        wrapper = upstream.LightingModelWrapper.load_from_checkpoint(str(checkpoint), map_location="cpu",
            args=Namespace(**args), config=upstream.Attribution_Config(**values), strict=True)
        self.torch, self.device = torch, torch.device(self.device_name)
        model = wrapper.model.eval().to(self.device)
        if model.pool_frame_num != 8:
            raise ValueError("Native BAM must pool eight SSL frames per decision")
        geometry = inspect_ssl_geometry(model.ssl_layer, torch)
        self.model, self.geometry = model, geometry
        self.identity = {"commit": COMMIT, "source_files_lf_sha256": hashes,
                         "strict_checkpoint_load": True, "cpu_capability": self.cpu_capability}

    def score_chunk(self, waveform):
        self.load()
        torch, valid = self.torch, len(waveform)
        padded = -(-valid // self.hop_samples) * self.hop_samples
        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32).copy()).unsqueeze(0).to(self.device)
        tensor = torch.nn.functional.pad(tensor, (0, padded - valid))
        with torch.no_grad():
            logits, boundary = self.model(tensor)
        logits, boundary = _array(logits)[0], _array(boundary)[0]
        expected = feature_count(padded + 256, self.geometry)
        if expected % 8 or logits.shape != (expected // 8, 2) or boundary.shape != (len(logits),):
            raise ValueError("BAM outputs differ from its loaded native feature geometry")
        return {"logits": logits, "boundary": boundary, "proposals": {},
                "padding_samples": padded - valid, "native_internal_padding_samples": 256}


class CFPRFBackend:
    name, hop_samples = "CFPRF", 320
    cpu_capability = {"supported": False, "reason": "Native PositionalEncoding allocates an unregistered tensor on CUDA; CPU placement wrapper requires separate numerical validation"}

    def __init__(self, repo, fdn, prn, frontend, scratch, device="cuda"):
        self.repo, self.fdn_path, self.prn_path, self.frontend, self.scratch = map(Path, (repo, fdn, prn, frontend, scratch))
        self.device_name, self.models, self.geometry, self.identity = device, None, None, None

    def load(self):
        if self.models is not None:
            return
        if str(self.device_name).split(":")[0] != "cuda":
            raise NotImplementedError(self.cpu_capability["reason"])
        from p1_cfprf import COMMIT, copy_source, patch_frontend
        runtime = self.scratch / "cfprf_native"
        runtime.mkdir(parents=True, exist_ok=True)
        hashes = copy_source(self.repo, runtime)
        patch = patch_frontend(runtime, self.frontend.resolve())
        import torch
        native = _source_import(runtime, "evaluate_CFPRF")
        device = torch.device(self.device_name)
        with torch.cuda.device(device):
            fdn = native.CFPRF_FDN(seq_len=1070, gmlp_layers=1).to(device)
            prn = native.CFPRF_PRN(device=device).to(device)
        fdn.load_state_dict(_torch_load(torch, self.fdn_path), strict=True)
        prn.load_state_dict(_torch_load(torch, self.prn_path), strict=True)
        self.torch, self.device, self.native = torch, device, native
        geometry = inspect_ssl_geometry(fdn.asr, torch)
        self.models, self.geometry = (fdn.eval(), prn.eval()), geometry
        self.identity = {"commit": COMMIT, "source_files": hashes, "frontend_path_patch": patch,
                         "strict_checkpoint_load": True, "cpu_capability": self.cpu_capability,
                         "labels": "ones used only for native alignment; no supervision/evaluation labels"}

    def score_chunk(self, waveform):
        if len(waveform) > 340000:
            raise ValueError("CFPRF native skip limit exceeded; score actual prepared chunks")
        self.load()
        torch, (fdn, prn) = self.torch, self.models
        expected = feature_count(len(waveform), self.geometry)
        captured = []
        def capture(module, inputs, output):
            captured.append((_array(output[0])[0], _array(output[1])[0]))
        handle = fdn.register_forward_hook(capture)
        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32).copy()).unsqueeze(0)
        labels = torch.ones((1, expected), dtype=torch.float32)
        try:
            scores, _, coarse, verified, refined = self.native.Inference_CFPRF(
                [(tensor, ["chunk"], labels)], fdn, prn, 20, self.device)
        finally:
            handle.remove()
        if len(captured) != 1 or "chunk" not in scores:
            raise ValueError("Native CFPRF failed to score the chunk exactly once")
        logits, boundary = captured[0]
        if logits.shape != (expected, 2) or boundary.shape != (expected, 2):
            raise ValueError("CFPRF outputs differ from its loaded native feature geometry")
        if not np.array_equal(logits[:, 1], np.asarray(scores["chunk"])):
            raise ValueError("Native CFPRF cached class-1 scores differ from captured FDN logits")
        proposals = {stage: np.asarray(values["chunk"]).copy() for stage, values in
                     (("coarse", coarse), ("verified", verified), ("refined", refined))}
        return {"logits": logits, "boundary": boundary, "proposals": proposals,
                "padding_samples": 0, "native_internal_padding_samples": 0}


class CaptureFrontend:
    """Delegate the frozen transform, recording each whole-file result once."""
    def __init__(self, frontend):
        self.delegate, self.canonical_samples, self.normalized_span = frontend, None, None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def canonical(self, *args, **kwargs):
        if self.canonical_samples is not None:
            raise RuntimeError("The canonical decoder was invoked more than once")
        samples, rate = self.delegate.canonical(*args, **kwargs)
        self.canonical_samples = np.asarray(samples).copy()
        return samples, rate

    def frontend(self, *args, **kwargs):
        if self.normalized_span is not None:
            raise RuntimeError("Whole-file normalization was invoked more than once")
        normalized = self.delegate.frontend(*args, **kwargs)
        self.normalized_span = np.asarray(normalized).copy()
        return normalized


class CandidateScorer:
    def __init__(self, backend):
        self.backend = backend

    def score_prepared(self, prepared, normalized_span, *, total_samples, metadata=None):
        total = _integer(total_samples, "total_samples")
        trim_start = _integer(prepared.trim_start, "trim_start")
        trim_end = _integer(prepared.trim_end, "trim_end")
        span = np.asarray(normalized_span)
        if span.ndim != 1 or span.dtype != np.float32 or not np.isfinite(span).all():
            raise ValueError("Expected the globally normalized finite float32 span")
        if not 0 <= trim_start < trim_end <= total or len(span) != trim_end - trim_start:
            raise ValueError("Cached trim clock differs from the normalized span")
        if len(prepared.starts) != len(prepared.windows) or not prepared.starts:
            raise ValueError("Prepared window arrays and actual starts differ")
        starts = [_integer(value, "window start") for value in prepared.starts]
        if starts != sorted(set(starts)) or starts[0] != 0:
            raise ValueError("Actual prepared starts must be ordered, unique and begin at zero")
        chunks, aggregation = [], []
        for start, waveform in zip(starts, prepared.windows):
            waveform = np.asarray(waveform)
            if waveform.shape != (WINDOW,) or waveform.dtype != np.float32 or start < 0 or start + len(waveform) > len(span):
                raise ValueError("Expected actual unpadded four-second prepared windows")
            if not np.array_equal(waveform, span[start:start + len(waveform)]):
                raise ValueError("Window differs from the captured global normalization")
            raw = self.backend.score_chunk(waveform)
            logits, boundary = np.asarray(raw["logits"]), np.asarray(raw["boundary"])
            if logits.ndim != 2 or logits.shape[1] != 2 or boundary.shape not in ((len(logits),), (len(logits), 2)):
                raise ValueError("Invalid native two-class/boundary shape")
            if not np.isfinite(logits).all() or not np.isfinite(boundary).all():
                raise ValueError("Nonfinite native output")
            supports, valid = native_supports(len(logits), self.backend.hop_samples, len(waveform))
            proposals = raw["proposals"]
            for values in proposals.values():
                array = np.asarray(values)
                if array.size and (array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all()):
                    raise ValueError("Invalid native proposal array")
            chunk = dict(raw, origin_sample=trim_start + start, valid_samples=len(waveform),
                         local_supports=supports, valid_mask=valid, logits=logits, boundary=boundary,
                         unscored_tail_samples=len(waveform) - int(supports[-1, 1]) if len(supports) else len(waveform))
            chunks.append(chunk)
            aggregation.append(dict(local_supports=supports, scores=-logits[:, 1], valid_mask=valid,
                chunk_origin=start, trim_offset=trim_start, valid_local_samples=len(waveform)))
        if starts[-1] + WINDOW != len(span):
            raise ValueError("Actual prepared tail window does not reach the trimmed span end")
        supports, scores = aggregate_frames(aggregation, total)
        rolling = max_rolling_mean(supports, scores, total)
        return {"supports": supports, "scores": scores, "statistic": rolling["statistic"], "rolling": rolling,
                "chunks": chunks, "metadata": {"model": self.backend.name, "trim_start": trim_start, "trim_end": trim_end,
                    "total_samples": total, "sample_rate": SR, "frame_hop_s": self.backend.hop_samples / SR,
                    "geometry": self.backend.geometry, "orientation": ORIENTATION,
                    "proposal_coordinates": "native local seconds, columns confidence/start/end; unmerged and unclipped",
                    "native_decision_bins": "[i*hop,(i+1)*hop), clipped only to real input; no CF tail stretching",
                    "normalized_float32_sha256": hashlib.sha256(span.astype("<f4", copy=False).tobytes()).hexdigest(),
                    "identity": self.backend.identity, "input": dict(metadata or {})}}

    def score_audio(self, audio_bytes, *, temp_dir, ffmpeg="ffmpeg", decode_timeout=15):
        from pipeline import preprocess
        from pipeline.frozen import frontend, audio_windows
        capture = CaptureFrontend(frontend)
        prepared = preprocess.prepare(audio_bytes, ffmpeg=ffmpeg, temp_dir=temp_dir, decode_timeout=decode_timeout,
                                      modules=(capture, audio_windows))
        if capture.canonical_samples is None or capture.normalized_span is None:
            raise RuntimeError("Frozen prepare did not return captured whole-file transforms")
        canonical = capture.canonical_samples
        canonical_hash = hashlib.sha256(canonical.astype("<f8", copy=False).tobytes()).hexdigest()
        return self.score_prepared(prepared, capture.normalized_span, total_samples=len(canonical),
            metadata={"canonical_float64_sha256": canonical_hash, "native_rate": prepared.native_rate,
                      "duration_seconds": prepared.duration, "active_seconds": prepared.active_seconds})
