"""Data-independent clocks, grouping and TRAIN-only calibration primitives."""

import hashlib
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat

import numpy as np


SR = 16000
FOLD_SEED = 2026101502
PROTECTED = re.compile(r"calib|final|vl_data|verification", re.IGNORECASE)


def _within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must use the integer sample clock")
    return int(value)


def clip_interval(start, end, total_samples):
    """Clip a half-open interval to real decoded audio; return None when empty."""
    start, end, total = (_integer(value, name) for value, name in
                         ((start, "start"), (end, "end"), (total_samples, "total_samples")))
    if end < start or total < 0:
        raise ValueError("Invalid half-open interval or decoded length")
    left, right = max(0, min(start, total)), max(0, min(end, total))
    return (left, right) if right > left else None


def resample_interval(start, end, source_rate, *, target_length, target_rate=SR, decoder_delay_samples=0):
    """Map boundaries rationally (floor start, ceil end), then delay and clip.

    Positive decoder_delay_samples means the decoded event occurs later. The
    caller supplies its measured value; this function never guesses codec delay.
    """
    start, end = _integer(start, "start"), _integer(end, "end")
    source_rate, target_rate = _integer(source_rate, "source_rate"), _integer(target_rate, "target_rate")
    delay = _integer(decoder_delay_samples, "decoder_delay_samples")
    if source_rate <= 0 or target_rate <= 0 or end < start:
        raise ValueError("Invalid resampling rates or interval")
    if start == end:
        return clip_interval(0, 0, target_length)
    left = start * target_rate // source_rate + delay
    right = -((-end * target_rate) // source_rate) + delay
    return clip_interval(left, right, target_length)


def map_frame_supports(local_supports, *, chunk_origin, trim_offset=0, total_samples, valid_local_samples=None):
    """Map actual chunk supports to the original clock, retaining a validity mask."""
    origin, trim = _integer(chunk_origin, "chunk_origin"), _integer(trim_offset, "trim_offset")
    total = _integer(total_samples, "total_samples")
    if trim < 0 or total < 0:
        raise ValueError("Invalid trim offset or decoded length")
    local_limit = None if valid_local_samples is None else _integer(valid_local_samples, "valid_local_samples")
    if local_limit is not None and local_limit < 0:
        raise ValueError("Invalid unpadded chunk length")
    supports = np.asarray(local_supports)
    if supports.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=bool)
    if supports.ndim != 2 or supports.shape[1] != 2:
        raise ValueError("Frame supports must be pairs of integer boundaries")
    mapped, valid = np.zeros(supports.shape, dtype=np.int64), np.zeros(len(supports), dtype=bool)
    for index, (left, right) in enumerate(supports):
        left, right = _integer(left, "frame_start"), _integer(right, "frame_end")
        if left < 0 or right < left:
            raise ValueError("Invalid local frame support")
        if local_limit is not None:
            left, right = min(left, local_limit), min(right, local_limit)
        clipped = clip_interval(origin + trim + left, origin + trim + right, total)
        if clipped is not None:
            mapped[index], valid[index] = clipped, True
    return mapped, valid


def aggregate_frames(chunks, total_samples):
    """Mean overlapping valid supports without counting overlapping time twice.

    Each chunk supplies local_supports, scores, chunk_origin, and optionally
    trim_offset, valid_local_samples and valid_mask. Gaps remain unobserved.
    """
    events, values = {}, {}
    next_id = 0
    for chunk in chunks:
        supports, valid = map_frame_supports(chunk["local_supports"], chunk_origin=chunk["chunk_origin"],
            trim_offset=chunk.get("trim_offset", 0), total_samples=total_samples,
            valid_local_samples=chunk.get("valid_local_samples"))
        scores = np.asarray(chunk["scores"], dtype=float)
        supplied = np.asarray(chunk.get("valid_mask", np.ones(len(scores), dtype=bool)))
        if scores.shape != (len(supports),) or supplied.shape != scores.shape or supplied.dtype != np.bool_:
            raise ValueError("Frame scores and validity mask differ from supports")
        if not np.isfinite(scores).all():
            raise ValueError("Frame scores must be finite")
        for (left, right), score, keep in zip(supports, scores, valid & supplied):
            if not keep:
                continue
            values[next_id] = float(score)
            events.setdefault(int(left), []).append((next_id, True))
            events.setdefault(int(right), []).append((next_id, False))
            next_id += 1
    boundaries, active, intervals, scores = sorted(events), {}, [], []
    for index, left in enumerate(boundaries[:-1]):
        for item, add in events[left]:
            if add:
                active[item] = values[item]
            else:
                active.pop(item)
        right = boundaries[index + 1]
        if active and right > left:
            mean = math.fsum(active.values()) / len(active)
            if intervals and intervals[-1][1] == left and scores[-1] == mean:
                intervals[-1] = (intervals[-1][0], right)
            else:
                intervals.append((left, right))
                scores.append(mean)
    return np.asarray(intervals, dtype=np.int64).reshape(-1, 2), np.asarray(scores, dtype=float)


def max_rolling_mean(supports, scores, total_samples):
    """Exact support integration over every fully observed one-second window.

    For piecewise-constant scores, maxima occur at a support boundary or one
    second before a boundary. Considering both includes partial BAM bins and
    the actual tail window; gaps never acquire synthetic scores.
    """
    total = _integer(total_samples, "total_samples")
    if total < SR:
        raise ValueError("Files shorter than one second are outside the contract")
    intervals = np.asarray(supports)
    values = np.asarray(scores, dtype=float)
    if intervals.ndim != 2 or intervals.shape[1] != 2 or values.shape != (len(intervals),) or not np.isfinite(values).all():
        raise ValueError("Invalid observed supports/scores")
    clean = []
    for index, (left, right) in enumerate(intervals):
        left, right = _integer(left, "support_start"), _integer(right, "support_end")
        if not 0 <= left < right <= total or (index and left < clean[-1][1]):
            raise ValueError("Rolling supports must be ordered, nonoverlapping and valid")
        clean.append((left, right))
    best = None
    first = 0
    while first < len(clean):
        last = first + 1
        while last < len(clean) and clean[last - 1][1] == clean[last][0]:
            last += 1
        run = np.asarray(clean[first:last], dtype=np.int64)
        run_scores = values[first:last]
        low, high = int(run[0, 0]), int(run[-1, 1]) - SR
        if high >= low:
            prefix = np.concatenate(([0.0], np.cumsum((run[:, 1] - run[:, 0]) * run_scores)))
            def integral(position):
                index = int(np.searchsorted(run[:, 0], position, side="right") - 1)
                return float(prefix[index] + run_scores[index] * (position - run[index, 0]))
            edges = [int(value) for value in run.reshape(-1)]
            candidates = sorted({low, high} | {edge for edge in edges if low <= edge <= high}
                                | {edge - SR for edge in edges if low <= edge - SR <= high})
            for left in candidates:
                value = (integral(left + SR) - integral(left)) / SR
                if best is None or value > best["statistic"]:
                    best = {"statistic": value, "start_sample": left, "end_sample": left + SR}
        first = last
    if best is None:
        raise ValueError("No fully observed one-second window exists")
    return best


def merge_intervals(intervals, total_samples=None):
    """Union overlapping/touching positive intervals, with no gap filling."""
    clean = []
    for left, right in intervals:
        left, right = _integer(left, "start"), _integer(right, "end")
        if right < left or (total_samples is None and left < 0):
            raise ValueError("Invalid detection interval")
        clipped = clip_interval(left, right, total_samples) if total_samples is not None else ((left, right) if right > left else None)
        if clipped is not None:
            clean.append(clipped)
    merged = []
    for left, right in sorted(clean):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return merged


def file_flag(statistic, threshold):
    if not math.isfinite(statistic) or not math.isfinite(threshold):
        raise ValueError("A file decision needs finite statistic and threshold")
    return bool(statistic > threshold)


def guard_dataset_path(path, *, workspace, dataset_root=None, must_exist=False):
    """Validate one explicit path without listing data or following reparse links."""
    raw = str(path)
    parts = re.split(r"[\\/]", raw)
    if "\x00" in raw or any(part.strip() == ".." for part in parts):
        raise ValueError("Dataset path traversal is forbidden")
    if os.name != "nt" and PureWindowsPath(raw).is_absolute():
        raise ValueError("A Windows dataset path cannot be resolved on this host")
    native = Path(raw.replace("\\", "/"))
    if not native.is_absolute():
        if dataset_root is None:
            raise ValueError("Dataset paths require an explicit absolute root")
        native = Path(dataset_root) / native
    native = native.absolute()
    workspace = Path(workspace).absolute()
    inside_workspace = _within(native, workspace)
    if not inside_workspace and any(PROTECTED.search(part) for part in native.parts):
        raise ValueError("Protected dataset path is forbidden")
    for component in [*reversed(native.parents), native]:
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError("Dataset symlinks and reparse points are forbidden")
    resolved = native.resolve()
    if dataset_root is not None and not _within(resolved, Path(dataset_root).resolve()):
        raise ValueError("Dataset path escapes its authorized root")
    if not _within(resolved, workspace.resolve()) and any(PROTECTED.search(part) for part in resolved.parts):
        raise ValueError("Protected resolved dataset path is forbidden")
    if must_exist and not resolved.is_file():
        raise ValueError("Dataset file does not exist")
    return resolved


def guard_dataset_row(row, *, dataset_root, workspace, must_exist=False):
    roles = {"train": "TRAIN", "dev": "DEV_ASSESSMENT"}
    split = row.get("split")
    if split not in roles or row.get("allocation_role") != roles[split]:
        raise ValueError("Unauthorized dataset split/allocation role")
    raw = row.get("audio_path", "")
    supplied = PurePosixPath(raw)
    if (not raw or "\\" in raw or supplied.is_absolute() or ".." in supplied.parts
            or supplied.parts[:2] != ("audio", split) or any(PROTECTED.search(part) for part in supplied.parts)):
        raise ValueError("Unauthorized dataset row path")
    return guard_dataset_path(raw, workspace=workspace, dataset_root=dataset_root, must_exist=must_exist)


def connected_components(identity_links, *, weights=None):
    """Group supplied identities without assigning cross-validation folds.

    Keys are stable sample/recipe IDs. A source: token retains an unavailable
    source identity for grouping only; it adds no sample or speaker identity.
    Present sources use sample:. Generic engines cannot become speakers.
    """
    if not identity_links:
        raise ValueError("Missing stable sample identity/provenance links")
    parent = {}
    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    for item, links in identity_links.items():
        if not isinstance(item, str) or not item or not links:
            raise ValueError("Missing stable sample identity/provenance links")
        node = "sample:" + item
        for link in links:
            if not isinstance(link, str):
                raise ValueError("Identity links must be typed strings")
            namespace, separator, value = link.partition(":")
            if separator != ":" or namespace not in ("speaker", "recording", "sample", "source") or not value:
                raise ValueError("Missing/unsupported identity linkage; engines are not identities")
            if namespace == "sample" and value not in identity_links:
                raise ValueError("Generation/recipe source identity is missing")
            union(node, link)
    members, nodes = {}, {}
    for node in parent:
        nodes.setdefault(find(node), []).append(node)
    for item in identity_links:
        members.setdefault(find("sample:" + item), []).append(item)
    components = []
    for root, values in members.items():
        if not any(node.startswith("speaker:") for node in nodes[root]):
            raise ValueError("A component has no speaker identity linkage")
        values = sorted(values)
        component = hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
        item_weights = [1 if weights is None else _integer(weights[item], "component weight") for item in values]
        if any(weight <= 0 for weight in item_weights):
            raise ValueError("Component weights must be positive")
        weight = sum(item_weights)
        components.append({"id": component, "sample_ids": values, "identity_nodes": sorted(nodes[root]), "weight": weight})
    return {"component_by_id": {item: entry["id"] for entry in components for item in entry["sample_ids"]},
            "components": components}


def component_folds(identity_links, *, weights=None, folds=5, seed=FOLD_SEED):
    """Assign the preregistered five TRAIN folds to complete identity components."""
    if folds != 5 or not identity_links:
        raise ValueError("The preregistered protocol requires five folds")
    components = connected_components(identity_links, weights=weights)["components"]
    for component in components:
        component["tie"] = hashlib.sha256((str(seed) + "|" + "\n".join(component["sample_ids"])).encode("utf-8")).hexdigest()
    if len(components) < folds:
        raise ValueError("At least five independent speaker-connected components are required")
    loads, assignment, component_by_id = [0] * folds, {}, {}
    for component in sorted(components, key=lambda entry: (-entry["weight"], entry["tie"])):
        fold = min(range(folds), key=lambda index: (loads[index], index))
        loads[fold] += component["weight"]
        component["fold"] = fold
        for item in component["sample_ids"]:
            assignment[item], component_by_id[item] = fold, component["id"]
    assert_fold_isolation(identity_links, assignment)
    return {"fold_by_id": assignment, "component_by_id": component_by_id, "components": components,
            "fold_weights": loads, "seed": seed, "fold_count": folds}


def assert_fold_isolation(identity_links, fold_by_id):
    if set(identity_links) != set(fold_by_id):
        raise ValueError("Fold assignment does not cover exactly the linked samples")
    seen = {}
    for item, links in identity_links.items():
        for node in ["sample:" + item, *links]:
            if node in seen and seen[node] != fold_by_id[item]:
                raise ValueError("Identity/source provenance crosses folds")
            seen[node] = fold_by_id[item]
    return True


def duration_threshold(coefficients, duration_samples):
    duration = _integer(duration_samples, "duration_samples")
    a, b = float(coefficients["a"]), float(coefficients["b"])
    if duration <= 0 or not math.isfinite(a) or not math.isfinite(b) or b < 0:
        raise ValueError("Invalid duration threshold or decoded duration")
    return a + b * math.log(max(duration / SR, 1.0))


def fit_duration_threshold(rows):
    """Fit genuine TRAIN only: nonnegative Theil-Sen slope, higher 95% residual."""
    rows = list(rows)
    if not rows:
        raise ValueError("No genuine fitting observations")
    x, y = [], []
    for row in rows:
        if row.get("split") != "train" or row.get("allocation_role") != "TRAIN" or row.get("label_fake") != 0:
            raise ValueError("Threshold fitting accepts genuine TRAIN observations only")
        duration = _integer(row["duration_samples"], "duration_samples")
        value = float(row["statistic"])
        if duration <= 0 or not math.isfinite(value):
            raise ValueError("Invalid threshold fitting observation")
        x.append(math.log(max(duration / SR, 1.0)))
        y.append(value)
    x, y = np.asarray(x), np.asarray(y)
    left, right = np.triu_indices(len(x), 1)
    different = x[right] != x[left]
    slopes = (y[right[different]] - y[left[different]]) / (x[right[different]] - x[left[different]])
    b = max(0.0, float(np.median(slopes))) if len(slopes) else 0.0
    residual = y - b * x
    try:
        a = float(np.quantile(residual, 0.95, method="higher"))
    except TypeError:
        a = float(np.quantile(residual, 0.95, interpolation="higher"))
    return {"a": a, "b": b, "q": 0.95, "fit_count": len(rows), "fit_split": "train", "fit_allocation_role": "TRAIN",
            "fitting_false_alarm_rate": float(np.mean(y > a + b * x))}
