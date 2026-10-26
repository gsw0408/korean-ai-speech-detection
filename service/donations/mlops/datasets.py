"""Immutable, hash-identified dataset versions built only from reviewed, training-consented contributions."""
import random
from pathlib import Path

from .common import PolicyError, canonical_hash, now, read_json, refuse_protected, write_json
from .review import require_admissible


def build_version(intake_dir, datasets_dir, *, label, seed, dev_fraction=0.2, parent=None, **admission):
    """Write a new dataset version. Each contribution is its own group: no speaker identity is collected, so
    near-duplicate recordings by one person can still fall on both sides of the split (documented limitation)."""
    refuse_protected(label)
    admitted, refused = require_admissible(intake_dir, **admission)
    order = sorted(admitted, key=lambda item: item["contribution_id"])
    random.Random(seed).shuffle(order)
    dev_count = max(1, round(len(order) * dev_fraction)) if len(order) > 1 else 0
    items = [dict(item, split="dev" if index < dev_count else "train") for index, item in enumerate(order)]
    config = {"label": label, "seed": seed, "dev_fraction": dev_fraction, "parent": parent}
    version_id = canonical_hash({"config": config, "items": sorted((i["contribution_id"], i["sha256"], i["label"], i["split"]) for i in items)})[:16]
    target = Path(datasets_dir) / version_id
    if (target / "manifest.json").exists():
        raise PolicyError(f"dataset version {version_id} already exists; versions are immutable")
    manifest = {"version_id": version_id, "created_at": now(), "status": "active", "config": config, "items": items,
                "counts": {"train": sum(i["split"] == "train" for i in items), "dev": sum(i["split"] == "dev" for i in items),
                           "real": sum(i["label"] == "real" for i in items), "synthetic": sum(i["label"] == "synthetic" for i in items)},
                "refused": refused, "sources": "donation intake only (no CALIB/FINAL/research corpora)"}
    write_json(target / "manifest.json", manifest)
    return manifest


def rebuild_after_withdrawal(intake_dir, datasets_dir, superseded_version, *, seed, **admission):
    """New version from the same configuration minus withdrawn items; the superseded one stays as an audit record."""
    old = read_json(Path(datasets_dir) / superseded_version / "manifest.json")
    if old.get("status") != "superseded":
        raise PolicyError("only superseded versions are rebuilt")
    return build_version(intake_dir, datasets_dir, label=old["config"]["label"] + "-rebuild", seed=seed,
                         dev_fraction=old["config"]["dev_fraction"], parent=superseded_version, **admission)
