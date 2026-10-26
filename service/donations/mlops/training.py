"""Training plans. This package records what a retraining run would do; it never runs one."""
from dataclasses import asdict, dataclass
from pathlib import Path

from .common import PolicyError, canonical_hash, now, read_json, write_json


class RetrainingDisabled(PolicyError):
    """Actual retraining is out of scope for this project phase and requires a separate owner decision."""


@dataclass(frozen=True)
class TrainingSpec:
    base_bundle: str
    dataset_version: str
    recipe: str = "nii-small-finetune-candidate"
    seed: int = 20261023
    max_steps: int = 600
    learning_rate: float = 1e-5
    freeze_backbone_layers: int = 8
    checkpoint_rule: str = "final step only (no selection on dev)"


def plan(spec, datasets_dir, plans_dir):
    manifest = read_json(Path(datasets_dir) / spec.dataset_version / "manifest.json")
    if manifest.get("status") != "active":
        raise PolicyError("training may only use an active (not superseded) dataset version")
    if manifest["counts"]["train"] == 0 or manifest["counts"]["dev"] == 0:
        raise PolicyError("dataset version needs both train and dev items")
    body = {"spec": asdict(spec), "dataset_manifest_hash": canonical_hash(manifest), "created_at": now(), "status": "planned",
            "execution": "not run: retraining is disabled in this phase", "outputs": "a candidate bundle registered via registry.py"}
    plan_id = canonical_hash({"spec": body["spec"], "dataset": body["dataset_manifest_hash"]})[:16]
    body["plan_id"] = plan_id
    write_json(Path(plans_dir) / f"{plan_id}.json", body)
    return body


def run(*_args, **_kwargs):
    raise RetrainingDisabled("retraining is not executed by this pipeline; it needs an explicit owner decision and a reviewed run plan")
