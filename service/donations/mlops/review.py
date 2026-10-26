"""Human review: the only source of labels, plus the rights and consent checks needed before admission."""
from pathlib import Path

from . import TRAINING_CONSENT_VERSIONS
from .common import PolicyError, now, read_json, write_json

LABELS = ("real", "synthetic", "exclude")


def record_review(intake_dir, contribution_id, *, label, rights_confirmed, reviewer, notes=""):
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    if not reviewer:
        raise ValueError("reviewer is required")
    item = Path(intake_dir) / "items" / contribution_id
    if not (item / "meta.json").exists():
        raise KeyError("unknown or withdrawn contribution")
    review = {"label": label, "rights_confirmed": bool(rights_confirmed), "reviewer": reviewer, "notes": notes, "reviewed_at": now()}
    write_json(item / "review.json", review)
    return review


def admissible(intake_dir, *, training_consent_versions=TRAINING_CONSENT_VERSIONS):
    """Items that may enter a training dataset, and the reasons the others may not."""
    admitted, refused = [], {}
    items = Path(intake_dir) / "items"
    for item in sorted(items.iterdir()) if items.is_dir() else []:
        meta = read_json(item / "meta.json")
        review = read_json(item / "review.json") if (item / "review.json").exists() else None
        reason = (None if meta["consent_version"] in training_consent_versions else "consent_does_not_cover_training")
        reason = reason or (None if review else "not_reviewed")
        reason = reason or (None if review["rights_confirmed"] else "rights_not_confirmed")
        reason = reason or (None if review["label"] != "exclude" else "excluded_by_reviewer")
        if reason:
            refused[reason] = refused.get(reason, 0) + 1
        else:
            admitted.append({"contribution_id": meta["contribution_id"], "sha256": meta["sha256"],
                             "consent_version": meta["consent_version"], "label": review["label"]})
    return admitted, refused


def require_admissible(intake_dir, **kwargs):
    admitted, refused = admissible(intake_dir, **kwargs)
    if not admitted:
        raise PolicyError(f"no admissible contributions (refused: {refused or 'none available'})")
    return admitted, refused
