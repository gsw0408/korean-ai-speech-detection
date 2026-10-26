"""Promotion gate comparing a candidate with the current model on the same held-out scores (higher score = AI)."""
import numpy as np

from .common import canonical_hash, now, refuse_protected


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if not len(positives) or not len(negatives):
        return None
    greater = (positives[:, None] > negatives[None, :]).sum() + 0.5 * (positives[:, None] == negatives[None, :]).sum()
    return float(greater / (len(positives) * len(negatives)))


def evaluate(set_name, current, candidate, labels, *, current_threshold, candidate_threshold,
             max_real_warning_increase=0.0, draws=2000, seed=20261023):
    """Promote only if the candidate's AUC gain has a positive 95% lower bound and its false warnings on real
    speech do not rise by more than the allowed margin. Scores must come from a held-out set, never CALIB/FINAL."""
    refuse_protected(set_name)
    current, candidate, labels = (np.asarray(x, float) for x in (current, candidate, labels))
    if not (len(current) == len(candidate) == len(labels)) or len(labels) < 4:
        raise ValueError("score arrays must be aligned and non-trivial")
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(draws):
        index = rng.integers(0, len(labels), len(labels))
        a, b = auc(current[index], labels[index]), auc(candidate[index], labels[index])
        if a is not None and b is not None:
            gains.append(b - a)
    low, high = (float(np.percentile(gains, q)) for q in (2.5, 97.5))
    real = labels == 0
    fw_current = float(np.mean(current[real] >= current_threshold))
    fw_candidate = float(np.mean(candidate[real] >= candidate_threshold))
    checks = {"auc_gain_ci_lower_positive": low > 0, "real_false_warning_not_worse": fw_candidate - fw_current <= max_real_warning_increase}
    report = {"set": set_name, "n": int(len(labels)), "auc_current": auc(current, labels), "auc_candidate": auc(candidate, labels),
              "auc_gain_ci95": [low, high], "real_false_warning": {"current": fw_current, "candidate": fw_candidate},
              "checks": checks, "decision": "promote" if all(checks.values()) else "reject", "created_at": now(),
              "note": "internal held-out estimate; not an independent validation"}
    report["report_hash"] = canonical_hash({k: v for k, v in report.items() if k != "created_at"})
    return report
