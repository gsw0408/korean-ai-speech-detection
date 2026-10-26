import numpy as np
from calibration.fit_nii_v1 import fit, outcome_table


def test_threshold_ties_and_exact_float64_extrema():
    rows = [{"label_fake": label, "mean_negative_bonafide_logit": float(score)} for label in (0, 1) for score in range(100)]
    low, warn = fit(rows)
    assert warn == np.nextafter(94., np.inf)
    assert low == np.nextafter(1., -np.inf)
    assert sum(r["mean_negative_bonafide_logit"] >= warn for r in rows if not r["label_fake"]) == 5
    assert sum(r["mean_negative_bonafide_logit"] <= low for r in rows if r["label_fake"]) == 1


def test_rejections_stay_in_outcome_denominator():
    rows = [{"sample_id": str(i), "engine": "fixture", "label_fake": 1, "mean_negative_bonafide_logit": 10} for i in range(10)]
    table = outcome_table(rows, lambda r: (-1, 1), {"0": "too_short"})
    assert table["pooled/AI"]["n"] == 10
    assert table["pooled/AI"]["rates"]["warn"] == .9
    assert table["pooled/AI"]["rates"]["cannot"] == .1
