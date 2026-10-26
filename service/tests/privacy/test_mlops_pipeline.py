import json

import numpy as np
import pytest

from donations.mlops import TRAINING_CONSENT_VERSIONS, datasets, evaluation, intake, review, training
from donations.mlops.common import PolicyError, read_json
from donations.mlops.registry import Registry
from donations.store import CONSENT_VERSION, DonationStore
from scripts import mlops as cli

TRAINING_OK = frozenset({CONSENT_VERSION})  # test-only stand-in for a future, owner-approved training consent


def donate_many(store, count):
    return [store.donate(store.offer(bytes([index + 1]) * 64, "low", "bundle")) for index in range(count)]


@pytest.fixture
def stored(tmp_path):
    store = DonationStore(tmp_path / "store", store_audio=True)
    codes = donate_many(store, 6)
    return store, codes, tmp_path / "work"


def review_all(work, label="real"):
    for index, item in enumerate(sorted((work / "intake" / "items").iterdir())):
        review.record_review(work / "intake", item.name, label=label if index % 2 else "synthetic", rights_confirmed=True, reviewer="r1")


def test_no_audio_policy_disables_intake(tmp_path):
    store = DonationStore(tmp_path / "store", store_audio=False)
    donate_many(store, 2)
    result = intake.intake(tmp_path / "store", tmp_path / "work" / "intake", store_audio=False)
    assert result["status"] == "disabled_by_policy" and result["added"] == 0
    assert not (tmp_path / "work" / "intake").exists()


def test_existing_consent_versions_never_allow_training(stored):
    store, _, work = stored
    assert TRAINING_CONSENT_VERSIONS == frozenset()
    assert intake.intake(store.directory, work / "intake", store_audio=True)["added"] == 6
    review_all(work)
    admitted, refused = review.admissible(work / "intake")
    assert admitted == [] and refused == {"consent_does_not_cover_training": 6}
    with pytest.raises(PolicyError):
        datasets.build_version(work / "intake", work / "datasets", label="v1", seed=1)


def test_labels_come_only_from_review_and_rights_are_required(stored):
    store, _, work = stored
    intake.intake(store.directory, work / "intake", store_audio=True)
    meta = read_json(next((work / "intake" / "items").iterdir()) / "meta.json")
    assert "outcome" not in meta  # the model's own verdict is never carried into training data
    first, second = sorted(p.name for p in (work / "intake" / "items").iterdir())[:2]
    review.record_review(work / "intake", first, label="real", rights_confirmed=False, reviewer="r1")
    review.record_review(work / "intake", second, label="exclude", rights_confirmed=True, reviewer="r1")
    _, refused = review.admissible(work / "intake", training_consent_versions=TRAINING_OK)
    assert refused == {"rights_not_confirmed": 1, "excluded_by_reviewer": 1, "not_reviewed": 4}
    with pytest.raises(ValueError):
        review.record_review(work / "intake", first, label="maybe", rights_confirmed=True, reviewer="r1")


def test_versions_are_immutable_and_withdrawal_propagates(stored):
    store, codes, work = stored
    intake.intake(store.directory, work / "intake", store_audio=True)
    review_all(work)
    manifest = datasets.build_version(work / "intake", work / "datasets", label="v1", seed=7, training_consent_versions=TRAINING_OK)
    assert manifest["counts"]["train"] + manifest["counts"]["dev"] == 6 and manifest["counts"]["dev"] >= 1
    with pytest.raises(PolicyError):
        datasets.build_version(work / "intake", work / "datasets", label="v1", seed=7, training_consent_versions=TRAINING_OK)
    store.withdraw(codes[0])
    result = intake.sync_withdrawals(store.directory, work / "intake", work / "datasets")
    assert result == {"removed": 1, "superseded_versions": [manifest["version_id"]]}
    old = read_json(work / "datasets" / manifest["version_id"] / "manifest.json")
    assert old["status"] == "superseded" and old["withdrawn_count"] == 1
    assert sum(1 for entry in old["items"] if "contribution_id" in entry) == 5  # withdrawn id and hash scrubbed
    log = read_json(work / "intake" / "withdrawal_log.json")
    assert log[0]["removed"] == 1 and "ids" not in json.dumps(log)
    rebuilt = datasets.rebuild_after_withdrawal(work / "intake", work / "datasets", manifest["version_id"], seed=7,
                                                training_consent_versions=TRAINING_OK)
    assert rebuilt["config"]["parent"] == manifest["version_id"] and len(rebuilt["items"]) == 5
    with pytest.raises(PolicyError):
        training.plan(training.TrainingSpec("bundles/nii-small-v1", manifest["version_id"]), work / "datasets", work / "plans")
    plan = training.plan(training.TrainingSpec("bundles/nii-small-v1", rebuilt["version_id"]), work / "datasets", work / "plans")
    assert plan["status"] == "planned" and "not run" in plan["execution"]
    with pytest.raises(training.RetrainingDisabled):
        training.run(plan)


def test_protected_sets_are_refused(stored):
    store, _, work = stored
    for name in ("FINAL_v2", "calib-scores", "sealed"):
        with pytest.raises(PolicyError):
            evaluation.evaluate(name, [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], current_threshold=0.5, candidate_threshold=0.5)
        with pytest.raises(PolicyError):
            datasets.build_version(work / "intake", work / "datasets", label=name, seed=1)


def test_evaluation_gate_and_registry_promotion():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 60 + [1] * 60)
    current = labels + rng.normal(0, 1.2, 120)
    better = labels * 2.5 + rng.normal(0, 0.8, 120)
    good = evaluation.evaluate("contrib-dev-v1", current, better, labels, current_threshold=0.5, candidate_threshold=1.2)
    bad = evaluation.evaluate("contrib-dev-v1", current, current[::-1], labels, current_threshold=0.5, candidate_threshold=0.5)
    assert good["decision"] == "promote" and good["auc_gain_ci95"][0] > 0
    assert bad["decision"] == "reject"


def test_registry_requires_passing_evaluation_and_owner_approval(tmp_path):
    registry = Registry(tmp_path / "registry.json")
    passing = {"decision": "promote", "report_hash": "a" * 64}
    failing = {"decision": "reject", "report_hash": "b" * 64}
    registry.register("m1", bundle_hashes={"detector": "x"}, dataset_version="d1", plan_id="p1", evaluation=passing)
    registry.register("m2", bundle_hashes={"detector": "y"}, dataset_version="d2", plan_id="p2", evaluation=passing)
    registry.register("m3", bundle_hashes={"detector": "z"}, dataset_version="d3", plan_id="p3", evaluation=failing)
    with pytest.raises(PolicyError):
        registry.promote("m1", owner_approval=" ")
    with pytest.raises(PolicyError):
        registry.promote("m3", owner_approval="approved")
    with pytest.raises(PolicyError):
        registry.register("m1", bundle_hashes={}, dataset_version="d", plan_id="p", evaluation=passing)
    registry.promote("m1", owner_approval="owner approved 2026-10-23")
    registry.promote("m2", owner_approval="owner approved 2026-10-24")
    assert registry.rollback(reason="regression") == "m1"
    reloaded = Registry(tmp_path / "registry.json")
    assert reloaded.data["active"] == "m1" and reloaded.lineage("m2")["status"] == "rolled_back"


def test_cli_status_and_refusals(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DONATION_STORE_AUDIO", raising=False)
    assert cli.main(["intake", "--store", str(tmp_path / "store"), "--work", str(tmp_path / "w")]) == 0
    assert "disabled_by_policy" in capsys.readouterr().out
    assert cli.main(["run", "--work", str(tmp_path / "w")]) == 2
    assert "not executed" in capsys.readouterr().out
    assert cli.main(["status", "--work", str(tmp_path / "w")]) == 0
