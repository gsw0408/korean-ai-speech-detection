"""Command line for the contributed-data pipeline (no retraining is executed).

Examples:
  python scripts/mlops.py intake --store /data/donations --work /data/mlops
  python scripts/mlops.py sync --store /data/donations --work /data/mlops
  python scripts/mlops.py review --work /data/mlops --id <contribution> --label real --rights-confirmed --reviewer alice
  python scripts/mlops.py build --work /data/mlops --label v1 --seed 1
  python scripts/mlops.py plan --work /data/mlops --dataset <version> --base bundles/nii-small-v1
  python scripts/mlops.py status --work /data/mlops
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from donations.mlops import datasets, intake, review, training  # noqa: E402
from donations.mlops.common import PolicyError  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("intake", "sync"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--store", default=os.environ.get("DONATION_DIR", "/data/donations"))
        cmd.add_argument("--work", required=True)
    cmd = sub.add_parser("review")
    cmd.add_argument("--work", required=True)
    cmd.add_argument("--id", required=True)
    cmd.add_argument("--label", choices=review.LABELS, required=True)
    cmd.add_argument("--rights-confirmed", action="store_true")
    cmd.add_argument("--reviewer", required=True)
    cmd = sub.add_parser("build")
    cmd.add_argument("--work", required=True)
    cmd.add_argument("--label", required=True)
    cmd.add_argument("--seed", type=int, required=True)
    cmd = sub.add_parser("plan")
    cmd.add_argument("--work", required=True)
    cmd.add_argument("--dataset", required=True)
    cmd.add_argument("--base", required=True)
    sub.add_parser("run").add_argument("--work", required=True)
    sub.add_parser("status").add_argument("--work", required=True)
    args = parser.parse_args(argv)
    work = Path(args.work)
    try:
        if args.command == "intake":
            store_audio = os.environ.get("DONATION_STORE_AUDIO", "false") == "true"
            result = intake.intake(args.store, work / "intake", store_audio=store_audio)
        elif args.command == "sync":
            result = intake.sync_withdrawals(args.store, work / "intake", work / "datasets")
        elif args.command == "review":
            result = review.record_review(work / "intake", args.id, label=args.label, rights_confirmed=args.rights_confirmed, reviewer=args.reviewer)
        elif args.command == "build":
            result = datasets.build_version(work / "intake", work / "datasets", label=args.label, seed=args.seed)
        elif args.command == "plan":
            result = training.plan(training.TrainingSpec(base_bundle=args.base, dataset_version=args.dataset), work / "datasets", work / "plans")
        elif args.command == "run":
            training.run()
        else:
            admitted, refused = review.admissible(work / "intake")
            versions = sorted(p.parent.name for p in (work / "datasets").glob("*/manifest.json"))
            result = {"admissible": len(admitted), "refused": refused, "dataset_versions": versions}
    except (PolicyError, KeyError, ValueError) as error:
        print(json.dumps({"refused": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
