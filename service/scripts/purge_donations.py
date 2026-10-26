"""Project-end deletion of every donated file, consent record and pending copy.

Run inside the service container (it owns the donation volume), with the service stopped:
    docker compose run --rm --no-deps service python scripts/purge_donations.py --confirm DELETE-ALL-DONATIONS
Then remove the host-side donation filesystem image as described in README ("Donation storage").
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from donations.store import DonationStore  # noqa: E402

PHRASE = "DELETE-ALL-DONATIONS"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--directory", default=os.environ.get("DONATION_DIR", "/data/donations"))
    options = parser.parse_args(argv)
    if options.confirm != PHRASE:
        parser.error("--confirm must be " + PHRASE)
    store = DonationStore(options.directory)
    removed = store.purge_all()
    remaining_files = [path for path in store.quarantine.iterdir() if path.is_file()]
    with store.connect() as connection:
        remaining_rows = connection.execute("SELECT COUNT(*) FROM donations").fetchone()[0]
    result = {"removed_records": removed, "remaining_files": len(remaining_files), "remaining_rows": remaining_rows}
    print(json.dumps(result))
    return 0 if not remaining_files and not remaining_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
