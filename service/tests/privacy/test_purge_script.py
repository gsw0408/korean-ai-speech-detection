import json

import pytest

from donations.store import DonationStore
from scripts.purge_donations import main


def test_purge_requires_exact_phrase(tmp_path):
    with pytest.raises(SystemExit):
        main(["--confirm", "yes", "--directory", str(tmp_path)])


def test_purge_removes_all_records_and_files(tmp_path, capsys):
    store = DonationStore(tmp_path)
    for body in (b"a" * 10, b"b" * 10):
        store.donate(store.offer(body, "low", "bundle"))
    assert main(["--confirm", "DELETE-ALL-DONATIONS", "--directory", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"removed_records": 2, "remaining_files": 0, "remaining_rows": 0}
    assert not list(store.quarantine.iterdir())
