"""Parity selection uses synthetic identities, with balanced independent cases."""

import pytest

from scripts.partial_parity import select_composites


def test_selection_is_deterministic_balanced_and_unique():
    records = [dict(id=f"fixture-{kind}-{index:03}", kind=kind)
               for kind in ("real", "attack") for index in range(67)]
    selected = select_composites(records)
    assert selected == select_composites(list(reversed(records)))
    assert len(selected) == len({record["id"] for record in selected}) == 40
    assert sum(record["kind"] == "real" for record in selected) == 20
    assert sum(record["kind"] == "attack" for record in selected) == 20
    assert {record["id"] for record in selected if record["kind"] == "real"} == {
        f"fixture-real-{index:03}" for index in range(0, 60, 3)}


def test_selection_rejects_insufficient_or_unbalanced_input():
    with pytest.raises(ValueError, match="Insufficient"):
        select_composites([dict(id=f"fixture-{index}", kind="real") for index in range(100)])
    with pytest.raises(ValueError, match="positive even"):
        select_composites([], count=3)
