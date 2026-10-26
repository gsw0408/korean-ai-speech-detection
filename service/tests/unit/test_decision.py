import pytest

from pipeline.decision import classify, decide
from pipeline.frozen.audio_windows import starts
from pipeline.preprocess import PreparedAudio


@pytest.mark.parametrize("length,expected", [(64000, [0]), (64001, [0, 1]),
                                          (96000, [0, 32000]), (96001, [0, 32000, 32001])])
def test_tail_coverage(length, expected):
    assert starts(length) == expected
    assert expected[-1] + 64000 == length


def test_short_window_rejected():
    with pytest.raises(ValueError):
        starts(63999)


@pytest.mark.parametrize("score,expected", [(1, "warn"), (-1, "low"), (0, "defer"),
                                          (None, "cannot"), (float("nan"), "cannot")])
def test_boundaries(score, expected):
    assert classify(score, -1, 1) == expected


def test_original_clock_slots_max_and_uncovered_edges():
    prepared = PreparedAudio([], [0, 32000], 10, 32000, 128000, 6, 16000)
    result = decide(prepared, [-2, 2], -1, 1)
    assert result["file_score"] == 0
    assert result["outcome_code"] == "defer"
    assert [(w["start"], w["end"]) for w in result["windows"]] == [(2, 6), (4, 8)]
    assert [s["score"] for s in result["slots"]] == [None, -2, 2, 2, None]
    assert all(s["indicative_only"] for s in result["slots"])


def test_partial_slot():
    prepared = PreparedAudio([], [0], 4.5, 8000, 72000, 4, 16000)
    result = decide(prepared, [0], -1, 1)
    assert result["windows"][0]["start"] == 0.5
    assert result["slots"][-1]["end"] == 4.5


@pytest.mark.parametrize("scores", [[], [0, 1], [float("nan")]])
def test_invalid_scores(scores):
    with pytest.raises(ValueError):
        decide(PreparedAudio([], [0], 4, 0, 64000, 4, 16000), scores, -1, 1)
