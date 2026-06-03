import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from automata_model import (
    ProbabilisticAutomaton,
    levenshtein_distance,
    nearest_pattern,
)


def test_levenshtein_one_substitution():
    assert levenshtein_distance("abc", "adc") == 1


def test_levenshtein_same_string_zero():
    assert levenshtein_distance("abc", "abc") == 0


def test_levenshtein_empty_vs_full():
    assert levenshtein_distance("", "abc") == 3


def test_nearest_pattern_maps_unseen():
    vocab = ["aba", "bcb", "cbc"]
    nearest, dist = nearest_pattern("aca", vocab)
    assert nearest == "aba"
    assert dist == 1


def test_automaton_resolve_unseen_state():
    auto = ProbabilisticAutomaton()
    auto.fit_sequence(["aba", "bcb", "cbc"])
    assert auto.resolve_state("aba") == "aba"
    assert auto.resolve_state("aca") == "aba"
