import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from automata_model import ProbabilisticAutomaton, build_transition_matrix
from explainability import ExplainabilityEngine


def test_laplace_smoothing():
    # Fit with a sequence of states
    auto = ProbabilisticAutomaton(smoothing_k=1.0)
    # 3 states: "aba", "bcb", "cbc"
    auto.fit_sequence(["aba", "bcb", "cbc"])
    
    # Check that transition from "aba" to "cbc" (not seen) is non-zero
    p_unseen = auto.get_transition_probability("aba", "cbc")
    assert p_unseen > 0.0
    
    # Check that probabilities from "aba" to all states sum to 1.0
    # States are: "aba", "bcb", "cbc"
    p_sum = sum(auto.get_transition_probability("aba", s) for s in auto.states)
    assert abs(p_sum - 1.0) < 1e-6


def test_sliding_window_probability():
    auto = ProbabilisticAutomaton(smoothing_k=0.0) # disable smoothing for easy math
    # Train sequence: aba -> bcb -> cbc -> aba
    auto.fit_sequence(["aba", "bcb", "cbc", "aba"])
    
    # Transitions learned:
    # aba -> bcb (prob 1.0)
    # bcb -> cbc (prob 1.0)
    # cbc -> aba (prob 1.0)
    
    # Let's test sliding window explainability
    engine = ExplainabilityEngine(
        automaton=auto,
        training_vocabulary={"aba", "bcb", "cbc"},
        anomaly_threshold=0.01,
        window_size=3 # window size of 3 means last 2 transitions are multiplied
    )
    
    # Test sequence: ["aba", "bcb", "cbc", "bcb"]
    # Transitions:
    # t=0: ["aba"], path_prob = 1.0
    # t=1: ["aba", "bcb"], transition: aba -> bcb (p=1.0). path_prob = 1.0
    # t=2: ["aba", "bcb", "cbc"], transitions: aba -> bcb (p=1.0), bcb -> cbc (p=1.0). path_prob = 1.0
    # t=3: ["bcb", "cbc", "bcb"], transitions: bcb -> cbc (p=1.0), cbc -> bcb (p=0.0). path_prob = 0.0
    explanations = engine.explain_sequence(["aba", "bcb", "cbc", "bcb"])
    
    assert len(explanations) == 4
    assert explanations[0]["probability"] == 1.0
    assert explanations[1]["probability"] == 1.0
    assert explanations[2]["probability"] == 1.0
    assert explanations[3]["probability"] == 0.0
    assert explanations[3]["decision"] == "anomaly"
