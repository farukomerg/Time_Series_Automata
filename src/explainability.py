"""Olas─▒l─▒ksal otomata kararlar─▒n─▒ JSON a├ğ─▒klama ├ğ─▒kt─▒s─▒na d├Ân├╝┼şt├╝r├╝r."""

from __future__ import annotations

import json
from typing import Any

from automata_model import ProbabilisticAutomaton, nearest_pattern

ExplanationRecord = dict[str, Any]


class ExplainabilityEngine:
    def __init__(
        self,
        automaton: ProbabilisticAutomaton,
        training_vocabulary: set[str],
        anomaly_threshold: float,
        window_size: int = 4,
    ) -> None:
        self.automaton = automaton
        self.training_vocabulary = training_vocabulary
        self.anomaly_threshold = anomaly_threshold
        self.window_size = window_size

    def _status_and_mapping(self, pattern: str) -> tuple[str, str, int]:
        if pattern in self.training_vocabulary:
            return "seen", pattern, 0
        mapped, distance = nearest_pattern(pattern, self.training_vocabulary)
        return "unseen", mapped, distance

    def _outbound_transitions(self, from_state: str) -> list[str]:
        probs = self.automaton.transition_probabilities.get(from_state, {})
        return [
            f"{from_state} -> {dst}: {self.automaton.get_transition_probability(from_state, dst):.2f}"
            for dst, _ in sorted(probs.items())
        ]

    def explain_step(
        self,
        time_step: int,
        previous_state: str | None,
        pattern: str,
        path_probability: float,
        path_transitions: list[str],
    ) -> ExplanationRecord:
        status, mapped_to, distance = self._status_and_mapping(pattern)
        state = previous_state if previous_state is not None else mapped_to
        decision = (
            "anomaly" if path_probability < self.anomaly_threshold else "normal"
        )
        current_state = mapped_to
        transitions = self._outbound_transitions(current_state)
        if path_transitions:
            transitions = path_transitions + transitions
        return {
            "time_step": time_step,
            "state": state,
            "pattern": pattern,
            "status": status,
            "mapped_to": mapped_to,
            "distance": distance,
            "probability": float(path_probability),
            "confidence_score": float(path_probability),
            "transitions": transitions,
            "decision": decision,
        }

    def explain_sequence(self, patterns: list[str]) -> list[ExplanationRecord]:
        records: list[ExplanationRecord] = []
        mapped_states: list[str] = []
        step_probabilities: list[float] = []

        for pattern in patterns:
            _, mapped, _ = self._status_and_mapping(pattern)
            mapped_states.append(mapped)

        for t, pattern in enumerate(patterns):
            if t > 0:
                prev_mapped = mapped_states[t - 1]
                curr_mapped = mapped_states[t]
                step_p = self.automaton.get_transition_probability(prev_mapped, curr_mapped)
                step_probabilities.append(step_p)

            start_trans_idx = max(0, t - self.window_size + 1)
            active_transitions = step_probabilities[start_trans_idx : t]

            path_prob = 1.0
            for p in active_transitions:
                path_prob *= p

            path_transitions: list[str] = []
            for i in range(start_trans_idx, t):
                prev = mapped_states[i]
                curr = mapped_states[i + 1]
                p_val = step_probabilities[i]
                path_transitions.append(f"{prev} -> {curr}: {p_val:.2f}")

            records.append(
                self.explain_step(
                    t,
                    mapped_states[t - 1] if t > 0 else None,
                    pattern,
                    path_prob,
                    path_transitions,
                )
            )

        return records

    def to_json(self, records: list[ExplanationRecord], indent: int = 2) -> str:
        return json.dumps(records, indent=indent, ensure_ascii=False)

    def save_json(
        self,
        patterns: list[str],
        filepath: str,
        indent: int = 2,
    ) -> list[ExplanationRecord]:
        records = self.explain_sequence(patterns)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json(records, indent=indent))
        return records
