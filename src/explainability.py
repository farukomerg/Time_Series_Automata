"""Olasılıksal otomata kararlarını JSON açıklama çıktısına dönüştürür."""

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
    ) -> None:
        self.automaton = automaton
        self.training_vocabulary = training_vocabulary
        self.anomaly_threshold = anomaly_threshold

    def _status_and_mapping(self, pattern: str) -> tuple[str, str, int]:
        if pattern in self.training_vocabulary:
            return "seen", pattern, 0
        mapped, distance = nearest_pattern(pattern, self.training_vocabulary)
        return "unseen", mapped, distance

    def _outbound_transitions(self, from_state: str) -> list[str]:
        probs = self.automaton.transition_probabilities.get(from_state, {})
        return [
            f"{from_state} -> {dst}: {p:.2f}"
            for dst, p in sorted(probs.items())
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
        path_prob = 1.0
        prev_mapped: str | None = None
        path_transitions: list[str] = []

        for t, pattern in enumerate(patterns):
            _, mapped, _ = self._status_and_mapping(pattern)
            if prev_mapped is not None:
                step_p = self.automaton.get_transition_probability(prev_mapped, mapped)
                path_prob *= step_p
                path_transitions.append(
                    f"{prev_mapped} -> {mapped}: {step_p:.2f}"
                )

            records.append(
                self.explain_step(
                    t, prev_mapped, pattern, path_prob, list(path_transitions)
                )
            )
            prev_mapped = mapped

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
