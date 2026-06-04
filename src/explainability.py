"""Olasılıksal otomata kararlarını JSON açıklama çıktısına dönüştürür."""

from __future__ import annotations

import json
from typing import Any

from automata_model import ProbabilisticAutomaton

ExplanationRecord = dict[str, Any]


class ExplainabilityEngine:
    def __init__(
        self,
        automaton: ProbabilisticAutomaton,
        anomaly_threshold: float,
    ) -> None:
        self.automaton = automaton
        self.anomaly_threshold = anomaly_threshold

    def _status_and_mapping(self, pattern: str) -> tuple[str, str]:
        if pattern in self.automaton.states:
            return "seen", pattern
        return "unseen", self.automaton.resolve_state(pattern)

    def explain_step(
        self,
        time_step: int,
        previous_state: str | None,
        pattern: str,
        path_probability: float,
    ) -> ExplanationRecord:
        status, mapped_to = self._status_and_mapping(pattern)
        state = previous_state if previous_state is not None else mapped_to
        decision = (
            "anomaly" if path_probability < self.anomaly_threshold else "normal"
        )
        return {
            "time_step": time_step,
            "state": state,
            "pattern": pattern,
            "status": status,
            "mapped_to": mapped_to,
            "probability": float(path_probability),
            "decision": decision,
        }

    def explain_sequence(self, patterns: list[str]) -> list[ExplanationRecord]:
        records: list[ExplanationRecord] = []
        path_prob = 1.0
        prev_mapped: str | None = None

        for t, pattern in enumerate(patterns):
            mapped = self.automaton.resolve_state(pattern)
            if prev_mapped is not None:
                path_prob *= self.automaton.get_transition_probability(
                    prev_mapped, mapped
                )

            records.append(
                self.explain_step(t, prev_mapped, pattern, path_prob)
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
