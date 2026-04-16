from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QAExample:
    id: str
    question: str
    answers: list[str]


@dataclass(slots=True)
class Passage:
    id: str
    title: str
    text: str
    metadata: dict


@dataclass(slots=True)
class RetrievedPassage:
    passage: Passage
    score: float
    source: str
    rank: int
    component_scores: dict[str, float]

