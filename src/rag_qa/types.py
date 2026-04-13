from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QAExample:
    id: str
    question: str
    answers: list[str]

