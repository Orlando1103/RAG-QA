from __future__ import annotations

import re
import string


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in string.punctuation)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def tokenize(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return [token for token in normalized.split() if token]

