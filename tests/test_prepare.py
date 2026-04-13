from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_qa.config import load_config
from rag_qa.data.datasets import prepare_questions


def test_prepare_sample_creates_output(monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_MODEL", "Qwen/Qwen3-8B")
    cfg = load_config("configs/mvp.yaml")

    questions = prepare_questions(cfg, force_refresh=True)
    output_path = cfg.resolve_path(cfg.dataset.output_path)

    assert len(questions) == 4
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    first = json.loads(lines[0])
    assert first["question"] == "Who wrote Pride and Prejudice?"


def test_prepare_unsupported_source_raises(monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_MODEL", "Qwen/Qwen3-8B")
    cfg = load_config("configs/mvp.yaml")
    cfg.dataset.source = "unknown_source"

    with pytest.raises(ValueError, match="Unsupported dataset source"):
        prepare_questions(cfg, force_refresh=True)

