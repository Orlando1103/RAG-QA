from __future__ import annotations

from pathlib import Path

from rag_qa.config import load_config


def test_load_mvp_config_with_env(monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.setenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")

    cfg = load_config(str(Path("configs/mvp.yaml")))

    assert cfg.project_name == "learn-rag-qa"
    assert cfg.dataset.source == "sample"
    assert cfg.dataset.seed == 42
    assert cfg.generation.model_name == "Qwen/Qwen3-8B"
