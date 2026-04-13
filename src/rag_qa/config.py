from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rag_qa.utils.env import load_env_file


@dataclass(slots=True)
class PathsConfig:
    data_dir: str
    artifacts_dir: str
    cache_dir: str
    prepared_questions_path: str
    corpus_path: str
    dense_index_path: str
    dense_meta_path: str
    runs_dir: str


@dataclass(slots=True)
class DatasetConfig:
    source: str
    split: str
    subset_size: int
    seed: int
    output_path: str


@dataclass(slots=True)
class GenerationConfig:
    model_name: str | None
    model_name_env: str


@dataclass(slots=True)
class ProviderConfig:
    kind: str
    base_url: str
    base_url_env: str | None
    api_key_env: str
    timeout_seconds: int


@dataclass(slots=True)
class AppConfig:
    project_name: str
    seed: int
    paths: PathsConfig
    dataset: DatasetConfig
    generation: GenerationConfig
    provider: ProviderConfig
    project_root: Path

    def resolve_path(self, relative_or_absolute: str) -> Path:
        path = Path(relative_or_absolute)
        return path if path.is_absolute() else self.project_root / path

    def ensure_directories(self) -> None:
        targets = [
            self.resolve_path(self.paths.data_dir),
            self.resolve_path(self.paths.artifacts_dir),
            self.resolve_path(self.paths.cache_dir),
            self.resolve_path(self.paths.runs_dir),
            self.resolve_path(self.paths.prepared_questions_path).parent,
            self.resolve_path(self.paths.dense_index_path).parent,
        ]
        for target in targets:
            target.mkdir(parents=True, exist_ok=True)

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    函数功能：将两个配置文件进行深度合并，并返回一个AppConfig对象。
    :param base: 字典
    :param override: 字典
    :return: 合并后的字典
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: str | Path) -> AppConfig:
    cfg_path = Path(config_path).resolve()
    project_root = cfg_path.parent.parent if cfg_path.parent.name == "configs" else Path.cwd()
    load_env_file(project_root / ".env")

    base_path = project_root / "configs" / "base.yaml"
    config_data = _read_yaml(base_path)
    if cfg_path != base_path:
        config_data = _deep_merge(config_data, _read_yaml(cfg_path))

    provider_base_url_env = config_data["provider"].get("base_url_env") or "MODELSCOPE_BASE_URL"
    generation_model_env = config_data["generation"].get("model_name_env") or "MODELSCOPE_MODEL"
    config_data["provider"]["base_url"] = os.getenv(provider_base_url_env, config_data["provider"]["base_url"])
    config_data["generation"]["model_name"] = os.getenv(
        generation_model_env,
        config_data["generation"].get("model_name"),
    )

    if not config_data["generation"].get("model_name"):
        missing_name = config_data["generation"].get("model_name_env", "MODELSCOPE_MODEL")
        raise ValueError(f"Missing model name. Set `{missing_name}` in .env or environment.")

    settings = AppConfig(
        project_name=config_data["project_name"],
        seed=config_data["seed"],
        paths=PathsConfig(**config_data["paths"]),
        dataset=DatasetConfig(**config_data["dataset"]),
        generation=GenerationConfig(**config_data["generation"]),
        provider=ProviderConfig(**config_data["provider"]),
        project_root=project_root,
    )
    settings.ensure_directories()
    return settings
