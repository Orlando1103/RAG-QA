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
class RetrievalConfig:
    initial_top_k: int
    second_pass_top_k: int


@dataclass(slots=True)
class SparseConfig:
    enabled: bool
    k1: float
    b: float


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
    retrieval: RetrievalConfig
    sparse: SparseConfig
    generation: GenerationConfig
    provider: ProviderConfig
    project_root: Path

    def resolve_path(self, relative_or_absolute: str) -> Path:
        """
        统一处理相对路径 / 绝对路径：传入路径是绝对路径就直接用，是相对路径就自动拼接项目根目录，返回标准路径对象。
        :param 支持相对 / 绝对路径字符串
        :return:标准化的 Path 路径对象
        [知识点]：path对象
        """
        path = Path(relative_or_absolute)
        return path if path.is_absolute() else self.project_root / path
    # 创建目录
    def ensure_directories(self) -> None:
        """
        自动创建项目需要的所有文件夹：把数据、缓存、运行结果等目录统一检查，不存在就自动创建，已存在也不报错。
        """
        targets = [
            self.resolve_path(self.paths.data_dir),
            self.resolve_path(self.paths.artifacts_dir),
            self.resolve_path(self.paths.cache_dir),
            self.resolve_path(self.paths.runs_dir),
            self.resolve_path(self.paths.prepared_questions_path).parent,
            self.resolve_path(self.paths.dense_index_path).parent,
        ]
        for target in targets:
            # parents=True：自动创建多级父目录
            # exist_ok=True：目录已存在时不报错
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
    """
       从指定路径读取 YAML 文件，并转换为 Python 字典
       :param path: YAML 文件的 Path 对象
       :return: 解析后的字典（空文件返回 {}）
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: str | Path) -> AppConfig:
    """
    这段代码是项目启动时加载配置文件的逻辑：
    1.找到项目根目录
    2.加载环境变量（.env）
    3.读取基础配置 + 你的自定义配置，并自动合并
    4.用环境变量覆盖配置里的关键参数（模型地址、模型名称）
    5.最后检查必须的模型名称是否存在，不存在就报错
    :param config_path:
    :return:
    """
    #联合类型（Union Type）：这个参数 config_path 可以是 字符串 或者 Path 对象 两种类型都行。3.10+新增语法
    cfg_path = Path(config_path).resolve()
    project_root = cfg_path.parent.parent if cfg_path.parent.name == "configs" else Path.cwd() #cwd是当前目录
    load_env_file(project_root / ".env")

    base_path = project_root / "configs" / "base.yaml"
    config_data = _read_yaml(base_path)
    if cfg_path != base_path:
        config_data = _deep_merge(config_data, _read_yaml(cfg_path)) #将基础配置和自定义配置进行深度合并

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
        retrieval=RetrievalConfig(**config_data["retrieval"]),
        sparse=SparseConfig(**config_data["sparse"]),
        generation=GenerationConfig(**config_data["generation"]),
        provider=ProviderConfig(**config_data["provider"]),
        project_root=project_root,
    )
    settings.ensure_directories()  #AppConfig创建目录
    return settings
