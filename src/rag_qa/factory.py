from __future__ import annotations

from pathlib import Path

from rag_qa.config import AppConfig
from rag_qa.data.corpus import load_passages
from rag_qa.retrieval.dense import DenseRetriever, HashingEmbedder


def save_dense_index(config: AppConfig, use_hashing_embedder: bool = False) -> tuple[Path, Path]:
    """
    深度检索工厂：HashingEmbedder()、dense.build()、index.save
    :param config:
    :param use_hashing_embedder:
    :return:
    """
    passages = load_passages(config.resolve_path(config.paths.corpus_path)) #corpus中方法，读取段落jsonl文件
    dense = DenseRetriever(
        passages,
        config.dense,
        embedder=HashingEmbedder() if use_hashing_embedder else None,
    )
    index = dense.build()
    index_path = config.resolve_path(config.paths.dense_index_path)
    meta_path = config.resolve_path(config.paths.dense_meta_path)
    index.save(index_path, meta_path)
    return index_path, meta_path

