from __future__ import annotations

from rag_qa.config import load_config
from rag_qa.data.corpus import load_passages
from rag_qa.factory import save_dense_index
from rag_qa.retrieval.dense import DenseIndex, DenseRetriever, HashingEmbedder


def test_dense_index_build_and_save(monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_MODEL", "Qwen/Qwen3-8B")
    cfg = load_config("configs/mvp.yaml")

    index_path, meta_path = save_dense_index(cfg, use_hashing_embedder=True)

    assert index_path.exists()
    assert meta_path.exists()


def test_dense_retrieve_with_hashing_embedder(monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_MODEL", "Qwen/Qwen3-8B")
    cfg = load_config("configs/mvp.yaml")
    passages = load_passages(cfg.resolve_path(cfg.paths.corpus_path))
    retriever = DenseRetriever(passages, cfg.dense, embedder=HashingEmbedder())
    retriever.build()

    results = retriever.retrieve("Who created Python?", top_k=3)

    assert results
    assert len(results) <= 3
    assert results[0].rank == 1


def test_dense_load_then_retrieve(monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_MODEL", "Qwen/Qwen3-8B")
    cfg = load_config("configs/mvp.yaml")
    index_path, meta_path = save_dense_index(cfg, use_hashing_embedder=True)
    loaded = DenseIndex.load(index_path, meta_path)

    assert loaded.embeddings.shape[0] == len(loaded.passages)
    assert loaded.embeddings.shape[1] > 0

