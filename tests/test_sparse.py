from __future__ import annotations

from rag_qa.config import load_config
from rag_qa.data.corpus import load_passages
from rag_qa.data.datasets import load_local_questions
from rag_qa.retrieval.sparse import BM25Retriever
from rag_qa.utils.text import normalize_answer


def _is_supported(answers: list[str], retrieved_texts: list[str]) -> bool:
    normalized_retrieved = [normalize_answer(text) for text in retrieved_texts]
    for answer in answers:
        target = normalize_answer(answer)
        if any(target in text for text in normalized_retrieved):
            return True
    return False


def test_bm25_retrieve_known_question(monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_MODEL", "Qwen/Qwen3-8B")
    cfg = load_config("configs/mvp.yaml")
    passages = load_passages(cfg.resolve_path(cfg.paths.corpus_path))
    retriever = BM25Retriever(passages, k1=cfg.sparse.k1, b=cfg.sparse.b)

    results = retriever.retrieve("Who wrote Pride and Prejudice?", top_k=3)

    assert results
    assert results[0].rank == 1
    assert any(item.passage.id == "p1" for item in results)


def test_bm25_handles_empty_and_long_query(monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_MODEL", "Qwen/Qwen3-8B")
    cfg = load_config("configs/mvp.yaml")
    passages = load_passages(cfg.resolve_path(cfg.paths.corpus_path))
    retriever = BM25Retriever(passages, k1=cfg.sparse.k1, b=cfg.sparse.b)

    empty_results = retriever.retrieve("", top_k=3)
    long_results = retriever.retrieve("python " * 200, top_k=3)

    assert empty_results == []
    assert len(long_results) <= 3


def test_bm25_recall_at_k_acceptance(monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_MODEL", "Qwen/Qwen3-8B")
    cfg = load_config("configs/mvp.yaml")
    passages = load_passages(cfg.resolve_path(cfg.paths.corpus_path))
    questions = load_local_questions(cfg.project_root / "data" / "sample" / "questions.jsonl")
    retriever = BM25Retriever(passages, k1=cfg.sparse.k1, b=cfg.sparse.b)

    hits = 0
    for question in questions:
        retrieved = retriever.retrieve(question.question, top_k=3)
        texts = [f"{item.passage.title} {item.passage.text}" for item in retrieved]
        if _is_supported(question.answers, texts):
            hits += 1

    recall_at_3 = hits / len(questions)
    assert recall_at_3 >= 0.75

