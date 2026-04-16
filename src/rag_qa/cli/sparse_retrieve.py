from __future__ import annotations

import argparse
import json

from rag_qa.config import load_config
from rag_qa.data.corpus import load_passages
from rag_qa.retrieval.sparse import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sparse BM25 retrieval for a question.")
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    passages = load_passages(config.resolve_path(config.paths.corpus_path))
    retriever = BM25Retriever(passages=passages, k1=config.sparse.k1, b=config.sparse.b)
    results = retriever.retrieve(question=args.question, top_k=args.top_k)
    payload = [
        {
            "rank": item.rank,
            "id": item.passage.id,
            "title": item.passage.title,
            "score": item.score,
        }
        for item in results
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

