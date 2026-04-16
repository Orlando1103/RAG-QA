from __future__ import annotations

import math
from collections import Counter

from rag_qa.types import Passage, RetrievedPassage
from rag_qa.utils.text import tokenize


class BM25Retriever:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75) -> None:
        self.passages = passages
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(f"{passage.title} {passage.text}") for passage in passages] # 存储每个文档（标题 + 文本）token 化后的列表
        self.doc_term_freqs = [Counter(tokens) for tokens in self.doc_tokens] # ，用Counter存储每个文档中各 token 的出现频率（TF）
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens] # ，用列表存储每个文档的长度（DL）
        self.avg_doc_len = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.doc_freqs = self._compute_doc_freqs() # 用Counter存储每个 token 的文档频率（DF）

    def _compute_doc_freqs(self) -> Counter:
        """
        Compute the document frequency of each token.(DF)
        """
        counter: Counter = Counter()
        for terms in self.doc_term_freqs:
            counter.update(terms.keys())
        return counter

    def _idf(self, token: str) -> float:
        """
        Compute the inverse document frequency of a token.(IDF)
        """
        n_docs = len(self.passages)
        doc_freq = self.doc_freqs.get(token, 0)
        return math.log(1 + ((n_docs - doc_freq + 0.5) / (doc_freq + 0.5)))

    def _score(self, query_tokens: list[str], doc_index: int) -> float:
        """
        Compute the BM25 score of a query against a document.
        """
        doc_tf = self.doc_term_freqs[doc_index]
        doc_len = self.doc_lengths[doc_index]
        score = 0.0
        for token in query_tokens:
            freq = doc_tf.get(token, 0)
            if freq == 0:
                continue
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1e-9))
            score += self._idf(token) * (numerator / denominator)
        return score

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
        query_tokens = tokenize(question)
        scored = []
        for idx, passage in enumerate(self.passages):
            score = self._score(query_tokens, idx)
            if score <= 0:
                continue
            scored.append(
                RetrievedPassage(
                    passage=passage,
                    score=score,
                    source="sparse",
                    rank=0,
                    component_scores={"sparse": score},
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return [
            RetrievedPassage(
                passage=item.passage,
                score=item.score,
                source=item.source,
                rank=rank,
                component_scores=item.component_scores,
            )
            for rank, item in enumerate(scored[:top_k], start=1)
        ]

