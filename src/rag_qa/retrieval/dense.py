from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_qa.config import DenseConfig
from rag_qa.types import Passage, RetrievedPassage
from rag_qa.utils.io import write_jsonl


class BaseEmbedder:
    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, normalize: bool) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("sentence-transformers is required for dense retrieval.") from exc
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


class HashingEmbedder(BaseEmbedder):
    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:  # noqa: ARG002
        rows = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token in text.lower().split():
                rows[row_idx, hash(token) % self.dimension] += 1.0
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return rows / norms


@dataclass(slots=True)
class DenseIndex:
    passages: list[Passage]
    embeddings: np.ndarray

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        #-1 的意思：自动计算剩下的维度，不用手动写数字，转换成（1,x)的二维数组
        #因为 矩阵乘法、faiss 搜索都必须用二维数组！
        query = query_embedding.reshape(1, -1)
        try:
            import faiss  # type: ignore
        except ImportError:
            scores = np.dot(self.embeddings, query.T).reshape(-1)#拉平成1维向量
            #np.argsort(scores)把分数从小到大排序，返回索引；[::-1]反转顺序 ；取topk
            order = np.argsort(scores)[::-1][:top_k]
            #tuple（元组）就是 Python 里的「不可变列表」，有序、可重复、可存放任意类型数据
            return [(int(idx), float(scores[idx])) for idx in order]

        index = faiss.IndexFlatIP(self.embeddings.shape[1])
        index.add(self.embeddings)
        scores, indices = index.search(query.astype(np.float32), top_k)
        return [
            (int(doc_idx), float(score))
            for doc_idx, score in zip(indices[0].tolist(), scores[0].tolist(), strict=False)
            if doc_idx >= 0
        ]

    def save(self, index_path: Path, meta_path: Path) -> None:
        """
        保存索引和元数据：向量数据（embeddings） → 用 numpy 压缩保存
                      文本元数据（id、标题、正文、其他信息） → 用 JSON Lines 保存
        :param index_path:保存向量的文件路径
        :param meta_path:保存文本信息的文件路径
        """
        index_path.parent.mkdir(parents=True, exist_ok=True)
        #np.savez_compressed 是 NumPy 提供的、保存压缩数组的官方函数，把numpy 数组（向量矩阵） 以高压缩率、二进制格式保存到文件
        np.savez_compressed(index_path, embeddings=self.embeddings)
        write_jsonl(
            meta_path,
            [
                {
                    "id": passage.id,
                    "title": passage.title,
                    "text": passage.text,
                    **passage.metadata,
                }
                for passage in self.passages
            ],
        )
    #这是一个 “类方法” —— 不需要先创建对象，直接用 类名。方法 () 就能调用
    #因为 load 的目的就是创建并返回一个新对象，所以必须用 @classmethod，这是 Python 里从文件加载对象的标准写法
    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> "DenseIndex":
        """
        加载索引和元数据：用 numpy 读取向量数据（embeddings） ；用 JSON Lines 读取文本信息（id、标题、正文、其他信息）
        :param index_path:保存向量的文件路径
        :param meta_path:保存文本信息的文件路径
        :return:DenseIndex类
        """
        #1.打开 .npz 压缩文件 2.自动解压 3.返回一个字典 - like 对象，里面存着你保存的所有数组
        arrays = np.load(index_path)
        with meta_path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        passages = [
            Passage(
                id=record["id"],
                title=record.get("title", ""),
                text=record["text"],
                metadata={k: v for k, v in record.items() if k not in {"id", "title", "text"}},
            )
            for record in records
        ]
        #self = 当前已经创建好的对象；cls= 当前的类本身
        #cls 就是帮你创建并返回一个新的 DenseIndex 对象。
        return cls(passages=passages, embeddings=arrays["embeddings"])


class DenseRetriever:
    #| 是Python 3.10+ 新增的类型注解语法：联合类型（Union Type），可以是A也可以是B
    def __init__(self, passages: list[Passage], config: DenseConfig, embedder: BaseEmbedder | None = None) -> None:
        self.passages = passages
        self.config = config
        self.embedder = embedder or SentenceTransformerEmbedder(config.model_name, config.normalize)
        self.index: DenseIndex | None = None

    def build(self) -> DenseIndex:
        #将文档批量处理成向量并创建索引DenseIndex
        texts = [f"{passage.title}\n{passage.text}" for passage in self.passages]
        embeddings = self.embedder.encode(texts, batch_size=self.config.batch_size)
        self.index = DenseIndex(passages=self.passages, embeddings=embeddings)
        return self.index

    def load_or_build(self, index_path: Path | None = None, meta_path: Path | None = None) -> DenseIndex:
        #加载索引和元数据，如果索引和元数据都存在，则直接加载，否则就创建并保存
        if self.index is not None:
            return self.index
        if index_path and meta_path and index_path.exists() and meta_path.exists():
            self.index = DenseIndex.load(index_path, meta_path)
            return self.index
        return self.build()

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
        if self.index is None:
            self.build()
        assert self.index is not None
        query_embedding = self.embedder.encode([question], batch_size=1)[0]
        results = self.index.search(query_embedding, top_k=top_k)
        return [
            RetrievedPassage(
                passage=self.index.passages[doc_idx],
                score=score,
                source="dense",
                rank=rank,
                component_scores={"dense": score},
            )
            for rank, (doc_idx, score) in enumerate(results, start=1)
        ]

