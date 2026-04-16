from __future__ import annotations

from pathlib import Path

from rag_qa.types import Passage
from rag_qa.utils.io import read_jsonl


def load_passages(path: Path) -> list[Passage]:
    """
    从指定的 JSONL 文件中加载所有文本段落数据
    并转换成 Passage 对象列表返回
    :param path: JSONL 文件的路径（Path 类型）
    :return list[Passage]: 包含所有 Passage 实例的列表
    """
    records = read_jsonl(path)
    return [
        Passage(
            id=record["id"],
            title=record.get("title", ""),
            text=record["text"],
            # 把除了 id/title/text 之外的所有字段都放进 metadata 元数据
            metadata={k: v for k, v in record.items() if k not in {"id", "title", "text"}},
        )
        for record in records
    ]

