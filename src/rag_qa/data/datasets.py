from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from rag_qa.config import AppConfig
from rag_qa.types import QAExample
from rag_qa.utils.io import read_jsonl, write_jsonl


def _records_to_examples(records: Iterable[dict]) -> list[QAExample]:
    examples: list[QAExample] = []
    #enumerate 同时拿到（idx）+ 内容（record）
    for idx, record in enumerate(records):
        answers = record.get("answers") or record.get("answer") or []
        if isinstance(answers, str):
            answers = [answers] #如果 answer 是字符串（如 "yes"），自动转成列表 ["yes"]
        examples.append(
            QAExample(
                # f-string 格式化字符串，Python 里用来把变量塞进字符串里。生成一个字符串 = q + 当前数字编号
                id=str(record.get("id", f"q{idx}")),
                question=record["question"],
                answers=list(answers), #强制类型转换
            )
        )
    return examples


def load_local_questions(path: Path) -> list[QAExample]:
    """
    read_jsonl 是一个读取 .jsonl 文件的函数。.jsonl文件就是每一行都是JSON的文件
    作用：打开文件 → 逐行读取 → 转成字典列表返回
    :param path:
    :return:QAExample类的列表
    """
    return _records_to_examples(read_jsonl(path))


def save_questions(path: Path, questions: list[QAExample]) -> None:
    write_jsonl(
        path,
        [{"id": q.id, "question": q.question, "answers": q.answers} for q in questions],
    )


def write_prepare_report(config: AppConfig, questions: list[QAExample], source_used: str) -> Path:
    report_path = (
        config.resolve_path(config.paths.runs_dir)
        / f"{Path(config.dataset.output_path).stem}_prepare_report.json"
    )
    payload = {
        "source": source_used,
        "split": config.dataset.split,
        "subset_size": len(questions),
        "output_path": str(config.resolve_path(config.dataset.output_path)),
        "question_ids": [q.id for q in questions[:10]],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def load_nq_open_subset(config: AppConfig) -> list[QAExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The `datasets` package is required to download NQ Open.") from exc

    dataset = load_dataset(
        "nq_open",
        split=config.dataset.split,
        cache_dir=str(config.resolve_path(config.paths.cache_dir)),
    )
    records = list(dataset)
    random.Random(config.dataset.seed).shuffle(records)
    subset = records[: config.dataset.subset_size]
    normalized = [
        {
            "id": str(record.get("id", record["question"])),
            "question": record["question"],
            "answers": record.get("answer", []),
        }
        for record in subset
    ]
    return _records_to_examples(normalized)


def prepare_questions(config: AppConfig, force_refresh: bool = False) -> list[QAExample]:
    output_path = config.resolve_path(config.dataset.output_path)
    # 如果输出文件已经存在，并且不强制刷新，则直接返回
    if output_path.exists() and not force_refresh:
        return load_local_questions(output_path)

    sample_path = config.project_root / "data" / "sample" / "questions.jsonl"
    source = config.dataset.source.lower() # 将字符串转换为小写
    if source == "sample":
        #列表切片语法，作用：只取列表的前 N 个元素
        questions = load_local_questions(sample_path)[: config.dataset.subset_size]
        save_questions(output_path, questions)
        write_prepare_report(config, questions, source_used="sample")
        return questions

    if source != "nq_open":
        raise ValueError(f"Unsupported dataset source: {config.dataset.source}")

    questions = load_nq_open_subset(config)
    save_questions(output_path, questions)
    write_prepare_report(config, questions, source_used="nq_open")
    return questions

