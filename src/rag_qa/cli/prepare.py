from __future__ import annotations

import argparse

from rag_qa.config import load_config
from rag_qa.data.datasets import prepare_questions, write_prepare_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sample or NQ Open question subsets.")
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    questions = prepare_questions(config, force_refresh=args.force_refresh)
    report_path = write_prepare_report(config, questions, source_used=config.dataset.source)
    print(f"Prepared {len(questions)} questions at {config.resolve_path(config.dataset.output_path)}")
    print(f"Preparation report written to {report_path}")


if __name__ == "__main__":
    main()

