from __future__ import annotations

import argparse

from rag_qa.config import load_config
from rag_qa.factory import save_dense_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dense retrieval index.")
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--use-hashing-embedder", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    index_path, meta_path = save_dense_index(config, use_hashing_embedder=args.use_hashing_embedder)
    print(f"Dense index written to {index_path}")
    print(f"Dense metadata written to {meta_path}")


if __name__ == "__main__":
    main()

