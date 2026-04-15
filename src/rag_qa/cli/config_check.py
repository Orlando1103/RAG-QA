from __future__ import annotations

import argparse

from src.rag_qa.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether phase-1 config loads correctly.")
    parser.add_argument("--config", required=True, help="Path to yaml config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print("Config loaded successfully.")
    print(f"project_name={cfg.project_name}")
    print(f"dataset.source={cfg.dataset.source}")
    print(f"generation.model_name={cfg.generation.model_name}")
    print(f"provider.base_url={cfg.provider.base_url}")


if __name__ == "__main__":
    main()
