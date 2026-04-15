from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path) -> None:
    """
    读取env文件内容并添加到环境变量
    :param path:  .env文件路径
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip() # strip去掉字符串【两头】的空白字符
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        # setdefault(key, value)设置系统环境变量，如果key不存在则设置，如果key存在则不设置
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

