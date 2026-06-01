"""
pytest 配置和共享 fixtures。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 添加项目根目录到 Python 路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_csv_path() -> str:
    """返回示例 CSV 文件路径。"""
    return str(ROOT / "data" / "titanic.csv")


@pytest.fixture
def non_existent_csv_path() -> str:
    """返回不存在的 CSV 文件路径。"""
    return str(ROOT / "data" / "non_existent.csv")


@pytest.fixture
def target_column() -> str:
    """返回目标列名。"""
    return "Survived"


@pytest.fixture
def invalid_target_column() -> str:
    """返回无效的目标列名。"""
    return "NonExistentColumn"


@pytest.fixture
def tmp_csv_file(tmp_path: Path) -> Path:
    """创建临时 CSV 文件。"""
    csv_content = """id,name,age,survived
1,Alice,25,1
2,Bob,30,0
3,Charlie,35,1
4,Diana,28,0
5,Eve,32,1
"""
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    return csv_file


@pytest.fixture
def empty_csv_file(tmp_path: Path) -> Path:
    """创建空 CSV 文件。"""
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("", encoding="utf-8")
    return csv_file


@pytest.fixture
def large_csv_file(tmp_path: Path) -> Path:
    """创建较大的 CSV 文件用于测试。"""
    lines = ["id,feature1,feature2,target"]
    for i in range(1000):
        lines.append(f"{i},{i*0.1},{i*0.2},{i % 2}")
    csv_file = tmp_path / "large_data.csv"
    csv_file.write_text("\n".join(lines), encoding="utf-8")
    return csv_file


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """模拟环境变量。"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
