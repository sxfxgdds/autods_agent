"""
数据工具单元测试。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.data_tools import _to_json_serializable, audit_dataset, profile_dataset


class TestToJsonSerializable:
    """_to_json_serializable 函数测试。"""

    def test_none(self) -> None:
        """测试 None 值。"""
        assert _to_json_serializable(None) is None

    def test_integer(self) -> None:
        """测试整数。"""
        assert _to_json_serializable(42) == 42

    def test_float(self) -> None:
        """测试浮点数。"""
        assert _to_json_serializable(3.14) == 3.14

    def test_string(self) -> None:
        """测试字符串。"""
        assert _to_json_serializable("hello") == "hello"

    def test_list(self) -> None:
        """测试列表。"""
        assert _to_json_serializable([1, 2, 3]) == [1, 2, 3]

    def test_dict(self) -> None:
        """测试字典。"""
        assert _to_json_serializable({"a": 1}) == {"a": 1}

    def test_nested_structure(self) -> None:
        """测试嵌套结构。"""
        data = {"list": [1, {"nested": True}], "string": "test"}
        result = _to_json_serializable(data)
        assert result == data


class TestProfileDataset:
    """profile_dataset 函数测试。"""

    @pytest.mark.asyncio
    async def test_valid_dataset(self, sample_csv_path: str, target_column: str) -> None:
        """测试有效数据集。"""
        result = await profile_dataset(sample_csv_path, target_column)
        parsed = json.loads(result)
        assert "error" not in parsed
        assert "n_rows" in parsed
        assert "n_cols" in parsed
        assert "target" in parsed

    @pytest.mark.asyncio
    async def test_invalid_csv_path(self, non_existent_csv_path: str, target_column: str) -> None:
        """测试无效 CSV 路径。"""
        result = await profile_dataset(non_existent_csv_path, target_column)
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_invalid_target_column(self, sample_csv_path: str, invalid_target_column: str) -> None:
        """测试无效目标列。"""
        result = await profile_dataset(sample_csv_path, invalid_target_column)
        parsed = json.loads(result)
        assert "error" in parsed


class TestAuditDataset:
    """audit_dataset 函数测试。"""

    @pytest.mark.asyncio
    async def test_valid_dataset(self, sample_csv_path: str, target_column: str) -> None:
        """测试有效数据集。"""
        result = await audit_dataset(sample_csv_path, target_column)
        parsed = json.loads(result)
        assert "error" not in parsed
        assert "issues" in parsed
        assert "suggestions" in parsed

    @pytest.mark.asyncio
    async def test_invalid_csv_path(self, non_existent_csv_path: str, target_column: str) -> None:
        """测试无效 CSV 路径。"""
        result = await audit_dataset(non_existent_csv_path, target_column)
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_invalid_target_column(self, sample_csv_path: str, invalid_target_column: str) -> None:
        """测试无效目标列。"""
        result = await audit_dataset(sample_csv_path, invalid_target_column)
        parsed = json.loads(result)
        assert "error" in parsed
