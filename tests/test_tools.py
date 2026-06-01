"""
工具函数单元测试。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from tools.utils import (
    build_error_response,
    build_success_response,
    get_feature_types,
    safe_read_csv,
    validate_csv_path,
    validate_dataframe,
    validate_target_column,
)


class TestValidateCsvPath:
    """validate_csv_path 函数测试。"""

    def test_valid_csv_path(self, sample_csv_path: str) -> None:
        """测试有效 CSV 路径。"""
        is_valid, error_msg = validate_csv_path(sample_csv_path)
        assert is_valid is True
        assert error_msg is None

    def test_empty_path(self) -> None:
        """测试空路径。"""
        is_valid, error_msg = validate_csv_path("")
        assert is_valid is False
        assert "不能为空" in error_msg

    def test_non_existent_path(self, non_existent_csv_path: str) -> None:
        """测试不存在的路径。"""
        is_valid, error_msg = validate_csv_path(non_existent_csv_path)
        assert is_valid is False
        assert "不存在" in error_msg

    def test_not_a_file(self, tmp_path: Path) -> None:
        """测试路径不是文件。"""
        is_valid, error_msg = validate_csv_path(str(tmp_path))
        assert is_valid is False
        assert "不是文件" in error_msg

    def test_not_csv_extension(self, tmp_path: Path) -> None:
        """测试文件扩展名不是 CSV。"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test", encoding="utf-8")
        is_valid, error_msg = validate_csv_path(str(txt_file))
        assert is_valid is False
        assert "不是 CSV 格式" in error_msg


class TestValidateTargetColumn:
    """validate_target_column 函数测试。"""

    def test_valid_target_column(self, sample_csv_path: str, target_column: str) -> None:
        """测试有效目标列。"""
        df = pd.read_csv(sample_csv_path)
        is_valid, error_msg = validate_target_column(df, target_column)
        assert is_valid is True
        assert error_msg is None

    def test_empty_target_column(self, sample_csv_path: str) -> None:
        """测试空目标列名。"""
        df = pd.read_csv(sample_csv_path)
        is_valid, error_msg = validate_target_column(df, "")
        assert is_valid is False
        assert "不能为空" in error_msg

    def test_invalid_target_column(self, sample_csv_path: str, invalid_target_column: str) -> None:
        """测试无效目标列。"""
        df = pd.read_csv(sample_csv_path)
        is_valid, error_msg = validate_target_column(df, invalid_target_column)
        assert is_valid is False
        assert "不存在" in error_msg


class TestValidateDataframe:
    """validate_dataframe 函数测试。"""

    def test_valid_dataframe(self) -> None:
        """测试有效 DataFrame。"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        is_valid, error_msg = validate_dataframe(df)
        assert is_valid is True
        assert error_msg is None

    def test_none_dataframe(self) -> None:
        """测试 None DataFrame。"""
        is_valid, error_msg = validate_dataframe(None)  # type: ignore[arg-type]
        assert is_valid is False
        assert "为空" in error_msg

    def test_insufficient_rows(self) -> None:
        """测试行数不足。"""
        df = pd.DataFrame({"a": [1]})
        is_valid, error_msg = validate_dataframe(df, min_rows=5)
        assert is_valid is False
        assert "不足" in error_msg


class TestSafeReadCsv:
    """safe_read_csv 函数测试。"""

    def test_valid_csv(self, sample_csv_path: str) -> None:
        """测试有效 CSV 文件。"""
        df, error_msg = safe_read_csv(sample_csv_path)
        assert df is not None
        assert error_msg is None
        assert len(df) > 0

    def test_non_existent_file(self, non_existent_csv_path: str) -> None:
        """测试不存在的文件。"""
        df, error_msg = safe_read_csv(non_existent_csv_path)
        assert df is None
        assert "不存在" in error_msg

    def test_empty_file(self, empty_csv_file: Path) -> None:
        """测试空文件。"""
        df, error_msg = safe_read_csv(str(empty_csv_file))
        assert df is None
        assert "为空" in error_msg


class TestGetFeatureTypes:
    """get_feature_types 函数测试。"""

    def test_mixed_types(self) -> None:
        """测试混合类型列。"""
        df = pd.DataFrame({
            "numeric": [1, 2, 3],
            "categorical": ["a", "b", "c"],
            "also_numeric": [4.0, 5.0, 6.0],
        })
        numeric_cols, categorical_cols = get_feature_types(df)
        assert "numeric" in numeric_cols
        assert "also_numeric" in numeric_cols
        assert "categorical" in categorical_cols

    def test_all_numeric(self) -> None:
        """测试全数值列。"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        numeric_cols, categorical_cols = get_feature_types(df)
        assert len(numeric_cols) == 2
        assert len(categorical_cols) == 0

    def test_all_categorical(self) -> None:
        """测试全类别列。"""
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
        numeric_cols, categorical_cols = get_feature_types(df)
        assert len(numeric_cols) == 0
        assert len(categorical_cols) == 2


class TestBuildSuccessResponse:
    """build_success_response 函数测试。"""

    def test_simple_data(self) -> None:
        """测试简单数据。"""
        data = {"key": "value", "number": 42}
        response = build_success_response(data)
        parsed = json.loads(response)
        assert parsed == data

    def test_nested_data(self) -> None:
        """测试嵌套数据。"""
        data = {"level1": {"level2": [1, 2, 3]}}
        response = build_success_response(data)
        parsed = json.loads(response)
        assert parsed == data


class TestBuildErrorResponse:
    """build_error_response 函数测试。"""

    def test_error_message(self) -> None:
        """测试错误信息。"""
        error_msg = "测试错误信息"
        response = build_error_response(error_msg)
        parsed = json.loads(response)
        assert "error" in parsed
        assert parsed["error"] == error_msg
