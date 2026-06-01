"""
建模工具单元测试。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.modeling_tools import (
    _detect_id_like_columns,
    _infer_task_type,
    train_baseline_models,
)


class TestInferTaskType:
    """_infer_task_type 函数测试。"""

    def test_classification_task(self) -> None:
        """测试分类任务推断。"""
        y = pd.Series([0, 1, 0, 1, 0, 1])
        assert _infer_task_type(y) == "classification"

    def test_regression_task(self) -> None:
        """测试回归任务推断。"""
        y = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5])
        assert _infer_task_type(y) == "regression"

    def test_categorical_target(self) -> None:
        """测试类别型目标。"""
        y = pd.Series(["cat", "dog", "bird", "cat", "dog"])
        assert _infer_task_type(y) == "classification"


class TestDetectIdLikeColumns:
    """_detect_id_like_columns 函数测试。"""

    def test_id_column(self) -> None:
        """测试 ID 列检测。"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "value": [10, 20, 30],
        })
        result = _detect_id_like_columns(df)
        assert "id" in result

    def test_index_column(self) -> None:
        """测试索引列检测。"""
        df = pd.DataFrame({
            "index": [1, 2, 3],
            "name": ["a", "b", "c"],
        })
        result = _detect_id_like_columns(df)
        assert "index" in result

    def test_unique_numeric_column(self) -> None:
        """测试唯一数值列检测。"""
        df = pd.DataFrame({
            "passenger_id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
            "age": [25, 30, 35, 40, 45],
        })
        result = _detect_id_like_columns(df)
        assert "passenger_id" in result

    def test_no_id_columns(self) -> None:
        """测试无 ID 列。"""
        df = pd.DataFrame({
            "name": ["a", "b", "c"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "SF"],
        })
        result = _detect_id_like_columns(df)
        assert len(result) == 0


class TestTrainBaselineModels:
    """train_baseline_models 函数测试。"""

    @pytest.mark.asyncio
    async def test_classification_task(self, sample_csv_path: str, target_column: str) -> None:
        """测试分类任务。"""
        result = await train_baseline_models(sample_csv_path, target_column)
        parsed = json.loads(result)
        assert "error" not in parsed
        assert parsed["task_type"] == "classification"
        assert "models" in parsed
        assert "best_model_by_rule" in parsed

    @pytest.mark.asyncio
    async def test_invalid_csv_path(self, non_existent_csv_path: str, target_column: str) -> None:
        """测试无效 CSV 路径。"""
        result = await train_baseline_models(non_existent_csv_path, target_column)
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_invalid_target_column(self, sample_csv_path: str, invalid_target_column: str) -> None:
        """测试无效目标列。"""
        result = await train_baseline_models(sample_csv_path, invalid_target_column)
        parsed = json.loads(result)
        assert "error" in parsed
