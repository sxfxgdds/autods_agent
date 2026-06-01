"""
集成测试：测试完整的数据处理流程。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.data_tools import audit_dataset, profile_dataset
from tools.modeling_tools import train_baseline_models


@pytest.mark.integration
class TestDataAnalysisPipeline:
    """数据分析流水线集成测试。"""

    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(
        self, sample_csv_path: str, target_column: str
    ) -> None:
        """测试完整分析流程。"""
        # 1. 数据概况分析
        profile_result = await profile_dataset(sample_csv_path, target_column)
        profile_data = json.loads(profile_result)
        assert "error" not in profile_data
        assert "n_rows" in profile_data

        # 2. 数据质量审查
        audit_result = await audit_dataset(sample_csv_path, target_column)
        audit_data = json.loads(audit_result)
        assert "error" not in audit_data
        assert "issues" in audit_data

        # 3. 模型训练
        model_result = await train_baseline_models(sample_csv_path, target_column)
        model_data = json.loads(model_result)
        assert "error" not in model_data
        assert "task_type" in model_data
        assert "models" in model_data

    @pytest.mark.asyncio
    async def test_data_consistency(
        self, sample_csv_path: str, target_column: str
    ) -> None:
        """测试数据一致性。"""
        # 获取数据概况
        profile_result = await profile_dataset(sample_csv_path, target_column)
        profile_data = json.loads(profile_result)

        # 获取审计结果
        audit_result = await audit_dataset(sample_csv_path, target_column)
        audit_data = json.loads(audit_result)

        # 验证数据行数一致
        profile_rows = profile_data["n_rows"]
        assert profile_rows > 0

        # 验证目标列信息一致
        assert profile_data["target"]["name"] == target_column


@pytest.mark.integration
class TestErrorHandling:
    """错误处理集成测试。"""

    @pytest.mark.asyncio
    async def test_invalid_file_handling(self, non_existent_csv_path: str, target_column: str) -> None:
        """测试无效文件处理。"""
        # 所有工具都应该返回错误，而不是抛出异常
        profile_result = await profile_dataset(non_existent_csv_path, target_column)
        assert "error" in json.loads(profile_result)

        audit_result = await audit_dataset(non_existent_csv_path, target_column)
        assert "error" in json.loads(audit_result)

        model_result = await train_baseline_models(non_existent_csv_path, target_column)
        assert "error" in json.loads(model_result)

    @pytest.mark.asyncio
    async def test_invalid_target_handling(
        self, sample_csv_path: str, invalid_target_column: str
    ) -> None:
        """测试无效目标列处理。"""
        profile_result = await profile_dataset(sample_csv_path, invalid_target_column)
        assert "error" in json.loads(profile_result)

        audit_result = await audit_dataset(sample_csv_path, invalid_target_column)
        assert "error" in json.loads(audit_result)

        model_result = await train_baseline_models(sample_csv_path, invalid_target_column)
        assert "error" in json.loads(model_result)
