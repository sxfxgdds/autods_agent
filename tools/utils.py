"""
工具函数通用模块：提供 JSON 响应构建、输入验证等通用功能。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from logging_config import get_logger

logger = get_logger(__name__)


def build_success_response(data: Any) -> str:
    """
    构建成功响应 JSON 字符串。

    Args:
        data: 要序列化的数据

    Returns:
        JSON 字符串
    """
    from tools.data_tools import _to_json_serializable

    return json.dumps(
        _to_json_serializable(data),
        ensure_ascii=False,
        indent=2,
    )


def build_error_response(error_message: str) -> str:
    """
    构建错误响应 JSON 字符串。

    Args:
        error_message: 错误信息

    Returns:
        JSON 字符串
    """
    logger.error(f"工具错误: {error_message}")
    return json.dumps({"error": error_message}, ensure_ascii=False, indent=2)


def validate_csv_path(csv_path: str) -> tuple[bool, str | None]:
    """
    验证 CSV 文件路径。

    Args:
        csv_path: CSV 文件路径

    Returns:
        (是否有效, 错误信息)
    """
    if not csv_path:
        return False, "CSV 文件路径不能为空"

    if not os.path.exists(csv_path):
        return False, f"文件不存在: {csv_path}"

    if not os.path.isfile(csv_path):
        return False, f"路径不是文件: {csv_path}"

    if not csv_path.lower().endswith(".csv"):
        return False, f"文件不是 CSV 格式: {csv_path}"

    # 检查文件大小（限制为 100MB）
    file_size = os.path.getsize(csv_path)
    max_size = 100 * 1024 * 1024  # 100MB
    if file_size > max_size:
        return False, f"文件过大: {file_size / (1024 * 1024):.2f}MB，最大支持 100MB"

    return True, None


def validate_target_column(
    df: pd.DataFrame, target_col: str
) -> tuple[bool, str | None]:
    """
    验证目标列是否存在。

    Args:
        df: DataFrame
        target_col: 目标列名

    Returns:
        (是否有效, 错误信息)
    """
    if not target_col:
        return False, "目标列名不能为空"

    if target_col not in df.columns:
        available_cols = list(df.columns)
        return False, (
            f"目标列 '{target_col}' 不存在。可用列: {available_cols}"
        )

    return True, None


def validate_dataframe(
    df: pd.DataFrame, min_rows: int = 1
) -> tuple[bool, str | None]:
    """
    验证 DataFrame 是否有效。

    Args:
        df: DataFrame
        min_rows: 最小行数要求

    Returns:
        (是否有效, 错误信息)
    """
    if df is None:
        return False, "DataFrame 为空"

    if len(df) < min_rows:
        return False, f"数据行数不足: {len(df)} < {min_rows}"

    return True, None


def safe_read_csv(csv_path: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    安全读取 CSV 文件。

    Args:
        csv_path: CSV 文件路径

    Returns:
        (DataFrame 或 None, 错误信息或 None)
    """
    try:
        logger.info(f"读取 CSV 文件: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"成功读取 CSV 文件: {df.shape[0]} 行, {df.shape[1]} 列")
        return df, None
    except FileNotFoundError:
        return None, f"文件不存在: {csv_path}"
    except PermissionError:
        return None, f"没有权限读取文件: {csv_path}"
    except pd.errors.EmptyDataError:
        return None, f"文件为空: {csv_path}"
    except pd.errors.ParserError as e:
        return None, f"CSV 解析错误: {e}"
    except Exception as e:  # noqa: BLE001
        return None, f"读取文件失败: {type(e).__name__}: {e}"


def get_feature_types(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    获取数值型和类别型特征列。

    Args:
        df: DataFrame

    Returns:
        (数值型列列表, 类别型列列表)
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols
