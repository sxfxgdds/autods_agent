"""
异步数据工具：概况分析与质量审查。
所有统计结果来自 pandas，不由 LLM 编造。
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd


def _to_json_serializable(obj: Any) -> Any:
    """将 numpy / pandas 等类型转为 JSON 可序列化 Python 原生类型。"""
    if obj is None:
        return None
    if obj is pd.NaT:
        return None
    try:
        if obj is pd.NA:  # type: ignore[comparison-overlap]
            return None
    except (AttributeError, TypeError):
        pass

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _to_json_serializable(obj.tolist())
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, Decimal):
        return float(obj) if obj.is_finite() else None
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, (pd.Series,)):
        return _to_json_serializable(obj.to_dict())
    if isinstance(obj, (pd.Index,)):
        return [_to_json_serializable(x) for x in obj.tolist()]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _read_csv_checked(
    csv_path: str, target_col: str
) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return None, f"错误：找不到文件 {csv_path}"
    except Exception as e:  # noqa: BLE001 — 向用户返回可读信息
        return None, f"错误：读取 CSV 失败 — {type(e).__name__}: {e}"

    if target_col not in df.columns:
        return None, (
            f"错误：目标列 '{target_col}' 不存在。可用列：{list(df.columns)}"
        )
    return df, None


async def profile_dataset(csv_path: str, target_col: str) -> str:
    """
    读取 CSV 并返回数据集概况（JSON 字符串）。
    """
    df, err = _read_csv_checked(csv_path, target_col)
    if err is not None:
        return json.dumps({"error": err}, ensure_ascii=False, indent=2)

    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    dtypes_map = {c: str(df[c].dtype) for c in df.columns}

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    missing_rate = df.isna().mean().sort_values(ascending=False)
    top_missing = missing_rate.head(10)
    top_missing_list = [
        {"column": str(idx), "missing_rate": float(val)} for idx, val in top_missing.items()
    ]

    dup_count = int(df.duplicated().sum())

    y = df[target_col]
    target_dtype = str(y.dtype)
    n_unique_target = int(y.nunique(dropna=True))
    vc = y.value_counts(dropna=False)
    value_counts_dict = {str(k): int(v) for k, v in vc.items()}

    payload = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "column_names": [str(c) for c in df.columns.tolist()],
        "dtypes": dtypes_map,
        "numeric_columns": [str(c) for c in num_cols],
        "categorical_columns": [str(c) for c in cat_cols],
        "top_10_missing_rate_columns": top_missing_list,
        "duplicate_rows": dup_count,
        "target": {
            "name": target_col,
            "dtype": target_dtype,
            "n_unique": n_unique_target,
            "value_counts": value_counts_dict,
        },
    }
    return json.dumps(_to_json_serializable(payload), ensure_ascii=False, indent=2)


async def audit_dataset(csv_path: str, target_col: str) -> str:
    """
    数据质量审查：缺失、重复、常数列、目标类别不平衡等。
    """
    df, err = _read_csv_checked(csv_path, target_col)
    if err is not None:
        return json.dumps({"error": err}, ensure_ascii=False, indent=2)

    issues: list[str] = []
    suggestions: list[str] = []

    # 缺失值
    na_per_col = df.isna().sum()
    cols_with_na = na_per_col[na_per_col > 0]
    if len(cols_with_na) > 0:
        top_na = cols_with_na.sort_values(ascending=False).head(15)
        na_dict = {str(c): int(top_na.loc[c]) for c in top_na.index}
        issues.append(
            f"存在缺失值的字段共 {len(cols_with_na)} 个（前 15 列计数）：{na_dict}"
        )
        suggestions.append("对数值列可考虑中位数/均值填补；类别列可用众数或单独缺失类别。")
    else:
        issues.append("未发现缺失值。")

    # 重复行
    dup = int(df.duplicated().sum())
    if dup > 0:
        issues.append(f"存在完全重复行：{dup} 条。")
        suggestions.append("评估是否删除重复行，或检查是否为数据录入错误。")
    else:
        issues.append("未发现完全重复行。")

    # 常数列（单值列，忽略全 NA）
    constant_cols = []
    for c in df.columns:
        s = df[c]
        nunique = s.nunique(dropna=True)
        if nunique <= 1 and s.notna().any():
            constant_cols.append(c)
    if constant_cols:
        issues.append(f"常数或近似常数字段：{constant_cols}")
        suggestions.append("常数列对建模无信息量，可考虑删除。")
    else:
        issues.append("未发现常数列（在 dropna 后 nunique<=1 意义下）。")

    # 目标变量类别不平衡（非数值或类别型目标）
    y = df[target_col]
    value_counts = y.value_counts(dropna=False)
    n_valid = int(y.notna().sum())
    imbalance_note = None
    if n_valid > 0 and y.nunique(dropna=True) > 1:
        max_ratio = float(value_counts.iloc[0] / n_valid)
        if max_ratio > 0.8:
            imbalance_note = (
                f"目标变量最大类别占比 {max_ratio:.4f} > 0.8，存在明显类别不平衡。"
            )
            issues.append(imbalance_note)
            suggestions.append(
                "可考虑 class_weight、重采样、或改用适合不平衡数据的指标（如 F1、PR-AUC）。"
            )
        else:
            imbalance_note = f"目标最大类别占比 {max_ratio:.4f}，未超过 0.8 阈值。"
            issues.append(imbalance_note)
    else:
        issues.append("目标变量唯一值不足或全缺失，无法评估类别不平衡。")

    payload = {
        "issues": issues,
        "suggestions": list(dict.fromkeys(suggestions)),  # 去重保序
    }
    return json.dumps(_to_json_serializable(payload), ensure_ascii=False, indent=2)
