"""
异步建模工具：基于 sklearn Pipeline 的基线模型与交叉验证。
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from logging_config import get_logger
from tools.data_tools import _read_csv_checked, _to_json_serializable
from tools.utils import build_error_response, build_success_response

logger = get_logger(__name__)


def _infer_task_type(y: pd.Series) -> str:
    """
    根据目标变量推断分类或回归。

    Args:
        y: 目标变量

    Returns:
        "classification" 或 "regression"
    """
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    nuniq = y.nunique(dropna=True)
    if nuniq <= 20:
        return "classification"
    return "regression"


def _detect_id_like_columns(X: pd.DataFrame) -> list[str]:
    """
    识别 ID / 索引类列，避免把明显标识符送进通用数值/独热流程。

    Args:
        X: 特征 DataFrame

    Returns:
        被识别为 ID 类的列名列表
    """
    dropped: list[str] = []
    n = len(X)
    if n == 0:
        return dropped

    reserved_names = re.compile(r"^(id|index|unnamed:\s*0)$", re.IGNORECASE)
    id_suffix = re.compile(r"id$", re.IGNORECASE)

    for col in X.columns:
        cl = str(col).strip()
        if reserved_names.match(cl):
            dropped.append(col)
            continue
        if id_suffix.search(cl):
            dropped.append(col)
            continue
        s = X[col]
        nu = s.nunique(dropna=True)
        # 数值列且每行唯一 → 典型 PassengerId 类
        if nu == n and n > 1 and pd.api.types.is_numeric_dtype(s):
            dropped.append(col)
    return list(dict.fromkeys(dropped))


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    构建预处理器。

    Args:
        numeric_features: 数值型特征列
        categorical_features: 类别型特征列

    Returns:
        ColumnTransformer 预处理器
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    return ColumnTransformer(transformers=transformers)


def _train_classification_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
) -> tuple[dict[str, Any], int]:
    """
    训练分类模型。

    Args:
        X: 特征
        y: 目标变量
        preprocessor: 预处理器

    Returns:
        (模型结果字典, 类别数)
    """
    # 编码目标为整数标签
    y_codes, uniques = pd.factorize(y, sort=True)
    n_classes = len(uniques)

    models: dict[str, Any] = {}

    clf_lr = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    clf_rf = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "f1_weighted": "f1_weighted"}
    roc_available = n_classes == 2

    for name, est in [("logistic_regression", clf_lr), ("random_forest", clf_rf)]:
        try:
            cv_res = cross_validate(
                est,
                X,
                y_codes,
                cv=cv,
                scoring=scoring,
                n_jobs=None,
            )
        except Exception as e:  # noqa: BLE001
            models[name] = {"error": f"{type(e).__name__}: {e}"}
            continue

        entry: dict[str, Any] = {
            "accuracy_mean": float(np.nanmean(cv_res["test_accuracy"])),
            "accuracy_std": float(np.nanstd(cv_res["test_accuracy"])),
            "f1_weighted_mean": float(np.nanmean(cv_res["test_f1_weighted"])),
            "f1_weighted_std": float(np.nanstd(cv_res["test_f1_weighted"])),
        }
        if roc_available:
            try:
                cv_roc = cross_validate(
                    est,
                    X,
                    y_codes,
                    cv=cv,
                    scoring={"roc_auc": "roc_auc"},
                    n_jobs=None,
                )
                entry["roc_auc_mean"] = float(np.nanmean(cv_roc["test_roc_auc"]))
                entry["roc_auc_std"] = float(np.nanstd(cv_roc["test_roc_auc"]))
            except Exception as e:  # noqa: BLE001
                entry["roc_auc_error"] = f"{type(e).__name__}: {e}"
        models[name] = entry

    return models, n_classes


def _train_regression_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
) -> dict[str, Any]:
    """
    训练回归模型。

    Args:
        X: 特征
        y: 目标变量
        preprocessor: 预处理器

    Returns:
        模型结果字典
    """
    models: dict[str, Any] = {}

    reg_ridge = Pipeline(steps=[("prep", preprocessor), ("model", Ridge())])
    reg_rf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    for name, est in [("ridge", reg_ridge), ("random_forest", reg_rf)]:
        try:
            cv_res = cross_validate(
                est,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=None,
            )
        except Exception as e:  # noqa: BLE001
            models[name] = {"error": f"{type(e).__name__}: {e}"}
            continue

        mae_pos = -cv_res["test_mae"]
        rmse_pos = -cv_res["test_rmse"]
        models[name] = {
            "mae_mean": float(np.nanmean(mae_pos)),
            "mae_std": float(np.nanstd(mae_pos)),
            "rmse_mean": float(np.nanmean(rmse_pos)),
            "rmse_std": float(np.nanstd(rmse_pos)),
            "r2_mean": float(np.nanmean(cv_res["test_r2"])),
            "r2_std": float(np.nanstd(cv_res["test_r2"])),
        }

    return models


def _select_best_model(
    models: dict[str, Any],
    task_type: str,
) -> tuple[str | None, float | None]:
    """
    选择最优模型。

    Args:
        models: 模型结果字典
        task_type: 任务类型

    Returns:
        (最优模型名称, 最优分数)
    """
    best_model = None
    best_score: float | None = None

    if task_type == "classification":
        for mname, mval in models.items():
            if isinstance(mval, dict) and "f1_weighted_mean" in mval:
                s = mval["f1_weighted_mean"]
                if best_score is None or s > best_score:
                    best_score = s
                    best_model = mname
    else:
        for mname, mval in models.items():
            if isinstance(mval, dict) and "rmse_mean" in mval:
                s = mval["rmse_mean"]
                if best_score is None or s < best_score:
                    best_score = s
                    best_model = mname

    return best_model, best_score


async def train_baseline_models(csv_path: str, target_col: str) -> str:
    """
    训练基线模型并返回 5 折交叉验证指标（JSON）。

    Args:
        csv_path: CSV 文件路径
        target_col: 目标列名

    Returns:
        JSON 字符串，包含模型训练结果或错误信息
    """
    logger.info(f"开始训练基线模型: {csv_path}")

    df, err = _read_csv_checked(csv_path, target_col)
    if err is not None:
        return build_error_response(err)

    try:
        work = df.dropna(subset=[target_col]).copy()
        n_after_drop = len(work)
        if n_after_drop < 5:
            return build_error_response(
                f"有效样本过少（删除目标缺失后 n={n_after_drop}），无法进行 5 折交叉验证。"
            )

        y = work[target_col]
        task_type = _infer_task_type(y)

        X = work.drop(columns=[target_col])
        dropped_id_like = _detect_id_like_columns(X)
        if dropped_id_like:
            X = X.drop(columns=dropped_id_like)

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        if not numeric_features and not categorical_features:
            return build_error_response("删除 ID 类字段后无可用特征列。")

        preprocessor = _build_preprocessor(numeric_features, categorical_features)
        n_features_after_prep = "（经 ColumnTransformer 后特征维度取决于 OneHot）"

        if task_type == "classification":
            if y.nunique(dropna=True) < 2:
                return build_error_response("分类任务需要至少 2 个类别。")

            models, n_classes = _train_classification_models(X, y, preprocessor)
        else:
            y_num = pd.to_numeric(y, errors="coerce")
            if y_num.isna().all():
                return build_error_response("回归任务目标列无法转为数值。")

            # 若存在无法解析为数值的行，已在 y 上；对齐 X
            mask = y_num.notna()
            X = X.loc[mask].reset_index(drop=True)
            y_num = y_num.loc[mask].reset_index(drop=True)
            n_after_drop = len(y_num)
            if n_after_drop < 5:
                return build_error_response(
                    f"回归任务在剔除目标非数值行后样本数 n={n_after_drop}，"
                    "不足以进行 5 折交叉验证。"
                )

            models = _train_regression_models(X, y_num, preprocessor)

        best_model, best_score = _select_best_model(models, task_type)

        payload = {
            "task_type": task_type,
            "n_samples_after_dropna_target": int(n_after_drop),
            "n_raw_features": int(X.shape[1]),
            "dropped_id_like_columns": [str(c) for c in dropped_id_like],
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
            "note_on_feature_dim": n_features_after_prep,
            "models": models,
            "best_model_by_rule": best_model,
            "best_primary_metric_value": best_score,
        }

        logger.info(f"基线模型训练完成: {task_type}, 最优模型: {best_model}")
        return build_success_response(payload)

    except Exception as e:  # noqa: BLE001
        logger.error(f"训练基线模型时出错: {e}")
        return build_error_response(f"训练基线模型时出错: {type(e).__name__}: {e}")
