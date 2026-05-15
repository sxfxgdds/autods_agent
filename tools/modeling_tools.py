"""
异步建模工具：基于 sklearn Pipeline 的基线模型与交叉验证。
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from tools.data_tools import _read_csv_checked, _to_json_serializable


def _infer_task_type(y: pd.Series) -> str:
    """根据目标变量推断分类或回归。"""
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    nuniq = y.nunique(dropna=True)
    if nuniq <= 20:
        return "classification"
    return "regression"


def _detect_id_like_columns(X: pd.DataFrame) -> list[str]:
    """识别 ID / 索引类列，避免把明显标识符送进通用数值/独热流程。"""
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


async def train_baseline_models(csv_path: str, target_col: str) -> str:
    """
    训练基线模型并返回 5 折交叉验证指标（JSON）。
    """
    df, err = _read_csv_checked(csv_path, target_col)
    if err is not None:
        return json.dumps({"error": err}, ensure_ascii=False, indent=2)

    work = df.dropna(subset=[target_col]).copy()
    n_after_drop = len(work)
    if n_after_drop < 5:
        return json.dumps(
            {
                "error": f"有效样本过少（删除目标缺失后 n={n_after_drop}），无法进行 5 折交叉验证。"
            },
            ensure_ascii=False,
            indent=2,
        )

    y = work[target_col]
    task_type = _infer_task_type(y)

    X = work.drop(columns=[target_col])
    dropped_id_like = _detect_id_like_columns(X)
    if dropped_id_like:
        X = X.drop(columns=dropped_id_like)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

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

    if not numeric_features and not categorical_features:
        return json.dumps(
            {"error": "删除 ID 类字段后无可用特征列。"},
            ensure_ascii=False,
            indent=2,
        )

    if not categorical_features:
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, numeric_features)]
        )
    elif not numeric_features:
        preprocessor = ColumnTransformer(
            transformers=[("cat", categorical_transformer, categorical_features)]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

    models: dict[str, Any] = {}
    n_features_after_prep = "（经 ColumnTransformer 后特征维度取决于 OneHot）"

    if task_type == "classification":
        # 编码目标为整数标签
        y_codes, uniques = pd.factorize(y, sort=True)
        n_classes = len(uniques)
        if n_classes < 2:
            return json.dumps(
                {"error": "分类任务需要至少 2 个类别。"},
                ensure_ascii=False,
                indent=2,
            )

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

    else:
        y_num = pd.to_numeric(y, errors="coerce")
        if y_num.isna().all():
            return json.dumps(
                {"error": "回归任务目标列无法转为数值。"},
                ensure_ascii=False,
                indent=2,
            )
        # 若存在无法解析为数值的行，已在 y 上；对齐 X
        mask = y_num.notna()
        X = X.loc[mask].reset_index(drop=True)
        y_num = y_num.loc[mask].reset_index(drop=True)
        n_after_drop = len(y_num)
        if n_after_drop < 5:
            return json.dumps(
                {
                    "error": (
                        f"回归任务在剔除目标非数值行后样本数 n={n_after_drop}，"
                        "不足以进行 5 折交叉验证。"
                    )
                },
                ensure_ascii=False,
                indent=2,
            )

        reg_ridge = Pipeline(
            steps=[("prep", preprocessor), ("model", Ridge())]
        )
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
                    y_num,
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

    # 选出最优模型（简单规则）
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
    return json.dumps(_to_json_serializable(payload), ensure_ascii=False, indent=2)
