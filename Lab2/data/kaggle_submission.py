from __future__ import annotations

import pandas as pd

from data_eda_preprocessing import (
    load_train_data,
    add_date_features,
    build_preprocessor,
    TARGET_COL,
)

from sklearn.ensemble import GradientBoostingClassifier


def _build_model(model_name: str | None, params: dict | None, random_state: int):
    model_name = model_name or "sklearn_GradientBoosting"
    default_params: dict = {}

    if model_name == "sklearn_GradientBoosting":
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
        }
    elif model_name == "XGBoost":
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
    elif model_name == "LightGBM":
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
    elif model_name == "CatBoost":
        default_params = {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
        }

    params = {**default_params, **(params or {})}

    if model_name == "sklearn_GradientBoosting":
        return GradientBoostingClassifier(
            random_state=random_state, **params
        )

    if model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("[WARN] xgboost недоступен, используем sklearn GradientBoosting.")
            return GradientBoostingClassifier(random_state=random_state)
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            tree_method="hist",
            **params,
        )

    if model_name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            print("[WARN] lightgbm недоступен, используем sklearn GradientBoosting.")
            return GradientBoostingClassifier(random_state=random_state)
        return LGBMClassifier(
            objective="binary",
            random_state=random_state,
            **params,
        )

    if model_name == "CatBoost":
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            print("[WARN] catboost недоступен, используем sklearn GradientBoosting.")
            return GradientBoostingClassifier(random_state=random_state)
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=random_state,
            **params,
        )

    print(f"[WARN] Неизвестная модель '{model_name}', используем sklearn GradientBoosting.")
    return GradientBoostingClassifier(random_state=random_state)


def train_final_model(
    train_csv: str = "train_c.csv",
    random_state: int = 42,
    best_model_name: str | None = None,
    best_params: dict | None = None,
):
    df = load_train_data(train_csv)
    df = add_date_features(df)

    df = df[df[TARGET_COL].notna()].copy()
    y = df[TARGET_COL].astype(int).values
    X = df.drop(columns=[TARGET_COL])

    preprocessor = build_preprocessor(df)
    X_all = preprocessor.fit_transform(X)

    model = _build_model(best_model_name, best_params, random_state)

    model.fit(X_all, y)

    return preprocessor, model


def make_submission(
    train_csv: str = "train_c.csv",
    test_csv: str = "test_c.csv",
    sample_csv: str = "ex_c.csv",
    out_csv: str = "submission.csv",
    random_state: int = 42,
    best_model_name: str | None = None,
    best_params: dict | None = None,
):
    preprocessor, model = train_final_model(
        train_csv=train_csv,
        random_state=random_state,
        best_model_name=best_model_name,
        best_params=best_params,
    )

    df_test = pd.read_csv(test_csv)
    df_test = add_date_features(df_test)

    if "ID" in df_test.columns:
        df_test = df_test.sort_values("ID").reset_index(drop=True)

    X_test = preprocessor.transform(df_test)

    proba_test = model.predict_proba(X_test)[:, 1]

    submission = pd.read_csv(sample_csv)

    if "ID" in submission.columns:
        submission = submission.sort_values("ID").reset_index(drop=True)

    if "LoanApproved" not in submission.columns:
        raise ValueError("В ex_c.csv нет столбца 'LoanApproved'")

    if len(submission) != len(df_test):
        raise ValueError(
            f"ex_c.csv и test_c.csv имеют разное число строк: "
            f"{len(submission)} vs {len(df_test)}. Проверь файлы."
        )

    submission["LoanApproved"] = proba_test

    submission.to_csv(out_csv, index=False)
