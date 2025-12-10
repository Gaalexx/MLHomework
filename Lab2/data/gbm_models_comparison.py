from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from data_eda_preprocessing import get_train_valid_data


def _fit_and_score(
    model, X_train, y_train, X_valid, y_valid, name: str
) -> float:
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, y_proba)
    return score


def compare_gradient_boosting_models(
    csv_path: str = "train_c.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    precomputed_split=None,
) -> Tuple[Dict[str, float], str]:
    if precomputed_split is None:
        X_train, X_valid, y_train, y_valid, _ = get_train_valid_data(
            csv_path=csv_path,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        X_train, X_valid, y_train, y_valid, _ = precomputed_split

    scores: Dict[str, float] = {}

    skl_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=random_state,
    )
    scores["sklearn_GradientBoosting"] = _fit_and_score(
        skl_model, X_train, y_train, X_valid, y_valid, "sklearn GradientBoosting"
    )

    try:
        from xgboost import XGBClassifier

        xgb_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            tree_method="hist",
        )

        scores["XGBoost"] = _fit_and_score(
            xgb_model, X_train, y_train, X_valid, y_valid, "XGBoost"
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        lgbm_model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=random_state,
        )

        scores["LightGBM"] = _fit_and_score(
            lgbm_model, X_train, y_train, X_valid, y_valid, "LightGBM"
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        cat_model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=random_state,
        )

        scores["CatBoost"] = _fit_and_score(
            cat_model, X_train, y_train, X_valid, y_valid, "CatBoost"
        )
    except ImportError:
        pass

    if not scores:
        raise RuntimeError("Не удалось обучить ни одну модель (все библиотеки отсутствуют).")

    best_model_name = max(scores, key=scores.get)

    return scores, best_model_name
