from typing import Tuple, Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from data_eda_preprocessing import get_train_valid_data
from gbm_models_comparison import compare_gradient_boosting_models


def tune_best_model(
    csv_path: str = "train_c.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    n_trials: int = 10,
    timeout: int | None = 120,
    precomputed_split=None,
    comparison_result: Tuple[Dict[str, float], str] | None = None,
) -> Tuple[str, Dict, float]:
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Модуль optuna не установлен. Установи его командой: pip install optuna"
        )

    if precomputed_split is None:
        X_train, X_valid, y_train, y_valid, _ = get_train_valid_data(
            csv_path=csv_path,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        X_train, X_valid, y_train, y_valid, _ = precomputed_split

    if comparison_result is None:
        scores, best_model_name = compare_gradient_boosting_models(
            csv_path=csv_path,
            test_size=test_size,
            random_state=random_state,
            precomputed_split=precomputed_split,
        )
    else:
        scores, best_model_name = comparison_result

    def objective(trial: "optuna.trial.Trial") -> float:
        if best_model_name == "sklearn_GradientBoosting":
            from sklearn.ensemble import GradientBoostingClassifier

            n_estimators = trial.suggest_int("n_estimators", 50, 400)
            learning_rate = trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            )
            max_depth = trial.suggest_int("max_depth", 1, 5)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )

        elif best_model_name == "XGBoost":
            from xgboost import XGBClassifier

            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            learning_rate = trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            )
            max_depth = trial.suggest_int("max_depth", 3, 8)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            )
            reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)

            model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                tree_method="hist",
            )

        elif best_model_name == "LightGBM":
            from lightgbm import LGBMClassifier

            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            learning_rate = trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            )
            num_leaves = trial.suggest_int("num_leaves", 16, 256)
            max_depth = trial.suggest_int("max_depth", -1, 10)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            )
            reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)

            model = LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                objective="binary",
                random_state=random_state,
            )

        elif best_model_name == "CatBoost":
            from catboost import CatBoostClassifier

            iterations = trial.suggest_int("iterations", 150, 500)
            learning_rate = trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            )
            depth = trial.suggest_int("depth", 4, 10)
            l2_leaf_reg = trial.suggest_float(
                "l2_leaf_reg", 1.0, 10.0, log=True
            )

            model = CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=random_state,
                verbose=False,
            )

        else:
            raise ValueError(f"Неизвестное имя модели: {best_model_name}")

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, y_proba)
        return score

    from optuna.samplers import TPESampler

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
    except KeyboardInterrupt:
        pass

    if len(study.trials) == 0:
        raise RuntimeError("Optuna не успела провести ни одного trial.")

    return best_model_name, study.best_params, study.best_value
