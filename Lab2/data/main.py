from data_eda_preprocessing import (
    load_train_data,
    make_eda_plots,
    get_train_valid_data,
)
from custom_bagging import evaluate_custom_bagging
from custom_gradient_boosting import evaluate_custom_gradient_boosting
from gbm_models_comparison import compare_gradient_boosting_models
from optuna_tuning import tune_best_model

from sklearn.ensemble import GradientBoostingClassifier

from metrics import (
    accuracy_score_manual,
    precision_score_manual,
    recall_score_manual,
    f1_score_manual,
    roc_auc_score_manual,
    pr_auc_score_manual,
)

from kaggle_submission import make_submission


TRAIN_PATH = "train_c.csv"
TEST_PATH = "test_c.csv"
SAMPLE_SUB_PATH = "ex_c.csv"
SUBMISSION_PATH = "submission.csv"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

RUN_CACHE = {
    "data_split": {},
    "bagging": {},
    "grad_boost": {},
    "gbm_compare": {},
    "optuna": {},
    "metrics_demo": {},
}


def get_cached_split(
    csv_path: str = TRAIN_PATH,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    key = (csv_path, test_size, random_state)
    cache = RUN_CACHE["data_split"]
    if key in cache:
        return cache[key], key

    data_split = get_train_valid_data(
        csv_path=csv_path, test_size=test_size, random_state=random_state
    )
    cache[key] = data_split
    return data_split, key


def run_eda() -> None:
    df = load_train_data(TRAIN_PATH)
    make_eda_plots(df, output_dir="eda_plots")


def run_bagging(data_split, split_key) -> None:
    params = (split_key, 50, 0.8, DEFAULT_RANDOM_STATE)
    cache = RUN_CACHE["bagging"]

    if params in cache:
        roc_custom, roc_sk = cache[params]
    else:
        roc_custom, roc_sk = evaluate_custom_bagging(
            n_estimators=50,
            max_samples=0.8,
            random_state=DEFAULT_RANDOM_STATE,
            precomputed_split=data_split,
        )
        cache[params] = (roc_custom, roc_sk)

    return roc_custom, roc_sk


def run_gradient_boosting(data_split, split_key) -> None:
    params = (split_key, 150, 0.1, 3, 1, DEFAULT_RANDOM_STATE)
    cache = RUN_CACHE["grad_boost"]

    if params in cache:
        roc_custom, roc_sk = cache[params]
    else:
        roc_custom, roc_sk = evaluate_custom_gradient_boosting(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=1,
            random_state=DEFAULT_RANDOM_STATE,
            precomputed_split=data_split,
        )
        cache[params] = (roc_custom, roc_sk)

    return roc_custom, roc_sk


def run_gbm_comparison(data_split, split_key) -> tuple:
    cache = RUN_CACHE["gbm_compare"]

    if split_key in cache:
        scores, best_name = cache[split_key]
    else:
        scores, best_name = compare_gradient_boosting_models(
            csv_path=TRAIN_PATH,
            test_size=split_key[1],
            random_state=split_key[2],
            precomputed_split=data_split,
        )
        cache[split_key] = (scores, best_name)

    return scores, best_name


def run_optuna_tuning(data_split, split_key, comparison_result):
    cache = RUN_CACHE["optuna"]
    key = (split_key, 10, 120)

    if key in cache:
        best_name, best_params, best_score = cache[key]
    else:
        best_name, best_params, best_score = tune_best_model(
            csv_path=TRAIN_PATH,
            test_size=split_key[1],
            random_state=split_key[2],
            n_trials=10,
            timeout=120,
            precomputed_split=data_split,
            comparison_result=comparison_result,
        )
        cache[key] = (best_name, best_params, best_score)

    return best_name, best_params, best_score



def run_metrics_demo(data_split, split_key) -> None:
    cache = RUN_CACHE["metrics_demo"]

    if split_key in cache:
        results = cache[split_key]
    else:
        X_train, X_valid, y_train, y_valid, _ = data_split

        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            random_state=DEFAULT_RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        y_valid_pred = (y_valid_proba >= 0.5).astype(int)

        manual = {
            "acc": accuracy_score_manual(y_valid, y_valid_pred),
            "prec": precision_score_manual(y_valid, y_valid_pred),
            "rec": recall_score_manual(y_valid, y_valid_pred),
            "f1": f1_score_manual(y_valid, y_valid_pred),
            "roc_auc": roc_auc_score_manual(y_valid, y_valid_proba),
            "pr_auc": pr_auc_score_manual(y_valid, y_valid_proba),
        }

        sklearn_results = None
        try:
            from sklearn import metrics as skm

            sklearn_results = {
                "acc": skm.accuracy_score(y_valid, y_valid_pred),
                "prec": skm.precision_score(y_valid, y_valid_pred),
                "rec": skm.recall_score(y_valid, y_valid_pred),
                "f1": skm.f1_score(y_valid, y_valid_pred),
                "roc_auc": skm.roc_auc_score(y_valid, y_valid_proba),
            }
            prec_curve, rec_curve, _ = skm.precision_recall_curve(
                y_valid, y_valid_proba
            )
            sklearn_results["pr_auc"] = skm.auc(rec_curve, prec_curve)
        except ImportError:
            pass

        results = {"manual": manual, "sklearn": sklearn_results}
        cache[split_key] = results

    manual = results["manual"]
    sklearn_results = results["sklearn"]

    return manual, sklearn_results


def run_submission(best_model_name=None, best_params=None) -> None:
    make_submission(
        train_csv=TRAIN_PATH,
        test_csv=TEST_PATH,
        sample_csv=SAMPLE_SUB_PATH,
        out_csv=SUBMISSION_PATH,
        random_state=DEFAULT_RANDOM_STATE,
        best_model_name=best_model_name,
        best_params=best_params,
    )


if __name__ == "__main__":
    run_eda()
    data_split, split_key = get_cached_split(
        csv_path=TRAIN_PATH,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
    )

    bagging_scores = run_bagging(data_split, split_key)
    grad_scores = run_gradient_boosting(data_split, split_key)
    comparison_result = run_gbm_comparison(data_split, split_key)
    best_name, best_params, optuna_score = run_optuna_tuning(
        data_split, split_key, comparison_result
    )
    base_score = comparison_result[0].get(best_name)
    final_params = best_params
    if optuna_score is not None and base_score is not None and optuna_score < base_score:
        final_params = None

    run_metrics_demo(data_split, split_key)
    run_submission(best_model_name=best_name, best_params=final_params)
