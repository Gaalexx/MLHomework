from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from data_eda_preprocessing import get_train_valid_data


class CustomBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_: List = []
        self.n_samples_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None

    def _sample_indices(self, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        if isinstance(self.max_samples, float):
            n_subsamples = int(self.max_samples * n_samples)
        else:
            n_subsamples = int(self.max_samples)

        n_subsamples = max(1, min(n_subsamples, n_samples))

        if self.bootstrap:
            indices = rng.randint(0, n_samples, size=n_subsamples)
        else:
            indices = rng.choice(n_samples, size=n_subsamples, replace=False)

        return indices

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomBaggingClassifier":
        X = np.asarray(X)
        y = np.asarray(y)

        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        self.n_samples_ = n_samples
        self.classes_ = np.unique(y)

        self.estimators_ = []

        for i in range(self.n_estimators):
            indices = self._sample_indices(n_samples, rng)
            X_sample = X[indices]
            y_sample = y[indices]

            estimator = clone(self.base_estimator)
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("The model has not been fitted yet.")

        X = np.asarray(X)
        proba_sum = None

        for est in self.estimators_:
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(X)
            else:
                pred = est.predict(X)
                proba = np.zeros((X.shape[0], len(self.classes_)))
                for idx, cls in enumerate(self.classes_):
                    proba[:, idx] = (pred == cls).astype(float)

            if proba_sum is None:
                proba_sum = proba
            else:
                proba_sum += proba

        avg_proba = proba_sum / len(self.estimators_)
        return avg_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]


def evaluate_custom_bagging(
    base_estimator=None,
    n_estimators: int = 50,
    max_samples: float = 0.8,
    random_state: int = 42,
    precomputed_split: Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]
    ] = None,
) -> Tuple[float, float]:
    from sklearn.ensemble import BaggingClassifier

    if base_estimator is None:
        base_estimator = DecisionTreeClassifier(
            max_depth=5, random_state=random_state
        )

    if precomputed_split is None:
        X_train, X_valid, y_train, y_valid, _ = get_train_valid_data(
            csv_path="train_c.csv", test_size=0.2, random_state=random_state
        )
    else:
        X_train, X_valid, y_train, y_valid, _ = precomputed_split

    custom_model = CustomBaggingClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=True,
        random_state=random_state,
    )
    custom_model.fit(X_train, y_train)
    y_valid_proba_custom = custom_model.predict_proba(X_valid)[:, 1]
    roc_auc_custom = roc_auc_score(y_valid, y_valid_proba_custom)

    try:
        sk_model = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=True,
            random_state=random_state,
        )
    except TypeError:
        sk_model = BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=True,
            random_state=random_state,
        )

    sk_model.fit(X_train, y_train)
    y_valid_proba_sk = sk_model.predict_proba(X_valid)[:, 1]
    roc_auc_sk = roc_auc_score(y_valid, y_valid_proba_sk)

    return roc_auc_custom, roc_auc_sk
