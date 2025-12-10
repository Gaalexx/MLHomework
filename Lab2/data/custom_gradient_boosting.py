from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier

from data_eda_preprocessing import get_train_valid_data


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class CustomGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.trees_: List[DecisionTreeRegressor] = []
        self.init_score_: float = 0.0  # начальный log-odds
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomGradientBoostingClassifier":
        X = np.asarray(X)
        y = np.asarray(y)

        if set(np.unique(y)) - {0, 1}:
            raise ValueError("CustomGradientBoostingClassifier поддерживает только бинарные метки {0,1}.")

        n_samples = X.shape[0]
        np.random.RandomState(self.random_state)

        p = np.clip(y.mean(), 1e-5, 1 - 1e-5)
        self.init_score_ = np.log(p / (1 - p))

        F = np.full(n_samples, self.init_score_, dtype=float)
        self.trees_ = []
        self.classes_ = np.array([0, 1])

        for m in range(self.n_estimators):
            prob = _sigmoid(F)
            residual = y - prob  # градиент лог-лосса

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=None if self.random_state is None else self.random_state + m,
            )
            tree.fit(X, residual)
            update = tree.predict(X)

            F += self.learning_rate * update
            self.trees_.append(tree)

        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        F = np.full(X.shape[0], self.init_score_, dtype=float)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("The model has not been fitted yet.")

        F = self._raw_predict(X)
        prob_pos = _sigmoid(F)
        prob_neg = 1.0 - prob_pos
        return np.vstack([prob_neg, prob_pos]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


def evaluate_custom_gradient_boosting(
    n_estimators: int = 150,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    precomputed_split: Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]
    ] = None,
) -> Tuple[float, float]:
    if precomputed_split is None:
        X_train, X_valid, y_train, y_valid, _ = get_train_valid_data(
            csv_path="train_c.csv", test_size=0.2, random_state=random_state
        )
    else:
        X_train, X_valid, y_train, y_valid, _ = precomputed_split

    custom_model = CustomGradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    custom_model.fit(X_train, y_train)
    y_valid_proba_custom = custom_model.predict_proba(X_valid)[:, 1]
    roc_auc_custom = roc_auc_score(y_valid, y_valid_proba_custom)

    sk_model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    sk_model.fit(X_train, y_train)
    y_valid_proba_sk = sk_model.predict_proba(X_valid)[:, 1]
    roc_auc_sk = roc_auc_score(y_valid, y_valid_proba_sk)

    return roc_auc_custom, roc_auc_sk
