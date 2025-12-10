import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


TARGET_COL = "LoanApproved"
DATE_COL = "ApplicationDate"

CATEGORICAL_COLUMNS = [
    "MaritalStatus",
    "HomeOwnershipStatus",
    "BankruptcyHistory",
    "LoanPurpose",
    "PaymentHistory",
    "UtilityBillsPaymentHistory",
    "EmploymentStatus",
    "EducationLevel",
]


def load_train_data(csv_path: str = "train_c.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def make_eda_plots(
    df: pd.DataFrame, output_dir: str = "eda_plots"
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    df[TARGET_COL].value_counts().plot(kind="bar")
    plt.title("Target distribution: LoanApproved")
    plt.xlabel("LoanApproved")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_distribution.png"))
    plt.close()

    scatter_pairs = [
        ("AnnualIncome", "LoanAmount"),
        ("CreditScore", "InterestRate"),
        ("DebtToIncomeRatio", "LoanAmount"),
    ]

    for x_col, y_col in scatter_pairs:
        if x_col in df.columns and y_col in df.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(df[x_col], df[y_col], alpha=0.3)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{y_col} vs {x_col}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"scatter_{x_col}_vs_{y_col}.png",
                )
            )
            plt.close()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)

    if numeric_cols:
        corr = df[numeric_cols + [TARGET_COL]].corr()

        plt.figure(figsize=(12, 10))
        im = plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        plt.close()


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL not in df.columns:
        return df

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    df["ApplicationYear"] = df[DATE_COL].dt.year
    df["ApplicationMonth"] = df[DATE_COL].dt.month
    df["ApplicationDay"] = df[DATE_COL].dt.day

    df = df.drop(columns=[DATE_COL])
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    df = add_date_features(df)

    all_feature_cols = [c for c in df.columns if c != TARGET_COL]

    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    numeric_cols = [c for c in all_feature_cols if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def get_train_valid_data(
    csv_path: str = "train_c.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    df = load_train_data(csv_path)
    df = add_date_features(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    before = len(df)
    df = df[df[TARGET_COL].notna()].copy()
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows with NaN in target '{TARGET_COL}'")

    y = df[TARGET_COL].astype(int).values  # если таргет 0/1, приводим к int
    X = df.drop(columns=[TARGET_COL])

    preprocessor = build_preprocessor(df)

    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_valid = preprocessor.transform(X_valid_raw)

    return X_train, X_valid, y_train, y_valid, preprocessor
