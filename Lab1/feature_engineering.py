import numpy as np
import pandas as pd


def create_features(df):
    df_new = df.copy()
    
    if 'CreditScore' in df_new.columns:
        df_new['CreditScore_squared'] = df_new['CreditScore'] ** 2
        df_new['CreditScore_cubed'] = df_new['CreditScore'] ** 3
    
    if 'MonthlyIncome' in df_new.columns:
        df_new['MonthlyIncome_squared'] = df_new['MonthlyIncome'] ** 2
    
    if 'AnnualIncome' in df_new.columns:
        df_new['AnnualIncome_squared'] = df_new['AnnualIncome'] ** 2
    
    if 'CreditScore' in df_new.columns and 'MonthlyIncome' in df_new.columns:
        df_new['CreditScore_x_Income'] = df_new['CreditScore'] * df_new['MonthlyIncome']
    
    if 'CreditScore' in df_new.columns and 'TotalDebtToIncomeRatio' in df_new.columns:
        df_new['Credit_x_DebtRatio'] = df_new['CreditScore'] * df_new['TotalDebtToIncomeRatio']
    
    if 'InterestRate' in df_new.columns and 'LoanAmount' in df_new.columns:
        df_new['Interest_x_Loan'] = df_new['InterestRate'] * df_new['LoanAmount']
    
    if 'LoanAmount' in df_new.columns and 'AnnualIncome' in df_new.columns:
        df_new['LoanToAnnualIncome'] = df_new['LoanAmount'] / (df_new['AnnualIncome'] + 1)
    
    if 'MonthlyLoanPayment' in df_new.columns and 'MonthlyIncome' in df_new.columns:
        df_new['PaymentToIncome'] = df_new['MonthlyLoanPayment'] / (df_new['MonthlyIncome'] + 1)
    
    return df_new


def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(to_drop, axis=1), to_drop


def select_best_features(train_df, target_col='RiskScore', top_n=30):
    correlations = train_df.corr()[target_col].abs().sort_values(ascending=False)
    correlations = correlations[correlations.index != target_col]
    top_features = correlations.head(top_n).index.tolist()
    
    print(f"Лучшие признаки по корреляции: {len(top_features)}")
    for feat in top_features[:5]:
        print(f"- {feat}: {correlations[feat]:.4f}")
    
    return top_features


def improve_preprocessing(train_df, test_df=None):
    print("Предобработка:")

    train_clean = train_df.copy()
    initial_rows = len(train_clean)

    if 'RiskScore' in train_clean.columns:
        train_clean = train_clean.dropna(subset=['RiskScore']).copy()
        train_clean = train_clean[(train_clean['RiskScore'] > -100) & (train_clean['RiskScore'] < 150)].copy()
        low_q = train_clean['RiskScore'].quantile(0.005)
        high_q = train_clean['RiskScore'].quantile(0.995)
        before = len(train_clean)
        train_clean = train_clean[(train_clean['RiskScore'] >= low_q) & (train_clean['RiskScore'] <= high_q)].copy()
        print(f"- Удалено выбросов: {initial_rows - len(train_clean)}")

    y = train_clean['RiskScore'].values
    train_features = train_clean.drop(columns=['RiskScore'])
    test_features = test_df.copy() if test_df is not None else None

    def split_dates(df):
        if df is None or 'ApplicationDate' not in df.columns:
            return df
        df = df.copy()
        df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'], errors='coerce')
        df['AppYear'] = df['ApplicationDate'].dt.year
        df['AppMonth'] = df['ApplicationDate'].dt.month
        df['AppWeek'] = df['ApplicationDate'].dt.isocalendar().week.astype(float)
        df['AppDay'] = df['ApplicationDate'].dt.day
        df['AppDayOfWeek'] = df['ApplicationDate'].dt.dayofweek
        df.drop(columns=['ApplicationDate'], inplace=True)
        return df

    train_features = split_dates(train_features)
    test_features = split_dates(test_features)

    train_features = create_features(train_features)
    test_features = create_features(test_features) if test_features is not None else None

    categorical_cols = train_features.select_dtypes(include=['object']).columns.tolist()
    print(f"- Категориальные признаки: {len(categorical_cols)}")

    numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
    medians = {col: train_features[col].median() for col in numeric_cols}
    train_features[numeric_cols] = train_features[numeric_cols].fillna(medians)
    if test_features is not None:
        for col in numeric_cols:
            if col in test_features.columns:
                test_features[col] = test_features[col].fillna(medians[col])

    # Легкая winsorization для числовых признаков (убираем экстремальные значения)
    quantiles = train_features[numeric_cols].quantile([0.01, 0.99])
    for col in numeric_cols:
        low, high = quantiles.loc[0.01, col], quantiles.loc[0.99, col]
        train_features[col] = train_features[col].clip(low, high)
        if test_features is not None and col in test_features.columns:
            test_features[col] = test_features[col].clip(low, high)

    skewed_cols = [col for col in numeric_cols if train_features[col].min() > 0 and train_features[col].skew() > 1]
    for col in skewed_cols:
        train_features[f"{col}_log"] = np.log1p(train_features[col])
        if test_features is not None and col in test_features.columns:
            test_features[f"{col}_log"] = np.log1p(np.clip(test_features[col], a_min=0, a_max=None))

    def fill_cats(df, cols):
        for col in cols:
            if col in df.columns:
                df[col] = df[col].fillna("missing")
        return df

    train_features = fill_cats(train_features, categorical_cols)
    if test_features is not None:
        test_features = fill_cats(test_features, categorical_cols)

    if test_features is not None:
        combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)
    else:
        combined = train_features.copy()

    combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

    if test_features is not None:
        train_encoded = combined.iloc[: len(train_features)].reset_index(drop=True)
        test_encoded = combined.iloc[len(train_features) :].reset_index(drop=True)
    else:
        train_encoded = combined
        test_encoded = None

    corr = train_encoded.corrwith(pd.Series(y)).abs().sort_values(ascending=False)
    top_features = corr.head(70).index.tolist()
    top20 = corr.head(25).index.tolist()
    top_interactions = corr.head(12).index.tolist()
    print(f"- Выбрано признаков по корреляции: {len(top_features)}")

    def build_feature_matrix(df_encoded):
        X_base = df_encoded[top_features].copy()

        for col in top_features:
            if (df_encoded[col] > 0).all() and not col.endswith("_log"):
                X_base[f"{col}_log"] = np.log1p(df_encoded[col])

        for col in top20:
            X_base[f"{col}_sq"] = df_encoded[col] ** 2

        for i in range(len(top_interactions)):
            for j in range(i + 1, len(top_interactions)):
                f1, f2 = top_interactions[i], top_interactions[j]
                X_base[f"{f1}_x_{f2}"] = df_encoded[f1] * df_encoded[f2]

        X_base = X_base.replace([np.inf, -np.inf], np.nan).fillna(0)
        return X_base

    X_train_df = build_feature_matrix(train_encoded)
    X_test_df = build_feature_matrix(test_encoded) if test_encoded is not None else None

    # Приводим к float для последующей нормализации
    X_train_df = X_train_df.astype(float)
    if X_test_df is not None:
        X_test_df = X_test_df.astype(float)

    print(f"- Матрица train: {X_train_df.shape[0]} x {X_train_df.shape[1]}")
    print(f"- RiskScore: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

    X = X_train_df.values
    X_test = X_test_df.values if X_test_df is not None else None
    train_processed = pd.concat([X_train_df, pd.Series(y, name="RiskScore")], axis=1)

    return X, y, X_test, train_processed
