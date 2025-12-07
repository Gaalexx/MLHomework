import numpy as np
import pandas as pd


def improved_outlier_detection(df, target_col='RiskScore'):
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
    print(f"Выбросов обнаружено: {outliers.sum()} ({outliers.mean()*100:.1f}%)")
    return df[~outliers]


def improved_categorical_encoding(df, categorical_cols):
    for col in categorical_cols:
        if df[col].nunique() > 10:
            value_counts = df[col].value_counts()
            top_categories = value_counts.head(9).index
            df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
    
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


def remove_multicollinearity(df, target_col='RiskScore', threshold=0.9):
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = []
    for column in upper_triangle.columns:
        if any(upper_triangle[column] > threshold):
            corr_with_target = abs(df[column].corr(df[target_col]))
            other_cols = upper_triangle[column][upper_triangle[column] > threshold].index
            
            keep_column = column
            for other_col in other_cols:
                other_corr = abs(df[other_col].corr(df[target_col]))
                if other_corr > corr_with_target:
                    keep_column = other_col
                    corr_with_target = other_corr
            
            if keep_column != column:
                to_drop.append(column)
            else:
                to_drop.extend([col for col in other_cols if col not in to_drop])
    
    return df.drop(columns=to_drop), to_drop


def preprocess_data(train_df, test_df=None):
    print("Предобработка данных:")
    
    train_clean = train_df.copy()
    if 'RiskScore' in train_clean.columns:
        train_clean = improved_outlier_detection(train_clean)
    
    numeric_cols = train_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_clean.select_dtypes(include=['object']).columns.tolist()
    
    if 'RiskScore' in train_clean.columns:
        train_clean = train_clean.dropna(subset=['RiskScore']).copy()
    
    if categorical_cols:
        train_clean = improved_categorical_encoding(train_clean, categorical_cols)
    
    feature_cols = [col for col in train_clean.columns if col != 'RiskScore']
    medians = {col: train_clean[col].median() for col in feature_cols}
    train_clean = train_clean.fillna(medians)
    
    print(f"- Размер после обработки: {train_clean.shape}")
    print(f"- Пропусков осталось: {train_clean.isnull().sum().sum()}")
    
    X = train_clean.drop('RiskScore', axis=1).values
    y = train_clean['RiskScore'].values
    
    X_test = None
    if test_df is not None:
        test_clean = test_df.copy()
        
        if categorical_cols:
            test_clean = pd.get_dummies(test_clean, columns=[c for c in categorical_cols if c in test_clean.columns], drop_first=True)
        
        train_cols = [c for c in train_clean.columns if c != 'RiskScore']
        for col in train_cols:
            if col not in test_clean.columns:
                test_clean[col] = 0
        test_clean = test_clean[train_cols]
        
        for col in test_clean.columns:
            if col in medians:
                test_clean[col].fillna(medians[col], inplace=True)
            else:
                test_clean[col].fillna(test_clean[col].median(), inplace=True)
        
        X_test = test_clean.values
        print(f"- Test размер после обработки: {X_test.shape}")
    
    return X, y, X_test, train_clean


def improve_preprocessing(train_df, test_df=None):
    print("Предобработка (упрощенная):")
    
    train_clean = train_df.copy()
    if 'RiskScore' in train_clean.columns:
        outliers = (train_clean['RiskScore'] < 0) | (train_clean['RiskScore'] > 150)
        train_clean = train_clean[~outliers].copy()
        print(f"- Удалено выбросов: {outliers.sum()}")
    
    categorical_cols = train_clean.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"- Кодируем категориальные признаки: {len(categorical_cols)}")
        for col in categorical_cols:
            if col in train_clean.columns:
                train_clean[col] = train_clean[col].astype('category').cat.codes
    
    numeric_cols = train_clean.select_dtypes(include=[np.number]).columns.tolist()
    train_clean = train_clean[numeric_cols].copy()
    
    train_clean, dropped_features = remove_multicollinearity(train_clean, threshold=0.9)
    print(f"- Удалено признаков из-за корреляции: {len(dropped_features)}")
    
    if 'RiskScore' in train_clean.columns:
        before = len(train_clean)
        train_clean = train_clean.dropna(subset=['RiskScore']).copy()
        print(f"- Удалено строк без RiskScore: {before - len(train_clean)}")
    
    if categorical_cols:
        train_clean = improved_categorical_encoding(train_clean, categorical_cols)
    
    feature_cols = [col for col in train_clean.columns if col != 'RiskScore']
    medians = {col: train_clean[col].median() for col in feature_cols}
    train_clean = train_clean.fillna(medians)
    
    print(f"- Размер после обработки: {train_clean.shape}")
    print(f"- Пропусков осталось: {train_clean.isnull().sum().sum()}")
    
    X = train_clean.drop('RiskScore', axis=1).values
    y = train_clean['RiskScore'].values
    
    X_test = None
    if test_df is not None:
        test_clean = test_df.copy()
        
        if categorical_cols:
            test_clean = pd.get_dummies(test_clean, columns=[c for c in categorical_cols if c in test_clean.columns], drop_first=True)
        
        train_cols = [c for c in train_clean.columns if c != 'RiskScore']
        for col in train_cols:
            if col not in test_clean.columns:
                test_clean[col] = 0
        test_clean = test_clean[train_cols]
        
        for col in test_clean.columns:
            if col in medians:
                test_clean[col].fillna(medians[col], inplace=True)
            else:
                test_clean[col].fillna(test_clean[col].median(), inplace=True)
        
        X_test = test_clean.values
        print(f"- Test размер после обработки: {X_test.shape}")
    
    return X, y, X_test, train_clean
