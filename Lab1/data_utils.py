import pandas as pd
import numpy as np
from datetime import datetime


def load_data(train_path='train.csv', test_path='test.csv'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def preprocess_data(df, is_train=True):
    df = df.copy()
    
    if 'ApplicationDate' in df.columns:
        df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'])
        df['ApplicationYear'] = df['ApplicationDate'].dt.year
        df['ApplicationMonth'] = df['ApplicationDate'].dt.month
        df['ApplicationDay'] = df['ApplicationDate'].dt.day
        df = df.drop('ApplicationDate', axis=1)
    
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    if is_train and 'RiskScore' in df.columns:
        y = df['RiskScore'].values
        X = df.drop('RiskScore', axis=1)
    else:
        y = None
        X = df
    
    categorical_columns = X.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    feature_names = X.columns.tolist()
    X = X.values
    
    return X, y, feature_names


def get_numeric_features(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_features(df):
    return df.select_dtypes(include=['object']).columns.tolist()
