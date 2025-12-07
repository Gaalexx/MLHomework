"""Быстрая проверка MSE с Ridge регуляризацией"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Загрузка
train_df = pd.read_csv('train.csv')

# Очистка выбросов
q_low = train_df['RiskScore'].quantile(0.01)
q_high = train_df['RiskScore'].quantile(0.99)
train_clean = train_df[(train_df['RiskScore'] > q_low) & (train_df['RiskScore'] < q_high)].copy()

# Кодирование категориальных
for col in train_clean.select_dtypes(include=['object']).columns:
    train_clean[col] = train_clean[col].astype('category').cat.codes

# Удаление пропусков
train_clean = train_clean.dropna()

# Отбор топ признаков
X_base = train_clean.drop(['RiskScore'], axis=1).select_dtypes(include=[np.number])
y = train_clean['RiskScore']

# Корреляция с целевой
corr = X_base.corrwith(y).abs().sort_values(ascending=False)
top_features = corr.head(25).index.tolist()
X = X_base[top_features].copy()

# Логарифмы для всех положительных
for col in top_features:
    if (X[col] > 0).all():
        X[f'{col}_log'] = np.log1p(X[col])

# Квадраты для топ-20
for col in corr.head(20).index:
    X[f'{col}_sq'] = X[col] ** 2

# Взаимодействия топ-10
top10 = corr.head(10).index.tolist()
for i in range(len(top10)):
    for j in range(i+1, len(top10)):
        X[f'{top10[i]}_x_{top10[j]}'] = X[top10[i]] * X[top10[j]]

# Сплит
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Ridge с разными alpha
for alpha in [1, 5, 10, 20, 50]:
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Ridge (α={alpha:3d}): MSE = {mse:.4f}")
