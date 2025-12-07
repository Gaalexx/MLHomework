import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

from feature_engineering import improve_preprocessing
from normalization import apply_normalization
from linear_regression import LinearRegressionCustom
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X, y, X_test, _ = improve_preprocessing(train_df, test_df)

X_norm, norm = apply_normalization(X, method='z-score')

print("Проверка MSE:")
model = LinearRegressionCustom(method='analytical', alpha=1.0)
model.fit(X_norm, y)
y_pred = model.predict(X_norm)
mse_train = np.mean((y - y_pred) ** 2)
print(f"- MSE train (переобучение): {mse_train:.4f}")

X_train, X_val, y_train, y_val = train_test_split(X_norm, y, test_size=0.2, random_state=42)
model2 = LinearRegressionCustom(method='analytical', alpha=1.0)
model2.fit(X_train, y_train)
y_pred_val = model2.predict(X_val)
mse_val = np.mean((y_val - y_pred_val) ** 2)
print(f"- MSE валидация: {mse_val:.4f}")

from cross_validation import evaluate_cross_validation
cv_results = evaluate_cross_validation(X_norm, y, lambda: LinearRegressionCustom(method='analytical', alpha=1.0), k=5, loo_samples=0)
print(f"- MSE 5-fold: {cv_results['kfold']['mean']:.4f} ± {cv_results['kfold']['std']:.4f}")
print("Итог: использована линейная регрессия и честные оценки через k-fold")
