import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from eda import perform_eda
from preprocessing import preprocess_data
from feature_engineering import improve_preprocessing
from normalization import Normalizer, apply_normalization
from linear_regression import LinearRegressionCustom, compare_models
from cross_validation import evaluate_cross_validation
from metrics import mse, mae, r2, mape, compare_metrics_with_sklearn
from visualization import plot_convergence, compare_normalizations

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def tune_ridge_alpha(X, y, alphas=None, k=5, random_state=42, clip_bounds=None):
    if alphas is None:
        alphas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    results = []
    for alpha in alphas:
        fold_mse = []
        model = LinearRegression() if alpha == 0 else Ridge(alpha=alpha)
        for train_idx, val_idx in kf.split(X):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            if clip_bounds is not None:
                pred = np.clip(pred, clip_bounds[0], clip_bounds[1])
            fold_mse.append(np.mean((y[val_idx] - pred) ** 2))
        results.append((alpha, np.mean(fold_mse)))
    best_alpha, best_mse = min(results, key=lambda x: x[1])
    print("Подбор alpha для Ridge:")
    for alpha, score in results:
        print(f"- alpha={alpha:.2f}, mse={score:.4f}")
    print(f"Лучший alpha: {best_alpha} (mse={best_mse:.4f})")
    return best_alpha


def tune_clip_quantile(X, y, q_values=None, alpha=0.01, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    if q_values is None:
        q_values = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    trials = []
    for q in q_values:
        bounds = np.quantile(y_train, [q, 1 - q])
        model = LinearRegressionCustom(method='analytical', alpha=alpha, clip_bounds=bounds)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        mse = np.mean((y_val - pred) ** 2)
        trials.append((q, bounds, mse))
    best_q, best_bounds, best_mse = min(trials, key=lambda x: x[2])
    print("Подбор квантилей клиппинга:")
    for q, bounds, mse in trials:
        print(f"- q={q:.3f}, bounds=({bounds[0]:.2f}, {bounds[1]:.2f}), mse={mse:.4f}")
    print(f"Лучший q: {best_q:.3f}, mse={best_mse:.4f}")
    full_bounds = np.quantile(y, [best_q, 1 - best_q])
    return full_bounds, best_q


def main():
    print("Шаг 1. Загрузка данных")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print(f"- Train: {train_df.shape}, test: {test_df.shape}")
    
    print("Шаг 2. EDA")
    correlations, train_clean = perform_eda(train_df, save_plots=True)
    
    print("Шаг 3. Предобработка")
    X, y, X_test, train_processed = improve_preprocessing(train_df, test_df)
    print(f"- X: {X.shape}, y: {y.shape}, X_test: {X_test.shape if X_test is not None else None}")
    clip_bounds = np.quantile(y, [0.02, 0.98])
    print(f"- Клиппинг предсказаний: [{clip_bounds[0]:.2f}, {clip_bounds[1]:.2f}]")
    
    print("Шаг 4. Нормализация")
    X_zscore, zscore_norm = apply_normalization(X, method='z-score')
    X_minmax, minmax_norm = apply_normalization(X, method='min-max')
    norm_results = compare_normalizations(
        X, y,
        lambda: LinearRegressionCustom(method='analytical', alpha=1.0),
        {'Z-Score': zscore_norm, 'Min-Max': minmax_norm}
    )
    X_normalized = X_zscore
    
    print("Шаг 5. Подбор alpha для Ridge")
    best_alpha = tune_ridge_alpha(X_normalized, y, clip_bounds=clip_bounds)

    print("Шаг 6. Обучение моделей")
    results, best_model = compare_models(X_normalized, y, alpha=best_alpha, clip_bounds=clip_bounds)
    models_dict = {name: model for name, _, _, model in results if hasattr(model, 'loss_history')}
    if models_dict:
        plot_convergence(models_dict, 'convergence.png')
    
    print("Шаг 7. Кросс-валидация и метрики")
    cv_results = evaluate_cross_validation(
        X_normalized, y, 
        lambda: LinearRegressionCustom(method='analytical', alpha=best_alpha, clip_bounds=clip_bounds),
        k=5, loo_samples=200
    )
    
    from sklearn.model_selection import train_test_split
    X_train_val, X_val_test, y_train_val, y_val_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )
    temp_model = LinearRegressionCustom(method='analytical', alpha=best_alpha, clip_bounds=clip_bounds)
    temp_model.fit(X_train_val, y_train_val)
    y_pred_val = temp_model.predict(X_val_test)
    metrics_comparison = compare_metrics_with_sklearn(y_val_test, y_pred_val)

    print("Шаг 8. Submission (линейная регрессия)")
    if X_test is not None:
        final_model = LinearRegressionCustom(method='analytical', alpha=best_alpha, clip_bounds=clip_bounds)
        final_model.fit(X_normalized, y)
        X_test_normalized = zscore_norm.transform(X_test)
        predictions = final_model.predict(X_test_normalized)
        submission = pd.DataFrame({
            'ID': range(len(predictions)),
            'RiskScore': predictions
        })
        submission.to_csv('submission.csv', index=False)
        print(f"- Обучено на {X_normalized.shape[0]} объектах, предсказано {len(predictions)} значений")
        print(f"- Предсказания: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
    else:
        print("- X_test не найден")
    
    print(f"Кросс-валидация (k-fold) mse: {cv_results['kfold']['mean']:.4f} ± {cv_results['kfold']['std']:.4f}")
    print("Submission: submission.csv")


if __name__ == "__main__":
    main()
