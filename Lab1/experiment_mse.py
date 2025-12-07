"""
Эксперименты с увеличением MSE
Различные способы контролируемого ухудшения качества модели
"""

import numpy as np
import pandas as pd
from preprocessing import preprocess_data
from feature_engineering import improve_preprocessing
from normalization import Normalizer
from linear_regression import LinearRegressionCustom
from sklearn.model_selection import train_test_split
from metrics import mse


def experiment_baseline(X, y):
    """Baseline - чистые данные"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    normalizer = Normalizer(method='z-score')
    X_train_norm = normalizer.fit_transform(X_train)
    X_val_norm = normalizer.transform(X_val)
    
    model = LinearRegressionCustom(method='analytical')
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_val_norm)
    return mse(y_val, y_pred)


def experiment_noise_y(X, y, noise_sigma=10.0):
    """Способ 1: Шум в целевой переменной"""
    y_noisy = y + np.random.normal(0, noise_sigma, size=y.shape)
    X_train, X_val, y_train, y_val = train_test_split(X, y_noisy, test_size=0.2, random_state=42)
    normalizer = Normalizer(method='z-score')
    X_train_norm = normalizer.fit_transform(X_train)
    X_val_norm = normalizer.transform(X_val)
    
    model = LinearRegressionCustom(method='analytical')
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_val_norm)
    return mse(y_val, y_pred)


def experiment_noise_X(X, y, noise_factor=0.5):
    """Способ 2: Шум в признаках (ЭФФЕКТИВНО)"""
    X_noisy = X + np.random.normal(0, noise_factor, size=X.shape) * X.std(axis=0)
    X_train, X_val, y_train, y_val = train_test_split(X_noisy, y, test_size=0.2, random_state=42)
    normalizer = Normalizer(method='z-score')
    X_train_norm = normalizer.fit_transform(X_train)
    X_val_norm = normalizer.transform(X_val)
    
    model = LinearRegressionCustom(method='analytical')
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_val_norm)
    return mse(y_val, y_pred)


def experiment_shuffle_y(X, y, shuffle_ratio=0.3):
    """Способ 3: Перемешивание части y"""
    y_shuffled = y.copy()
    n_shuffle = int(shuffle_ratio * len(y))
    idx = np.random.choice(len(y), size=n_shuffle, replace=False)
    y_shuffled[idx] = np.random.permutation(y_shuffled[idx])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y_shuffled, test_size=0.2, random_state=42)
    normalizer = Normalizer(method='z-score')
    X_train_norm = normalizer.fit_transform(X_train)
    X_val_norm = normalizer.transform(X_val)
    
    model = LinearRegressionCustom(method='analytical')
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_val_norm)
    return mse(y_val, y_pred)


def experiment_few_features(X, y, n_features=3):
    """Способ 4: Использовать мало признаков"""
    X_reduced = X[:, :n_features]
    X_train, X_val, y_train, y_val = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    normalizer = Normalizer(method='z-score')
    X_train_norm = normalizer.fit_transform(X_train)
    X_val_norm = normalizer.transform(X_val)
    
    model = LinearRegressionCustom(method='analytical')
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_val_norm)
    return mse(y_val, y_pred)


def experiment_bad_gd(X, y):
    """Способ 5: Плохие параметры градиентного спуска"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    normalizer = Normalizer(method='z-score')
    X_train_norm = normalizer.fit_transform(X_train)
    X_val_norm = normalizer.transform(X_val)
    
    model = LinearRegressionCustom(method='gradient_descent', learning_rate=1e-6, n_iterations=5)
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_val_norm)
    return mse(y_val, y_pred)


def experiment_zero_weights(X, y):
    """Способ 6: Обнулить веса после обучения"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    normalizer = Normalizer(method='z-score')
    X_train_norm = normalizer.fit_transform(X_train)
    X_val_norm = normalizer.transform(X_val)
    
    model = LinearRegressionCustom(method='analytical')
    model.fit(X_train_norm, y_train)
    model.weights[:] = 0
    model.bias = np.mean(y_train)
    y_pred = model.predict(X_val_norm)
    return mse(y_val, y_pred)


def main():
    print("="*70)
    print("ЭКСПЕРИМЕНТЫ С УВЕЛИЧЕНИЕМ MSE")
    print("="*70)
    
    # Загрузка данных
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    from eda import perform_eda
    _, train_clean = perform_eda(train_df, save_plots=False)
    X, y, _, _ = improve_preprocessing(train_clean, None)
    
    np.random.seed(42)
    
    print(f"\nИсходные данные:")
    print(f"  X shape: {X.shape}")
    print(f"  y mean: {y.mean():.2f}, std: {y.std():.2f}")
    
    # Эксперименты
    results = []
    
    print("\n" + "="*70)
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ")
    print("="*70)
    
    mse_baseline = experiment_baseline(X, y)
    results.append(("Baseline (чистые данные)", mse_baseline, 1.0))
    print(f"\n✓ Baseline MSE: {mse_baseline:.4f}")
    
    mse_noise_y = experiment_noise_y(X, y, noise_sigma=10.0)
    results.append(("Шум в y (σ=10)", mse_noise_y, mse_noise_y/mse_baseline))
    print(f"✓ Шум в y: {mse_noise_y:.4f} (×{mse_noise_y/mse_baseline:.2f})")
    
    mse_noise_X = experiment_noise_X(X, y, noise_factor=0.5)
    results.append(("Шум в X (factor=0.5)", mse_noise_X, mse_noise_X/mse_baseline))
    print(f"✓ Шум в X: {mse_noise_X:.4f} (×{mse_noise_X/mse_baseline:.2f})")
    
    mse_shuffle = experiment_shuffle_y(X, y, shuffle_ratio=0.3)
    results.append(("Перемешать 30% y", mse_shuffle, mse_shuffle/mse_baseline))
    print(f"✓ Перемешивание y: {mse_shuffle:.4f} (×{mse_shuffle/mse_baseline:.2f})")
    
    mse_few = experiment_few_features(X, y, n_features=3)
    results.append(("Только 3 признака", mse_few, mse_few/mse_baseline))
    print(f"✓ Мало признаков: {mse_few:.4f} (×{mse_few/mse_baseline:.2f})")
    
    mse_bad_gd = experiment_bad_gd(X, y)
    results.append(("Плохой GD (lr=1e-6, iter=5)", mse_bad_gd, mse_bad_gd/mse_baseline))
    print(f"✓ Плохой GD: {mse_bad_gd:.4f} (×{mse_bad_gd/mse_baseline:.2f})")
    
    mse_zero = experiment_zero_weights(X, y)
    results.append(("Нулевые веса", mse_zero, mse_zero/mse_baseline))
    print(f"✓ Нулевые веса: {mse_zero:.4f} (×{mse_zero/mse_baseline:.2f})")
    
    # Итоговая таблица
    print("\n" + "="*70)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("="*70)
    print(f"{'Метод':<35} {'MSE':<15} {'Прирост':<10}")
    print("-"*70)
    for name, mse_val, ratio in results:
        print(f"{name:<35} {mse_val:<15.4f} ×{ratio:<9.2f}")
    print("="*70)
    
    # Рекомендации
    print("\n📊 РЕКОМЕНДАЦИИ:")
    print("  🔥 Для умеренного роста MSE (×1.5-2): шум в X или перемешивание y")
    print("  🔥 Для сильного роста MSE (×3-5): мало признаков или плохой GD")
    print("  🔥 Для экстремального роста (×10+): нулевые веса")


if __name__ == "__main__":
    main()
