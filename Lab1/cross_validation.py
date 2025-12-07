import numpy as np
import time


def k_fold_cross_validation(X, y, model_factory, k=5, random_state=42):
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.RandomState(random_state)
    rng.shuffle(idx)
    fold_sizes = np.full(k, n // k)
    fold_sizes[:n % k] += 1
    scores = []
    current = 0
    
    for fs in fold_sizes:
        start, end = current, current + fs
        val_idx = idx[start:end]
        train_idx = np.concatenate([idx[:start], idx[end:]])
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(np.mean((y_val - y_pred) ** 2))
        current = end
    
    return scores


def leave_one_out_cross_validation(X, y, model, max_samples=None):
    n_samples = X.shape[0] if max_samples is None else min(X.shape[0], max_samples)
    scores = []
    
    if max_samples and max_samples < X.shape[0]:
        print(f"LOO выполняется на {max_samples} из {X.shape[0]} объектов")
    
    for i in range(n_samples):
        X_val = X[i:i+1]
        y_val = y[i:i+1]
        X_train = np.vstack([X[:i], X[i+1:]])
        y_train = np.concatenate([y[:i], y[i+1:]])
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse_score = (y_val - y_pred) ** 2
        scores.append(mse_score[0])
    
    return scores


def evaluate_cross_validation(X, y, model_class, k=5, loo_samples=100):
    print("Кросс-валидация:")
    start_time = time.time()
    kfold_scores = k_fold_cross_validation(X, y, model_class, k=k)
    kfold_time = time.time() - start_time
    
    print(f"- K-fold (k={k}): mean_mse={np.mean(kfold_scores):.4f}, std={np.std(kfold_scores):.4f}, time={kfold_time:.2f}s")
    print(f"- K-fold значения: {[f'{s:.4f}' for s in kfold_scores]}")
    start_time = time.time()
    model_loo = model_class()
    loo_scores = leave_one_out_cross_validation(X, y, model_loo, max_samples=loo_samples)
    loo_time = time.time() - start_time
    
    print(f"- Leave-one-out (n={loo_samples}): mean_mse={np.mean(loo_scores):.4f}, std={np.std(loo_scores):.4f}, time={loo_time:.2f}s")
    
    return {
        'kfold': {'scores': kfold_scores, 'mean': np.mean(kfold_scores), 'std': np.std(kfold_scores)},
        'loo': {'scores': loo_scores, 'mean': np.mean(loo_scores), 'std': np.std(loo_scores)}
    }
