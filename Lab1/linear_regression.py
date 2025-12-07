import numpy as np
import time
from sklearn.model_selection import train_test_split


class LinearRegressionCustom:
    def __init__(self, method='analytical', learning_rate=0.01, n_iterations=1000, 
                 batch_size=32, early_stopping=True, tol=1e-4, alpha=0.0, clip_bounds=None):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.tol = tol
        self.alpha = alpha
        self.clip_bounds = clip_bounds
        self.weights = None
        self.bias = None
        self.training_time = 0
        self.loss_history = []
    
    def fit(self, X, y):
        start_time = time.time()
        
        if self.method == 'analytical':
            self._fit_analytical(X, y)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X, y)
        elif self.method == 'sgd':
            self._fit_sgd(X, y)
        
        self.training_time = time.time() - start_time
        return self
    
    def _fit_analytical(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        if self.alpha > 0:
            XtX = X_b.T @ X_b
            reg_matrix = self.alpha * np.eye(XtX.shape[0])
            reg_matrix[0, 0] = 0
            theta = np.linalg.inv(XtX + reg_matrix) @ X_b.T @ y
        else:
            theta = np.linalg.pinv(X_b) @ y
        
        self.bias = float(theta[0])
        self.weights = theta[1:].flatten()
    
    def _fit_gradient_descent(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        for iteration in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            loss = np.mean((y_pred - y) ** 2)
            
            if not np.isfinite(loss):
                break
            
            self.loss_history.append(loss)
            
            dw = (2/n_samples) * X.T @ (y_pred - y)
            db = (2/n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if self.early_stopping and iteration > 10:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                    break
    
    def _fit_sgd(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        for iteration in range(self.n_iterations):
            perm = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[perm], y[perm]
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                y_pred = X_batch @ self.weights + self.bias
                batch_size_actual = len(X_batch)
                dw = (2/batch_size_actual) * X_batch.T @ (y_pred - y_batch)
                db = (2/batch_size_actual) * np.sum(y_pred - y_batch)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            y_pred_full = X @ self.weights + self.bias
            loss = np.mean((y_pred_full - y) ** 2)
            
            if not np.isfinite(loss):
                break
            
            self.loss_history.append(loss)
            
            if self.early_stopping and iteration > 10:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                    break
    
    def predict(self, X):
        if self.weights is None:
            raise ValueError("Модель не обучена. Вызовите fit() перед predict().")
        y_pred = X @ self.weights + self.bias
        if self.clip_bounds is not None:
            low, high = self.clip_bounds
            y_pred = np.clip(y_pred, low, high)
        return y_pred


def tune_hyperparameters(X_train, y_train, X_val, y_val, method='gradient_descent', clip_bounds=None):
    learning_rates = [0.0001, 0.001, 0.01] if method == 'gradient_descent' else [0.0001, 0.001, 0.01]
    best_lr, best_mse = None, float('inf')
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        for lr in learning_rates:
            model = LinearRegressionCustom(
                method=method,
                learning_rate=lr,
                n_iterations=1000,
                clip_bounds=clip_bounds,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = np.mean((y_val - y_pred) ** 2)
            
            if not np.isfinite(mse):
                continue
                
            if mse < best_mse:
                best_mse, best_lr = mse, lr
    
    return best_lr if best_lr is not None else 0.001


def analyze_feature_importance(model, feature_names=None):
    if model.weights is None:
        return
    
    weights_abs = np.abs(model.weights)
    top_indices = np.argsort(weights_abs)[-10:][::-1]
    
    print("Важные признаки:")
    for idx in top_indices[:5]:
        feat_name = f"Feature_{idx}" if feature_names is None else feature_names[idx]
        print(f"- {feat_name}: {model.weights[idx]:.4f}")


def compare_models(X, y, test_size=0.2, random_state=42, alpha=1.0, clip_bounds=None):
    print("Сравнение методов линейной регрессии:")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"- Train: {X_train.shape[0]}, val: {X_val.shape[0]}")
    
    results = []
    
    model_analytical = LinearRegressionCustom(method='analytical', alpha=alpha, clip_bounds=clip_bounds)
    model_analytical.fit(X_train, y_train)
    y_pred = model_analytical.predict(X_val)
    mse_analytical = np.mean((y_val - y_pred) ** 2)
    print(f"[Analytical ridge α={alpha}] mse={mse_analytical:.4f}, time={model_analytical.training_time:.4f}s")
    analyze_feature_importance(model_analytical)
    results.append(('Analytical', mse_analytical, model_analytical.training_time, model_analytical))
    
    best_lr_gd = tune_hyperparameters(
        X_train, y_train, X_val, y_val, 'gradient_descent', clip_bounds=clip_bounds
    )
    model_gd = LinearRegressionCustom(
        method='gradient_descent',
        learning_rate=best_lr_gd,
        n_iterations=1000,
        clip_bounds=clip_bounds,
    )
    model_gd.fit(X_train, y_train)
    y_pred = model_gd.predict(X_val)
    mse_gd = np.mean((y_val - y_pred) ** 2)
    print(f"[Gradient descent] lr={best_lr_gd}, mse={mse_gd:.4f}, time={model_gd.training_time:.4f}s, steps={len(model_gd.loss_history)}")
    results.append(('Gradient Descent', mse_gd, model_gd.training_time, model_gd))
    
    best_lr_sgd = tune_hyperparameters(
        X_train, y_train, X_val, y_val, 'sgd', clip_bounds=clip_bounds
    )
    model_sgd = LinearRegressionCustom(
        method='sgd',
        learning_rate=best_lr_sgd,
        n_iterations=100,
        batch_size=32,
        clip_bounds=clip_bounds,
    )
    model_sgd.fit(X_train, y_train)
    y_pred = model_sgd.predict(X_val)
    mse_sgd = np.mean((y_val - y_pred) ** 2)
    print(f"[SGD] lr={best_lr_sgd}, mse={mse_sgd:.4f}, time={model_sgd.training_time:.4f}s, steps={len(model_sgd.loss_history)}")
    results.append(('SGD', mse_sgd, model_sgd.training_time, model_sgd))
    
    from sklearn.linear_model import Ridge as SklearnRidge
    start_time = time.time()
    model_sklearn = SklearnRidge(alpha=alpha)
    model_sklearn.fit(X_train, y_train)
    sklearn_time = time.time() - start_time
    y_pred_sk = model_sklearn.predict(X_val)
    if clip_bounds is not None:
        y_pred_sk = np.clip(y_pred_sk, clip_bounds[0], clip_bounds[1])
    mse_sklearn = np.mean((y_val - y_pred_sk) ** 2)
    diff = abs(mse_analytical - mse_sklearn)
    print(f"[Sklearn Ridge] mse={mse_sklearn:.4f}, time={sklearn_time:.4f}s, разница с analytical={diff:.6f}")
    results.append(('Sklearn Ridge', mse_sklearn, sklearn_time, model_sklearn))
    
    print("Сводка по MSE и времени:")
    for name, mse_val, train_time, _ in results:
        print(f"- {name}: mse={mse_val:.4f}, time={train_time:.4f}s")
    
    best_method = min(results, key=lambda x: x[1])
    print(f"Лучший метод: {best_method[0]} (mse={best_method[1]:.4f})")
    
    return results, best_method[3]
