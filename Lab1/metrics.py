import numpy as np
from sklearn.metrics import mean_squared_error as sklearn_mse
from sklearn.metrics import mean_absolute_error as sklearn_mae
from sklearn.metrics import r2_score as sklearn_r2


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def mape(y_true, y_pred):
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compare_metrics_with_sklearn(y_true, y_pred):
    custom_mse = mse(y_true, y_pred)
    sklearn_mse_val = sklearn_mse(y_true, y_pred)
    mse_diff = abs(custom_mse - sklearn_mse_val)
    
    custom_mae = mae(y_true, y_pred)
    sklearn_mae_val = sklearn_mae(y_true, y_pred)
    mae_diff = abs(custom_mae - sklearn_mae_val)
    
    custom_r2 = r2(y_true, y_pred)
    sklearn_r2_val = sklearn_r2(y_true, y_pred)
    r2_diff = abs(custom_r2 - sklearn_r2_val)
    
    custom_mape = mape(y_true, y_pred)
    
    print("Сравнение метрик с sklearn:")
    print(f"- MSE: custom={custom_mse:.6f}, sklearn={sklearn_mse_val:.6f}, diff={mse_diff:.10f}")
    print(f"- MAE: custom={custom_mae:.6f}, sklearn={sklearn_mae_val:.6f}, diff={mae_diff:.10f}")
    print(f"- R2: custom={custom_r2:.6f}, sklearn={sklearn_r2_val:.6f}, diff={r2_diff:.10f}")
    print(f"- MAPE: custom={custom_mape:.6f}%")
    
    return {
        'mse': {'custom': custom_mse, 'sklearn': sklearn_mse_val, 'diff': mse_diff},
        'mae': {'custom': custom_mae, 'sklearn': sklearn_mae_val, 'diff': mae_diff},
        'r2': {'custom': custom_r2, 'sklearn': sklearn_r2_val, 'diff': r2_diff},
        'mape': {'custom': custom_mape}
    }
