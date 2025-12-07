import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(models_dict, save_path='convergence.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    has_gd, has_sgd = False, False
    for name, model in models_dict.items():
        if hasattr(model, 'loss_history') and len(model.loss_history) > 0:
            if 'GD' in name or 'Gradient' in name:
                axes[0].plot(model.loss_history, label=name, linewidth=2)
                has_gd = True
            elif 'SGD' in name:
                axes[1].plot(model.loss_history, label=name, linewidth=2)
                has_sgd = True
    
    axes[0].set_xlabel('Итерация')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Сходимость Gradient Descent')
    if has_gd:
        axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('Сходимость SGD (mini-batch)')
    if has_sgd:
        axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"График сходимости сохранен: {save_path}")
    plt.close()


def compare_normalizations(X, y, model_class, normalizers_dict):
    print("Сравнение нормализаций:")
    
    results = []
    
    for norm_name, normalizer in normalizers_dict.items():
        X_norm = normalizer.fit_transform(X)

        model = model_class()
        model.fit(X_norm, y)
        y_pred = model.predict(X_norm)
        mse = np.mean((y - y_pred) ** 2)
        
        results.append((norm_name, mse))
        print(f"- {norm_name}: mse={mse:.4f}")
        print(f"  mean={np.mean(X_norm, axis=0)[:3]}")
        print(f"  std={np.std(X_norm, axis=0)[:3]}")

    best = min(results, key=lambda x: x[1])
    print(f"Лучший метод: {best[0]} (mse={best[1]:.4f})")
    
    return results
