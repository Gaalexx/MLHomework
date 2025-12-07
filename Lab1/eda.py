import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(train_df, save_plots=False):
    print("EDA:")
    train_clean = train_df.copy()
    if 'RiskScore' in train_clean.columns:
        outliers = (train_clean['RiskScore'] < -1000) | (train_clean['RiskScore'] > 1000)
        print(f"- Выбросы в RiskScore: {outliers.sum()}")
        train_clean = train_clean[~outliers]
    
    print(f"- Размер: {train_clean.shape[0]} объектов, {train_clean.shape[1] - 1} признаков без цели")
    missing = train_clean.isnull().sum()
    print(f"- Пропусков всего: {missing.sum()}")
    numeric_cols = train_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_clean.select_dtypes(include=['object']).columns.tolist()
    print(f"- Числовые признаки: {len(numeric_cols)}")
    print(f"- Категориальные признаки: {len(categorical_cols)}")
    
    if 'RiskScore' in train_clean.columns:
        stats = train_clean['RiskScore'].describe()
        print(f"- RiskScore: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(train_clean['RiskScore'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].set_xlabel('RiskScore', fontsize=12)
    axes[0].set_ylabel('Частота', fontsize=12)
    axes[0].set_title('Распределение RiskScore', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(train_clean['RiskScore'])
    axes[1].set_ylabel('RiskScore', fontsize=12)
    axes[1].set_title('Boxplot RiskScore', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('eda_target_variable.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    numeric_df = train_clean.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['RiskScore'].sort_values(ascending=False)
    print("Топ признаков по корреляции с RiskScore:")
    for feat, corr in correlations.abs().sort_values(ascending=False).head(6)[1:].items():
        print(f"- {feat}: {correlations[feat]:+.4f}")
    
    top_features = correlations.abs().sort_values(ascending=False).head(11).index.tolist()
    
    plt.figure(figsize=(12, 10))
    corr_matrix = numeric_df[top_features].corr()
    sns.heatmap(corr_matrix, 
                annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={'label': 'Корреляция'})
    plt.title('Матрица корреляций (топ-10 признаков по |корреляции|)', fontsize=14, pad=20, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        plt.savefig('eda_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    top_5_features = correlations.abs().sort_values(ascending=False).head(6).index.tolist()[1:]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_5_features):
        mask = train_clean[[feature, 'RiskScore']].notna().all(axis=1)
        data = train_clean[mask]
        
        axes[idx].scatter(data[feature], data['RiskScore'], 
                         alpha=0.3, s=10, color='steelblue')
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('RiskScore', fontsize=10)
        axes[idx].set_title(f'{feature} vs RiskScore\n(corr={correlations[feature]:.3f})', 
                           fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    # Удаляем лишний subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('eda_feature_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    if save_plots:
        print("Сохранены графики: eda_target_variable.png, eda_correlation_matrix.png, eda_feature_relationships.png")
    
    return correlations, train_clean


if __name__ == "__main__":
    train_df = pd.read_csv('train.csv')
    correlations = perform_eda(train_df, save_plots=True)
