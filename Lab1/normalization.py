import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Normalizer:
    def __init__(self, method='z-score'):
        self.method = method
        self.params = {}
    
    def fit(self, X):
        if self.method == 'z-score':
            self.params['mean'] = np.mean(X, axis=0)
            self.params['std'] = np.std(X, axis=0)
        elif self.method == 'min-max':
            self.params['min'] = np.min(X, axis=0)
            self.params['max'] = np.max(X, axis=0)
        return self
    
    def transform(self, X):
        if self.method == 'z-score':
            return (X - self.params['mean']) / (self.params['std'] + 1e-8)
        elif self.method == 'min-max':
            return (X - self.params['min']) / (self.params['max'] - self.params['min'] + 1e-8)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


def apply_normalization(X, method='z-score'):
    print(f"Нормализация: {method}")
    normalizer = Normalizer(method=method)
    X_normalized = normalizer.fit_transform(X)
    
    print(f"- До: mean={np.mean(X, axis=0)[:3]}, std={np.std(X, axis=0)[:3]}")
    print(f"- После: mean={np.mean(X_normalized, axis=0)[:3]}, std={np.std(X_normalized, axis=0)[:3]}")
    
    return X_normalized, normalizer
