"""
Утилита для создания submission файла для Kaggle
"""
import pandas as pd


def create_submission(predictions, test_ids, filename='submission.csv'):
    """
    Создает submission файл в формате Kaggle
    
    Args:
        predictions: numpy array с предсказаниями RiskScore
        test_ids: numpy array или list с ID из test.csv
        filename: имя выходного файла
    """
    submission = pd.DataFrame({
        'ID': test_ids,
        'RiskScore': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Submission файл сохранен: {filename}")
    print(f"Количество предсказаний: {len(predictions)}")
    print(f"\nПервые 5 строк:")
    print(submission.head())


if __name__ == "__main__":
    # Пример использования
    import numpy as np
    
    # Создаем пример submission файла
    test_ids = range(5000)
    predictions = np.random.uniform(20, 80, 5000)
    
    create_submission(predictions, test_ids, 'example_submission.csv')
