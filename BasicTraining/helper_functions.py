import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import pandas as pd

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_wine_quality_dataset():
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    red_data = pd.read_csv(red_url, sep=';')
    white_data = pd.read_csv(white_url, sep=';')
    data = pd.concat([red_data, white_data], ignore_index=True)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y_remapped = np.zeros_like(y)
    y_remapped[y <= 4] = 0
    y_remapped[(y >= 5) & (y <= 6)] = 1
    y_remapped[y >= 7] = 2
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y_remapped

def load_dataset(dataset_name):
    if dataset_name == "wine_quality":
        return load_wine_quality_dataset()
    raise ValueError(f"Unknown dataset: {dataset_name}")

def prepare_data(X, y, train_size=0.8, random_state=42):
    return train_test_split(X, y, train_size=train_size, random_state=random_state, stratify=y) 