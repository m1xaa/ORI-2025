from sklearn.model_selection import train_test_split
import numpy as np


def train_test_val_split(x, y, sentences):
    x_train, x_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        x, y, sentences, test_size=0.30, random_state=42, shuffle=True
    )
    x_val, x_test, y_val, y_test, s_val, s_test = train_test_split(
        x_temp, y_temp, s_temp, test_size=0.5, random_state=42, shuffle=True
    )

    return x_train, x_val, x_test, y_train, y_val, y_test, s_train, s_val, s_test

def normalize(array: np.ndarray) -> np.ndarray:
    return (array - array.min()) / (array.max() - array.min() + 1e-8)