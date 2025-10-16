from typing import List, Dict, Any

from sklearn.model_selection import train_test_split
import numpy as np


def train_test_val_split(x, y, sentences, groups, val_ratio=0.15, test_ratio=0.15):

    x_train, x_temp, y_train, y_temp, s_train, s_temp, groups_train, groups_temp = train_test_split(
        x, y, sentences, groups, test_size=val_ratio + test_ratio, random_state=42, shuffle=True
    )

    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    x_val, x_test, y_val, y_test, s_val, s_test, groups_val, groups_test = train_test_split(
        x_temp, y_temp, s_temp, groups_temp, test_size=1 - val_ratio_adjusted, random_state=42, shuffle=True
    )

    return (x_train, x_val, x_test, y_train, y_val, y_test, s_train, s_val, s_test,
            groups_train, groups_val, groups_test)


def normalize(array: np.ndarray) -> np.ndarray:
    return (array - array.min()) / (array.max() - array.min() + 1e-8)


def save_markdown_table(segments: List[Dict[str, Any]], path: str):
    if not segments:
        return
    keys = list(segments[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(keys) + " |\n")
        f.write("| " + " | ".join("---" for _ in keys) + " |\n")
        for s in segments:
            row = [str(s[k]) for k in keys]
            f.write("| " + " | ".join(row) + " |\n")
