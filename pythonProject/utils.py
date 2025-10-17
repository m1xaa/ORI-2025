from typing import List, Dict, Any

from sklearn.model_selection import train_test_split
import numpy as np


def split_videos(video_ids, val_ratio=0.15, test_ratio=0.15):
    video_ids_train, video_ids_temp = train_test_split(
        video_ids,
        test_size=val_ratio + test_ratio,
        random_state=42,
        shuffle=True
    )
    adjusted_test_size = test_ratio / (val_ratio + test_ratio)
    video_ids_val, video_ids_test = train_test_split(
        video_ids_temp,
        test_size=adjusted_test_size,
        random_state=42,
        shuffle=True
    )

    return video_ids_train, video_ids_val, video_ids_test


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