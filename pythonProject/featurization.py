import numpy as np


def align_segments_with_user_scores(
    segments,
    user_scores: np.ndarray,
    fps: float
):
    average_frame_importance = user_scores.mean(axis=1)
    min, max = average_frame_importance.min(), average_frame_importance.max()
    normalized_frame_importance = (average_frame_importance - min) / (max - min + 1e-8)

    labels = []
    number_of_frames = len(normalized_frame_importance)
    for segment in segments:
        start = segment['start']
        end = segment['end']
        start_idx = max(0, round(start * fps))
        end_idx = min(round(end * fps), number_of_frames-1)
        segment_window = normalized_frame_importance[start_idx:end_idx]
        if segment_window.size == 0:
            labels.append(0.0)
        else:
            labels.append(segment_window.mean())

    return np.array(labels, dtype=float)