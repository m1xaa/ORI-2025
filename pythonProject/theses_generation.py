from typing import List, Dict, Any
import joblib
import numpy as np
from featurization import make_feature_matrix
import xgboost as xgb
from transformers import pipeline


def generate_theses(
    segments: List[Dict[str, Any]],
    top_k: int,
    embedding_model: str,
    path_to_model: str,
    generative_model: str = None
):
    try:
        model = joblib.load(path_to_model)
        features = make_feature_matrix(segments, embedding_model)
        d_matrix = xgb.DMatrix(features)
        group = [len(segments)]
        d_matrix.set_group(group)

        scores = model.predict(d_matrix)
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_segments = [segments[i] for i in top_indices]

        if generative_model:
            summarizer = pipeline("summarization", model=generative_model)
            summarized_segments = []
            for seg in top_segments:
                try:
                    summary = summarizer(seg["text"], max_length=40, min_length=10, do_sample=False)[0]["summary_text"]
                except Exception:
                    summary = seg["text"]

                summarized_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": summary
                })
            return summarized_segments

        return top_segments

    except IOError as e:
        print(e)
        return []
