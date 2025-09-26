from typing import List, Dict, Any

import joblib
import numpy as np
from featurization import make_feature_matrix
import xgboost as xgb


def generate_theses(
    segments: List[Dict[str, Any]],
    top_k: int,
    embedding_model: str,
    path_to_model: str
):
    try:
        model = joblib.load(path_to_model)
        features = make_feature_matrix(segments, embedding_model)
        dmatrix = xgb.DMatrix(features)
        group = [len(segments)]
        dmatrix.set_group(group)
        scores = model.predict(dmatrix)
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_texts = [segments[i] for i in top_indices]

        return top_texts

    except IOError as e:
        print(e)
        return []




