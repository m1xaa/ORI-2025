import numpy as np
import xgboost as xgb

from evaluation import compute_bertscore


def train_ranker(X: np.ndarray, y: np.ndarray, group):
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(group)
    params = {
        "objective": "rank:pairwise",
        "eta": 0.1,
        "max_depth": 6,
        "eval_metric": "ndcg",
        "verbosity": 1
    }
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model


def evaluate_model(model, x, y_true, sentences, group, top_k, lang):
    dmatrix = xgb.DMatrix(x)
    dmatrix.set_group(group)
    scores = model.predict(dmatrix)
    top_indices_pred = np.argsort(scores)[::-1][:top_k]
    system_summary = [sentences[i] for i in top_indices_pred]
    top_indices_ref = np.argsort(y_true)[::-1][:top_k]
    reference_summary = [sentences[i] for i in top_indices_ref]
    bert = compute_bertscore(system_summary, reference_summary, lang)
    return bert