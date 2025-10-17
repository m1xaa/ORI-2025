import numpy as np
import xgboost as xgb


from scipy.stats import kendalltau

def normalized_discounted_cumulative_gain(y_true, y_pred, k):
    k = min(k, len(y_true))
    idx = np.argsort(-y_pred)[:k]
    gains = 2 ** y_true[idx] - 1
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains * discounts)

    idx_ideal = np.argsort(-y_true)[:k]
    gains_ideal = 2 ** y_true[idx_ideal] - 1
    idcg = np.sum(gains_ideal * discounts) + 1e-12
    return dcg / idcg


def evaluate_ranking_metrics(y_true, y_pred, groups, top_k=5):
    ndcgs = []
    start = 0

    for g in groups:
        end = start + g
        yt = y_true[start:end]
        yp = y_pred[start:end]

        ndcgs.append(normalized_discounted_cumulative_gain(yt, yp, top_k))
        start = end

    return {
        "ndcg@k": float(np.mean(ndcgs)),
    }


def evaluate_model(model, x, y_true, sentences, groups, top_k, lang):
    d_matrix = xgb.DMatrix(x)
    d_matrix.set_group(groups)
    scores = model.predict(d_matrix)

    metrics = evaluate_ranking_metrics(y_true, scores, groups, top_k)
    return metrics

