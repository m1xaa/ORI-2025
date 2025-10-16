import numpy as np
import xgboost as xgb


def train_ranker(x: np.ndarray, y: np.ndarray, group, params=None, num_boost_round=100):
    d_train = xgb.DMatrix(x, label=y)
    d_train.set_group(group)

    if params is None:
        params = {
            "objective": "rank:pairwise",
            "eta": 0.1,
            "max_depth": 6,
            "eval_metric": "ndcg",
            "verbosity": 0
        }

    model = xgb.train(params, d_train, num_boost_round=num_boost_round)
    return model