import argparse
import json
import os
import joblib
import numpy as np
from tqdm import tqdm
import constants
from data.TVSum.load_annotations import load_annotations
from evaluation import evaluate_model
from featurization import align_segments_with_user_scores, make_feature_matrix
from ranker import train_ranker
from text_processing import normalize_segments
from utils import train_test_val_split
from whisper_asr import transcribe_video


def collect_training_samples(
        root: str,
        whisper_model: str,
        language: str,
        embedding_model_name: str
):
    annotations = load_annotations(root)

    videos_dir = os.path.join(root, constants.VIDEOS_DIR)
    segments_dir = os.path.join(root, constants.SEGMENTS_CACHE)
    os.makedirs(segments_dir, exist_ok=True)

    all_features, all_labels, all_sentences, all_group_sizes = [], [], [], []

    for video_id, metadata in tqdm(annotations.items(), desc=f"Preparing sentences"):
        path_to_video = None
        for ext in constants.VIDEO_EXTENSIONS:
            path = os.path.join(videos_dir, video_id + ext)
            if os.path.exists(path):
                path_to_video = path
                break
        if path_to_video is None:
            print(f"[warn] Missing video file for {video_id}, skipping")
            continue

        cached_path = os.path.join(segments_dir, f"{video_id}_segments.json")
        if os.path.exists(cached_path):
            with open(cached_path, "r", encoding="utf-8") as f:
                segments = json.load(f)
        else:
            segments = normalize_segments(
                transcribe_video(path_to_video, whisper_model, language),
                language
            )
            if not segments or all(s['text'].strip() == "" for s in segments):
                continue

            with open(cached_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)

        labels = align_segments_with_user_scores(segments, metadata['user_scores'], metadata['fps'])
        features = make_feature_matrix(segments, embedding_model_name)

        all_features.append(features)
        all_labels.append(labels)
        all_sentences.extend(s['text'] for s in segments)
        all_group_sizes.extend([1] * len(segments))

    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)

    return all_features, all_labels, all_sentences, all_group_sizes


def main():
    ap = argparse.ArgumentParser(description="Train supervised sentence ranker on TVSum/SumMe (with hyperparameter tuning using nDCG@K)")
    ap.add_argument("--root", required=True, help="Dataset root directory")
    ap.add_argument("--whisper_model", default="small")
    ap.add_argument("--embedding_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--language", default="en")
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--top_k", type=int, default=5, help="Number of sentences in summary")
    args = ap.parse_args()

    x, y, sentences, sentence_lengths = collect_training_samples(
        args.root, args.whisper_model, args.language, args.embedding_model
    )
    x_train, x_val, x_test, y_train, y_val, y_test, s_train, s_val, s_test, groups_train, groups_val, groups_test = train_test_val_split(
        x, y, sentences, sentence_lengths
    )

    eta_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    max_depth_values = [2, 4, 6, 8, 10]
    num_boost_round_values = [50, 100, 150, 200]

    best_score = -1.0
    best_params = None

    for eta in eta_values:
        for max_depth in max_depth_values:
            for num_round in num_boost_round_values:
                params = {
                    "objective": "rank:pairwise",
                    "eta": eta,
                    "max_depth": max_depth,
                    "eval_metric": "ndcg",
                    "verbosity": 0
                }

                model = train_ranker(x_train, y_train, groups_train, params, num_boost_round=num_round)
                val_metrics = evaluate_model(model, x_val, y_val, s_val, groups_val, args.top_k, args.language)

                val_ndcg = val_metrics["ndcg@k"]
                val_tau = val_metrics["kendall_tau"]

                print(f"eta={eta}, depth={max_depth}, rounds={num_round} → nDCG@{args.top_k}={val_ndcg:.4f}, Kendall τ={val_tau:.4f}")

                if val_ndcg > best_score:
                    best_score = val_ndcg
                    best_params = (eta, max_depth, num_round)

    print(f"Best params: eta={best_params[0]}, depth={best_params[1]}, rounds={best_params[2]} → nDCG@{args.top_k}={best_score:.4f}")

    x_train_val = np.vstack([x_train, x_val])
    y_train_val = np.concatenate([y_train, y_val])
    groups_train_val = np.concatenate([groups_train, groups_val])

    final_params = {
        "objective": "rank:pairwise",
        "eta": best_params[0],
        "max_depth": best_params[1],
        "eval_metric": "ndcg",
        "verbosity": 1
    }

    final_model = train_ranker(
        x_train_val, y_train_val, groups_train_val,
        final_params, num_boost_round=best_params[2]
    )

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(final_model, args.model_out)
    print(f"Saved final model to {args.model_out}")

    test_metrics = evaluate_model(final_model, x_test, y_test, s_test, groups_test, args.top_k, args.language)
    print(f"Final Test Results: nDCG@{args.top_k}={test_metrics['ndcg@k']:.4f}, Kendall τ={test_metrics['kendall_tau']:.4f}")



if __name__ == '__main__':
    main()
