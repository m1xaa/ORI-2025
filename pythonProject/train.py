import argparse
import os
import joblib
import numpy as np
import constants
from data.TVSum.load_annotations import load_annotations
from evaluation import compute_bertscore
from featurization import align_segments_with_user_scores, make_feature_matrix
from text_processing import normalize_segments
from utils import train_test_val_split
from whisper_asr import transcribe_video
from sklearn.linear_model import Ridge


def collect_training_samples(
        dataset: str,
        root: str,
        whisper_model: str,
        language: str,
        embedding_model_name: str
):
    if dataset == "tvsum":
        annotations = load_annotations(root)
    else:
        raise NotImplementedError("SumME not implemented yet")
    videos_dir = os.path.join(root, constants.VIDEOS_DIR)

    all_features, all_labels, all_sentences = [], [], []
    for video_id, metadata in annotations.items():
        path_to_video = None
        for ext in constants.VIDEO_EXTENSIONS:
            path = os.path.join(videos_dir, video_id + ext)
            if os.path.exists(path):
                path_to_video = path
                break
        if path_to_video is None:
            print(f"[warn] Missing video file for {video_id}, skipping")
            continue

        segments = transcribe_video(path_to_video, whisper_model, language)
        segments = normalize_segments(segments, language)
        if not segments or len(segments) == 0 or all(s['text'].strip() == "" for s in segments):
            continue

        labels = align_segments_with_user_scores(segments, metadata['user_scores'], metadata['fps'])
        features = make_feature_matrix(segments, embedding_model_name)

        all_features.append(features)
        all_labels.append(labels)
        all_sentences.extend(s['text'] for s in segments)

    return all_features, all_labels, all_sentences


def train_regression(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model


def select_summary(sentences, scores, top_k):
    idx = np.argsort(scores)[::-1][:top_k]
    return [sentences[i] for i in idx]


def evaluate(model, X, y, sentences, top_k, lang):
    y_pred = model.predict(X)
    system_summary = select_summary(sentences, y_pred, top_k)
    reference_summary = select_summary(sentences, y, top_k)
    bert = compute_bertscore(system_summary, reference_summary, lang)
    return bert


def main():
    ap = argparse.ArgumentParser(description="Train supervised sentence ranker on TVSum/SumMe (Ridge only)")
    ap.add_argument("--dataset", required=True, choices=["tvsum", "summe"])
    ap.add_argument("--root", required=True, help="Dataset root directory")
    ap.add_argument("--whisper_model", default="small")
    ap.add_argument("--embedding_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--language", default="en")
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--top_k", type=int, default=5, help="Number of sentences in summary")
    args = ap.parse_args()

    x, y, sentences = collect_training_samples(args.dataset, args.root, args.whisper_model, args.language,
                                               args.embedding_model)
    x_train, x_val, x_test, y_train, y_val, y_test, s_train, s_val, s_test = train_test_val_split(x, y, sentences)

    best_alpha, best_score = None, 0
    for alpha in [i / 10 for i in range(1, 100)]:
        model = train_regression(x_train, y_train, alpha=alpha)
        bert = evaluate(model, x_val, y_val, s_val, args.top_k, args.language)
        if bert["f1"] > best_score:
            best_score = bert["f1"]
            best_alpha = alpha

    print(f"Best alpha (on val): {best_alpha}, BERT-F1={best_score:.4f}")

    x_train_val = np.vstack([x_train, x_val])
    y_train_val = np.concatenate([y_train, y_val])
    final_model = train_regression(x_train_val, y_train_val, alpha=best_alpha)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(final_model, args.model_out)
    print(f"Saved final model to {args.model_out}")

    rouge, bert = evaluate(final_model, x_test, y_test, s_test, args.top_k, args.language)
    print("Final Test ROUGE:", rouge)
    print("Final Test BERTScore:", bert)


if __name__ == '__main__':
    main()
