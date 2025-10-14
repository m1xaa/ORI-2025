import argparse
import os
import joblib
import numpy as np
from tqdm import tqdm
import constants
from data.TVSum.load_annotations import load_annotations
from featurization import align_segments_with_user_scores, make_feature_matrix
from ranker import train_ranker, evaluate_model
from text_processing import normalize_segments
from utils import train_test_val_split
from whisper_asr import transcribe_video


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

    all_features, all_labels, all_sentences, all_group_sizes = [], [], [], []
    for video_id, metadata in tqdm(annotations.items(), desc=f"Preparing {dataset} sentences"):
        path_to_video = None
        for ext in constants.VIDEO_EXTENSIONS:
            path = os.path.join(videos_dir, video_id + ext)
            if os.path.exists(path):
                path_to_video = path
                break
        if path_to_video is None:
            print(f"[warn] Missing video file for {video_id}, skipping")
            continue

        segments = normalize_segments(
            transcribe_video(path_to_video, whisper_model, language),
            language
        )
        if not segments or all(s['text'].strip() == "" for s in segments):
            continue

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
    ap = argparse.ArgumentParser(description="Train supervised sentence ranker on TVSum/SumMe (Ridge only)")
    ap.add_argument("--dataset", required=True, choices=["tvsum", "summe"])
    ap.add_argument("--root", required=True, help="Dataset root directory")
    ap.add_argument("--whisper_model", default="small")
    ap.add_argument("--embedding_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--language", default="en")
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--top_k", type=int, default=5, help="Number of sentences in summary")
    args = ap.parse_args()

    x, y, sentences, sentence_lengths = collect_training_samples(args.dataset, args.root, args.whisper_model,
                                                                 args.language,
                                                                 args.embedding_model)
    x_train, x_val, x_test, y_train, y_val, y_test, s_train, s_val, s_test, groups_train, groups_val, groups_test = train_test_val_split(
        x, y, sentences, sentence_lengths)

    model = train_ranker(x_train, y_train, groups_train)
    val_bert = evaluate_model(model, x_val, y_val, s_val, groups_val, args.top_k, args.language)
    print("BERTScore on val: ", val_bert)

    x_train_val = np.vstack([x_train, x_val])
    y_train_val = np.concatenate([y_train, y_val])
    groups_train_val = np.concatenate([groups_train, groups_val])
    final_model = train_ranker(x_train_val, y_train_val, groups_train_val)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(final_model, args.model_out)
    print(f"Saved final model to {args.model_out}")

    bert = evaluate_model(final_model, x_test, y_test, s_test, groups_test, args.top_k, args.language)
    print("Final Test BERTScore:", bert)


if __name__ == '__main__':
    main()
