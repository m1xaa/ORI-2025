import argparse
import os
import numpy as np
import constants
from data.TVSum.load_annotations import load_annotations
from featurization import align_segments_with_user_scores, make_feature_matrix
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

        labels = align_segments_with_user_scores(segments, annotations['user_scores'], annotations['fps'])
        features = make_feature_matrix(segments, embedding_model_name)

        return np.vstack(features), np.concatenate(labels), [s['text'] for s in segments]







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

    x, y, sentences = collect_training_samples(args.dataset, args.root, args.whisper_model, args.language, args.embedding_model)
    x_train, x_val, x_test, y_train, y_val, y_test, s_train, s_val, s_test = train_test_val_split(x, y, sentences)






if __name__ == '__main__':
    main()