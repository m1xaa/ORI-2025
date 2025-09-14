import argparse
import os

import constants
from data.TVSum.load_annotations import load_annotations
from whisper_asr import transcribe_video


def collect_training_samples(dataset: str, root: str, whisper_model: str, language: str):
    if dataset == "tvsum":
        annotations = load_annotations(root)
    else:
        raise NotImplementedError("SumME not implemented yet")
    videos_dir = os.path.join(root, constants.VIDEOS_DIR)

    x, y, sentences = [], [], []
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

        segments = transcribe_video(path_to_video, model_size=whisper_model, language=language)

def main():
    ap = argparse.ArgumentParser(description="Train supervised sentence ranker on TVSum/SumMe (Ridge only)")
    ap.add_argument("--dataset", required=True, choices=["tvsum", "summe"])
    ap.add_argument("--root", required=True, help="Dataset root directory")
    ap.add_argument("--whisper_model", default="small")
    ap.add_argument("--language", default="en")
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--top_k", type=int, default=5, help="Number of sentences in summary")
    args = ap.parse_args()

    collect_training_samples(args.dataset, args.root, args.whisper_model, args.language)




if __name__ == '__main__':
    main()