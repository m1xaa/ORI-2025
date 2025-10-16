import argparse
import os

from text_processing import normalize_segments
from theses_generation import generate_theses

from utils import save_markdown_table
from whisper_asr import transcribe_video


def run_pipeline(
        path_to_video: str,
        output_directory: str,
        whisper_model: str,
        language: str,
        embedding_model: str,
        top_k: int,
        path_to_model: str,
        generative_model: str
):
    segments = normalize_segments(transcribe_video(path_to_video, whisper_model, language), language)
    theses = generate_theses(segments, top_k, embedding_model, path_to_model, generative_model)
    os.makedirs(os.path.dirname(output_directory), exist_ok=True)
    save_markdown_table(theses, output_directory)


def main():
    ap = argparse.ArgumentParser(description="Video Thesis Generator")
    ap.add_argument("--video_path", required=True, help="Path to input video file")
    ap.add_argument("--output", required=True, help="Path to output MD")
    ap.add_argument("--whisper_model", default="small", help="Whisper model size (tiny|base|small|medium|large)")
    ap.add_argument("--language", default=None, help="Force language code (e.g., 'sr', 'en')")
    ap.add_argument("--embedding_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--top_k", type=int, default=8, help="How many theses to return")
    ap.add_argument("--ranker_model", default=None, help="Path to joblib model")
    ap.add_argument("--generative_model", default=None,
                    help="Optional summarization model (e.g. 'facebook/bart-large-cnn' or 'google/mt5-base')")
    args = ap.parse_args()

    run_pipeline(args.video_path, args.output, args.whisper_model, args.language, args.embedding_model, args.top_k,
                 args.ranker_model, args.generative_model)


if __name__ == '__main__':
    main()
