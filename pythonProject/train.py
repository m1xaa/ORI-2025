import argparse

from data.TVSum.load_annotations import load_annotations


def main():
    ap = argparse.ArgumentParser(description="Train supervised sentence ranker on TVSum/SumMe (Ridge only)")
    ap.add_argument("--dataset", required=True, choices=["tvsum", "summe"])
    ap.add_argument("--root", required=True, help="Dataset root directory")
    ap.add_argument("--whisper_model", default="small")
    ap.add_argument("--language", default="en")
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--top_k", type=int, default=5, help="Number of sentences in summary")
    args = ap.parse_args()

    load_annotations(args.root)




if __name__ == '__main__':
    main()