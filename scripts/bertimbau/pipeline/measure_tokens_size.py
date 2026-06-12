from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


DEFAULT_DATASET = Path(__file__).parent / "data" / "master" / "clean_dataset.csv"
DEFAULT_MODEL = "neuralmind/bert-base-portuguese-cased"


def build_title_comment(row: pd.Series) -> str:
    title = "" if pd.isna(row["video_title"]) else str(row["video_title"]).strip()
    comment = "" if pd.isna(row["comment"]) else str(row["comment"]).strip()
    return f"{title} {comment}".strip()


def percentile(series: pd.Series, value: float) -> float:
    return float(series.quantile(value / 100.0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure BERTimbau token lengths for video_title + comment."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    required_cols = {"video_title", "comment"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    texts = df.apply(build_title_comment, axis=1).tolist()

    token_lengths = [
        len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in texts
    ]

    lengths = pd.Series(token_lengths, name="token_length")
    total = int(len(lengths))
    within_limit = int((lengths <= args.max_length).sum())
    over_limit = total - within_limit
    within_pct = (within_limit / total * 100.0) if total else 0.0
    over_pct = (over_limit / total * 100.0) if total else 0.0

    print(f"Dataset: {args.dataset}")
    print(f"Tokenizer: {args.model_name}")
    print(f"Input format: video_title + comment")
    print(f"Max length: {args.max_length}")
    print(f"Total records: {total}")
    print(f"<= {args.max_length} tokens: {within_limit} ({within_pct:.2f}%)")
    print(f">  {args.max_length} tokens: {over_limit} ({over_pct:.2f}%)")
    print()
    print("Token length summary:")
    print(f"min: {int(lengths.min())}")
    print(f"mean: {lengths.mean():.2f}")
    print(f"median: {percentile(lengths, 50):.2f}")
    print(f"p75: {percentile(lengths, 75):.2f}")
    print(f"p90: {percentile(lengths, 90):.2f}")
    print(f"p95: {percentile(lengths, 95):.2f}")
    print(f"p99: {percentile(lengths, 99):.2f}")
    print(f"max: {int(lengths.max())}")

    if args.output_csv:
        out_df = df[["dataset_comment_id", "video_title", "comment"]].copy()
        out_df["title_comment_text"] = texts
        out_df["token_length"] = token_lengths
        out_df["within_max_length"] = out_df["token_length"] <= args.max_length
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output_csv, index=False, encoding="utf-8")
        print()
        print(f"Per-record token lengths saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
