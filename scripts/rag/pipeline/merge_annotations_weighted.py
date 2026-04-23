import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

"""
python merge_annotations_weighted.py \
  --annotations ./data/temp_dataset/annotations_rows.csv \
  --raters ./data/temp_dataset/annotators_rows.csv \
  --comments ./data/temp_dataset/dataset_comments_rows.csv \
  --output ./data/temp_dataset/dataset_unificado.csv \
  --trans-weight 2.5 \
  --cis-weight 1.0 \
  --aspect-threshold 0.5
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge annotations into a single dataset using weighted voting."
    )
    parser.add_argument("--annotations", required=True, help="Path to annotations CSV")
    parser.add_argument("--raters", required=True, help="Path to raters CSV")
    parser.add_argument("--comments", required=True, help="Path to comments CSV")
    parser.add_argument("--output", required=True, help="Path to output merged CSV")
    parser.add_argument("--trans-weight", type=float, default=2.0, help="Weight for TRANS raters")
    parser.add_argument("--cis-weight", type=float, default=1.0, help="Weight for CIS raters")
    parser.add_argument(
        "--aspect-threshold",
        type=float,
        default=0.5,
        help="Minimum weighted support ratio for an aspect to be included",
    )
    return parser.parse_args()


def safe_parse_aspects(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    value = str(value).strip()
    if value == "" or value == "[]":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return []


def build_weight_map(raters_df, trans_weight, cis_weight):
    weight_map = {}
    group_map = {}

    for _, row in raters_df.iterrows():
        token = str(row["token"]).strip()
        group_type = str(row["group_type"]).strip().upper()
        group_map[token] = group_type
        if group_type == "TRANS":
            weight_map[token] = trans_weight
        else:
            weight_map[token] = cis_weight

    return weight_map, group_map


def weighted_winner(group_df, label_col, weight_map, group_map):
    weighted_scores = defaultdict(float)
    raw_counts = Counter()

    for _, row in group_df.iterrows():
        token = str(row["token"]).strip()
        label = str(row[label_col]).strip()
        weight = weight_map.get(token, 1.0)
        weighted_scores[label] += weight
        raw_counts[label] += 1

    if not weighted_scores:
        return None, 0.0, "no_annotations"

    max_weight = max(weighted_scores.values())
    tied = [label for label, score in weighted_scores.items() if score == max_weight]

    if len(tied) == 1:
        winner = tied[0]
        total = sum(weighted_scores.values())
        confidence = max_weight / total if total else 0.0
        return winner, confidence, "weighted_majority"

    max_count = max(raw_counts[label] for label in tied)
    tied2 = [label for label in tied if raw_counts[label] == max_count]

    if len(tied2) == 1:
        winner = tied2[0]
        total = sum(weighted_scores.values())
        confidence = weighted_scores[winner] / total if total else 0.0
        return winner, confidence, "tie_break_raw_count"

    trans_rows = group_df[group_df["token"].map(lambda t: group_map.get(str(t).strip(), "CIS")) == "TRANS"].copy()
    if not trans_rows.empty:
        trans_rows["created_at"] = pd.to_datetime(trans_rows["created_at"], utc=True, errors="coerce")
        trans_rows = trans_rows.sort_values("created_at", ascending=False)
        for _, row in trans_rows.iterrows():
            label = str(row[label_col]).strip()
            if label in tied2:
                total = sum(weighted_scores.values())
                confidence = weighted_scores[label] / total if total else 0.0
                return label, confidence, "tie_break_latest_trans"

    rows = group_df.copy()
    rows["created_at"] = pd.to_datetime(rows["created_at"], utc=True, errors="coerce")
    rows = rows.sort_values("created_at", ascending=False)
    for _, row in rows.iterrows():
        label = str(row[label_col]).strip()
        if label in tied2:
            total = sum(weighted_scores.values())
            confidence = weighted_scores[label] / total if total else 0.0
            return label, confidence, "tie_break_latest_any"

    winner = tied2[0]
    total = sum(weighted_scores.values())
    confidence = weighted_scores[winner] / total if total else 0.0
    return winner, confidence, "tie_break_fallback"


def aggregate_aspects(group_df, weight_map, threshold):
    aspect_scores = defaultdict(float)
    total_weight = 0.0

    for _, row in group_df.iterrows():
        token = str(row["token"]).strip()
        weight = weight_map.get(token, 1.0)
        total_weight += weight

        for aspect in safe_parse_aspects(row["aspects"]):
            aspect_scores[aspect] += weight

    if total_weight == 0:
        return [], {}

    chosen = []
    support = {}
    for aspect, score in sorted(aspect_scores.items(), key=lambda x: (-x[1], x[0])):
        ratio = score / total_weight
        support[aspect] = ratio
        if ratio >= threshold:
            chosen.append(aspect)

    return chosen, support


def nominal_distance(a, b):
    return 0 if a == b else 1


def krippendorff_alpha_by_item(value_lists):
    units = [vals for vals in value_lists if len(vals) >= 2]
    if not units:
        return None

    observed_disagreement = 0.0
    observed_pairs = 0

    all_values = []
    for vals in units:
        all_values.extend(vals)
        n = len(vals)
        for i in range(n):
            for j in range(i + 1, n):
                observed_disagreement += nominal_distance(vals[i], vals[j])
                observed_pairs += 1

    if observed_pairs == 0:
        return None

    do = observed_disagreement / observed_pairs

    value_counts = Counter(all_values)
    total = sum(value_counts.values())
    if total <= 1:
        return None

    expected_disagreement = 0.0
    expected_pairs = 0
    distinct_values = list(value_counts.keys())
    for i, v1 in enumerate(distinct_values):
        for j, v2 in enumerate(distinct_values):
            if j <= i:
                continue
            expected_disagreement += value_counts[v1] * value_counts[v2] * nominal_distance(v1, v2)
            expected_pairs += value_counts[v1] * value_counts[v2]

    if expected_pairs == 0:
        return 1.0

    de = expected_disagreement / expected_pairs
    if de == 0:
        return 1.0

    return 1 - (do / de)


def build_diagnostics(annotations_df):
    speech_lists = []
    offensive_lists = []

    for _, group in annotations_df.groupby("dataset_comment_id"):
        speech_vals = [str(x).strip() for x in group["speech_class"].dropna().tolist()]
        offensive_vals = [str(x).strip() for x in group["offensive_class"].dropna().tolist()]

        if len(speech_vals) >= 2:
            speech_lists.append(speech_vals)
        if len(offensive_vals) >= 2:
            offensive_lists.append(offensive_vals)

    diagnostics = {
        "num_annotations": int(len(annotations_df)),
        "num_comments_with_annotations": int(annotations_df["dataset_comment_id"].nunique()),
        "krippendorff_alpha_speech_class": krippendorff_alpha_by_item(speech_lists),
        "krippendorff_alpha_offensive_class": krippendorff_alpha_by_item(offensive_lists),
    }
    return diagnostics


def main():
    args = parse_args()

    annotations_df = pd.read_csv(args.annotations)
    raters_df = pd.read_csv(args.raters)
    comments_df = pd.read_csv(args.comments)

    annotations_df["dataset_comment_id"] = pd.to_numeric(annotations_df["dataset_comment_id"], errors="coerce")
    comments_df["id"] = pd.to_numeric(comments_df["id"], errors="coerce")

    weight_map, group_map = build_weight_map(
        raters_df,
        trans_weight=args.trans_weight,
        cis_weight=args.cis_weight,
    )

    merged_rows = []

    for comment_id, group in annotations_df.groupby("dataset_comment_id"):
        comment_row = comments_df[comments_df["id"] == comment_id]
        if comment_row.empty:
            continue

        comment_row = comment_row.iloc[0]

        speech_class, speech_conf, speech_rule = weighted_winner(group, "speech_class", weight_map, group_map)
        offensive_class, offensive_conf, offensive_rule = weighted_winner(
            group, "offensive_class", weight_map, group_map
        )
        aspects, aspect_support = aggregate_aspects(group, weight_map, args.aspect_threshold)

        trans_count = sum(1 for t in group["token"] if group_map.get(str(t).strip(), "CIS") == "TRANS")
        cis_count = sum(1 for t in group["token"] if group_map.get(str(t).strip(), "CIS") == "CIS")

        merged_rows.append(
            {
                "dataset_comment_id": int(comment_id),
                "video_title": comment_row["video_title"],
                "comment": comment_row["comment"],
                "speech_class": speech_class,
                "offensive_class": offensive_class,
                "aspects": json.dumps(aspects, ensure_ascii=False),
                "num_annotations": int(len(group)),
                "num_trans_annotations": int(trans_count),
                "num_cis_annotations": int(cis_count),
                "speech_confidence": round(float(speech_conf), 4),
                "offensive_confidence": round(float(offensive_conf), 4),
                "speech_rule": speech_rule,
                "offensive_rule": offensive_rule,
                "aspect_support": json.dumps(
                    {k: round(float(v), 4) for k, v in aspect_support.items()},
                    ensure_ascii=False,
                ),
            }
        )

    merged_df = pd.DataFrame(merged_rows).sort_values("dataset_comment_id")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False, encoding="utf-8")

    diagnostics = build_diagnostics(annotations_df)
    diagnostics.update(
        {
            "trans_weight": args.trans_weight,
            "cis_weight": args.cis_weight,
            "aspect_threshold": args.aspect_threshold,
            "num_output_rows": int(len(merged_df)),
        }
    )

    diagnostics_path = output_path.with_suffix(".diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Merged dataset saved to: {output_path}")
    print(f"Diagnostics saved to: {diagnostics_path}")


if __name__ == "__main__":
    main()
