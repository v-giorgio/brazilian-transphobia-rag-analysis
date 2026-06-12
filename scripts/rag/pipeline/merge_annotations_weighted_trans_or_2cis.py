import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

"""
python merge_annotations_weighted_trans_or_2cis.py \
  --annotations ./data/annotations/annotations_rows.csv \
  --raters ./data/annotations/annotators_rows.csv \
  --comments ./data/annotations/dataset_comments_rows.csv \
  --output ./data/annotations/dataset_final.csv \
  --output-full ./data/annotations/dataset_final_full.csv \
  --trans-weight 1.5 \
  --cis-weight 1.0 \
  --aspect-threshold 0.51
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge annotations into a single dataset using weighted voting."
    )
    parser.add_argument("--annotations", required=True, help="Path to annotations CSV")
    parser.add_argument("--raters", required=True, help="Path to raters CSV")
    parser.add_argument("--comments", required=True, help="Path to comments CSV")
    parser.add_argument("--output", required=True, help="Path to experimental output CSV")
    parser.add_argument(
        "--output-full",
        default=None,
        help="Path to extra artifact with all annotated comments. Defaults to <output>_full.csv",
    )
    parser.add_argument("--trans-weight", type=float, default=1.5, help="Weight for TRANS raters")
    parser.add_argument("--cis-weight", type=float, default=1.0, help="Weight for CIS raters")
    parser.add_argument(
        "--aspect-threshold",
        type=float,
        default=0.51,
        help="Minimum weighted support ratio for an aspect to be included",
    )
    return parser.parse_args()


def safe_parse_aspects(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
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


def count_annotation_groups(group_df, group_map):
    trans_count = sum(
        1 for t in group_df["token"]
        if group_map.get(str(t).strip(), "CIS") == "TRANS"
    )
    cis_count = sum(
        1 for t in group_df["token"]
        if group_map.get(str(t).strip(), "CIS") == "CIS"
    )
    return trans_count, cis_count


def is_experimental_comment(trans_count, cis_count):
    return (trans_count + cis_count) >= 2


def resolve_full_output_path(output_path, output_full):
    if output_full:
        return Path(output_full)
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_full{output_path.suffix}")


def _label_confidence(labels, winner):
    total = len(labels)
    if total == 0:
        return 0.0
    return sum(1 for label in labels if label == winner) / total


def _latest_trans_label(group_df, label_col, group_map):
    trans_rows = group_df[
        group_df["token"].map(lambda t: group_map.get(str(t).strip(), "CIS")) == "TRANS"
    ].copy()
    if trans_rows.empty:
        return None
    trans_rows["created_at"] = pd.to_datetime(trans_rows["created_at"], utc=True, errors="coerce")
    trans_rows = trans_rows.sort_values("created_at", ascending=False)
    return str(trans_rows.iloc[0][label_col]).strip()


def resolve_two_cis_speech(labels):
    labels = [str(label).strip() for label in labels]
    unique = set(labels)

    if len(unique) == 1:
        winner = labels[0]
        return winner, 1.0, "two_cis_agreement"

    if unique == {"HATEFUL", "SUPPORTIVE"}:
        return "NEUTRAL", 0.5, "two_cis_semantic_tie_hateful_supportive"
    if unique == {"HATEFUL", "NEUTRAL"}:
        return "HATEFUL", 0.5, "two_cis_semantic_tie_hateful_neutral"
    if unique == {"SUPPORTIVE", "NEUTRAL"}:
        return "SUPPORTIVE", 0.5, "two_cis_semantic_tie_supportive_neutral"

    return labels[0], _label_confidence(labels, labels[0]), "two_cis_fallback"


def resolve_two_cis_offensive(labels):
    labels = [str(label).strip() for label in labels]
    unique = set(labels)

    if len(unique) == 1:
        winner = labels[0]
        return winner, 1.0, "two_cis_agreement"

    if unique == {"OFFENSIVE", "NON_OFFENSIVE"}:
        return "OFFENSIVE", 0.5, "two_cis_semantic_tie_offensive_non_offensive"

    return labels[0], _label_confidence(labels, labels[0]), "two_cis_fallback"


def special_case_winner(group_df, label_col, group_map):
    trans_count, cis_count = count_annotation_groups(group_df, group_map)

    if trans_count == 0 and cis_count == 2:
        labels = [str(label).strip() for label in group_df[label_col].dropna().tolist()]
        if len(labels) == 2 and label_col == "speech_class":
            return resolve_two_cis_speech(labels)
        if len(labels) == 2 and label_col == "offensive_class":
            return resolve_two_cis_offensive(labels)

    if trans_count == 1 and cis_count == 1:
        trans_label = _latest_trans_label(group_df, label_col, group_map)
        if trans_label is not None:
            labels = [str(label).strip() for label in group_df[label_col].dropna().tolist()]
            return trans_label, _label_confidence(labels, trans_label), "one_trans_one_cis_trans_priority"

    return None


def weighted_winner(group_df, label_col, weight_map, group_map):
    special = special_case_winner(group_df, label_col, group_map)
    if special is not None:
        return special

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

    # Observed coincidence matrix o_ck.
    # For each item with n annotations, every category pair contributes
    # m_c * m_k / (n - 1), with m_c * (m_c - 1) for the diagonal.
    observed_coincidences = Counter()
    category_counts = Counter()
    total_observed_coincidences = 0.0

    for vals in units:
        n = len(vals)
        counts = Counter(vals)
        category_counts.update(counts)
        total_observed_coincidences += n

        for c, count_c in counts.items():
            for k, count_k in counts.items():
                if c == k:
                    contribution = count_c * (count_c - 1) / (n - 1)
                else:
                    contribution = count_c * count_k / (n - 1)
                observed_coincidences[(c, k)] += contribution

    if total_observed_coincidences <= 0:
        return None

    # D_o = sum_ck o_ck * delta(c,k) / sum_ck o_ck
    observed_disagreement = sum(
        count * nominal_distance(c, k)
        for (c, k), count in observed_coincidences.items()
    )
    do = observed_disagreement / total_observed_coincidences

    total = sum(category_counts.values())
    if total <= 1:
        return None

    # Expected coincidence matrix e_ck, derived from global category marginals.
    # D_e = sum_ck e_ck * delta(c,k) / sum_ck e_ck
    expected_disagreement = 0.0
    labels = list(category_counts.keys())
    for c in labels:
        for k in labels:
            if c == k:
                expected = category_counts[c] * (category_counts[c] - 1) / (total - 1)
            else:
                expected = category_counts[c] * category_counts[k] / (total - 1)
            expected_disagreement += expected * nominal_distance(c, k)

    de = expected_disagreement / total
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

    all_annotated_rows = []

    removed_single_annotation_experimental = 0
    composition_counts = Counter()

    for comment_id, group in annotations_df.groupby("dataset_comment_id"):
        trans_count, cis_count = count_annotation_groups(group, group_map)
        total_group_annotations = int(len(group))
        experimental_eligible = is_experimental_comment(trans_count, cis_count)
        if experimental_eligible:
            eligibility_rule = "experimental_min_2_annotations"
        else:
            eligibility_rule = "full_artifact_only_single_annotation"
            removed_single_annotation_experimental += 1

        composition_counts[f"{cis_count}_cis_{trans_count}_trans"] += 1

        comment_row = comments_df[comments_df["id"] == comment_id]
        if comment_row.empty:
            continue

        comment_row = comment_row.iloc[0]

        speech_class, speech_conf, speech_rule = weighted_winner(group, "speech_class", weight_map, group_map)
        offensive_class, offensive_conf, offensive_rule = weighted_winner(
            group, "offensive_class", weight_map, group_map
        )
        aspects, aspect_support = aggregate_aspects(group, weight_map, args.aspect_threshold)

        all_annotated_rows.append(
            {
                "dataset_comment_id": int(comment_id),
                "video_title": comment_row["video_title"],
                "comment": comment_row["comment"],
                "speech_class": speech_class,
                "offensive_class": offensive_class,
                "aspects": json.dumps(aspects, ensure_ascii=False),
                "num_annotations": total_group_annotations,
                "num_trans_annotations": int(trans_count),
                "num_cis_annotations": int(cis_count),
                "eligibility_rule": eligibility_rule,
                "experimental_eligible": bool(experimental_eligible),
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_output_path = resolve_full_output_path(output_path, args.output_full)
    full_output_path.parent.mkdir(parents=True, exist_ok=True)

    full_df = pd.DataFrame(all_annotated_rows).sort_values("dataset_comment_id")
    merged_df = full_df[full_df["experimental_eligible"]].copy()

    merged_df.to_csv(output_path, index=False, encoding="utf-8")
    full_df.to_csv(full_output_path, index=False, encoding="utf-8")

    diagnostics = build_diagnostics(annotations_df)
    total_comments = int(comments_df["id"].nunique())
    comments_with_annotations = int(annotations_df["dataset_comment_id"].nunique())
    comments_without_annotations = max(total_comments - comments_with_annotations, 0)
    diagnostics.update(
        {
            "num_total_comments": total_comments,
            "num_comments_without_annotations": comments_without_annotations,
            "trans_weight": args.trans_weight,
            "cis_weight": args.cis_weight,
            "aspect_threshold": args.aspect_threshold,
            "experimental_selection_rule": "include if num_trans_annotations + num_cis_annotations >= 2",
            "full_artifact_selection_rule": "include all comments with at least one annotation",
            "num_experimental_output_rows": int(len(merged_df)),
            "num_full_artifact_rows": int(len(full_df)),
            "num_annotated_comments_not_in_experimental_output": int(comments_with_annotations - len(merged_df)),
            "num_comments_removed_single_annotation_from_experiments": int(removed_single_annotation_experimental),
            "annotation_composition_counts": {str(k): int(v) for k, v in composition_counts.items()},
        }
    )

    diagnostics_path = output_path.with_suffix(".diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Experimental dataset saved to: {output_path}")
    print(f"Full annotated artifact saved to: {full_output_path}")
    print(f"Diagnostics saved to: {diagnostics_path}")


if __name__ == "__main__":
    main()
