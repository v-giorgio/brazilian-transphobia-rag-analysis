import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_METRICS_DIR = Path(__file__).resolve().parent / "data" / "final" / "metrics"
DEFAULT_OUTPUT_DIR = DEFAULT_METRICS_DIR / "ragas_summary"

RAGAS_METRICS = [
    "context_recall",
    "context_precision",
    "response_relevancy",
    "faithfulness",
]

RAGAS_ALIASES = {
    "response_relevancy": ["response_relevancy", "answer_relevancy"],
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_metadata_from_path(path: Path, metrics_dir: Path) -> Dict[str, str]:
    rel_parts = path.relative_to(metrics_dir).parts
    model = ""
    strategy = ""

    if "by_model" in rel_parts:
        idx = rel_parts.index("by_model")
        if len(rel_parts) > idx + 1:
            model = rel_parts[idx + 1]
        if len(rel_parts) > idx + 2:
            strategy = rel_parts[idx + 2]

    filename = path.name
    suffix = "_ragas_results.csv"
    task = filename[:-len(suffix)] if filename.endswith(suffix) else path.stem

    prefix = f"{model}_{strategy}_"
    if model and strategy and task.startswith(prefix):
        task = task[len(prefix):]

    return {
        "model": model or "unknown_model",
        "strategy": strategy or "unknown_strategy",
        "task": task or "unknown_task",
        "source_file": str(path),
    }


def numeric_series(df: pd.DataFrame, metric: str) -> pd.Series:
    candidate_columns = RAGAS_ALIASES.get(metric, [metric])
    for column in candidate_columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")


def load_ragas_results(metrics_dir: Path) -> pd.DataFrame:
    files = sorted(metrics_dir.rglob("*_ragas_results.csv"))
    frames = []

    for path in files:
        metadata = parse_metadata_from_path(path, metrics_dir)
        df = pd.read_csv(path)
        if df.empty:
            continue

        normalized = pd.DataFrame()
        normalized["_ragas_row_id"] = df.get("_ragas_row_id", pd.Series(range(len(df)))).astype(str)
        normalized["model"] = metadata["model"]
        normalized["strategy"] = metadata["strategy"]
        normalized["task"] = metadata["task"]
        normalized["source_file"] = metadata["source_file"]

        for metric in RAGAS_METRICS:
            normalized[metric] = numeric_series(df, metric)

        frames.append(normalized)

    if not frames:
        return pd.DataFrame(columns=["_ragas_row_id", "model", "strategy", "task", "source_file", *RAGAS_METRICS])

    all_rows = pd.concat(frames, ignore_index=True, sort=False)
    all_rows = all_rows.drop_duplicates(
        subset=["source_file", "_ragas_row_id"],
        keep="last",
    )
    return all_rows


def summarize_group(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(group_cols, dropna=False)
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {column: value for column, value in zip(group_cols, group_key)}
        row["num_rows"] = int(len(group_df))

        metric_means = []
        metric_coverages = []
        for metric in RAGAS_METRICS:
            values = pd.to_numeric(group_df[metric], errors="coerce")
            valid_count = int(values.notna().sum())
            row[f"{metric}_mean"] = float(values.mean()) if valid_count else np.nan
            row[f"{metric}_count"] = valid_count
            row[f"{metric}_coverage"] = valid_count / len(group_df) if len(group_df) else 0.0
            if valid_count:
                metric_means.append(row[f"{metric}_mean"])
            metric_coverages.append(row[f"{metric}_coverage"])

        complete_mask = group_df[RAGAS_METRICS].notna().all(axis=1)
        row["complete_rows"] = int(complete_mask.sum())
        row["complete_row_rate"] = float(complete_mask.mean()) if len(group_df) else 0.0
        row["mean_metric_coverage"] = float(np.mean(metric_coverages)) if metric_coverages else 0.0
        row["ragas_mean_available"] = float(np.mean(metric_means)) if metric_means else np.nan
        row["ragas_mean_strict"] = (
            float(np.mean([row[f"{metric}_mean"] for metric in RAGAS_METRICS]))
            if all(pd.notna(row[f"{metric}_mean"]) for metric in RAGAS_METRICS)
            else np.nan
        )

        rows.append(row)

    summary = pd.DataFrame(rows)
    sort_cols = [col for col in ["ragas_mean_strict", "ragas_mean_available", "mean_metric_coverage"] if col in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    return summary


def best_row(df: pd.DataFrame, label_cols: List[str]) -> Optional[Dict[str, object]]:
    if df.empty:
        return None

    ranked = df.copy()
    ranked["_rank_score"] = ranked["ragas_mean_strict"].fillna(ranked["ragas_mean_available"])
    ranked = ranked.sort_values(
        ["_rank_score", "mean_metric_coverage", "complete_row_rate", "num_rows"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    row = ranked.iloc[0].drop(labels=["_rank_score"]).to_dict()
    return {key: row.get(key) for key in [*label_cols, "ragas_mean_strict", "ragas_mean_available", "mean_metric_coverage", "complete_row_rate", "num_rows"]}


def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    clean = df.replace({np.nan: None})
    return clean.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume métricas RAGAS por estratégia, modelo e combinação.")
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR, help="Diretório base de métricas.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Diretório de saída dos resumos RAGAS.")
    args = parser.parse_args()

    metrics_dir = args.metrics_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    all_rows = load_ragas_results(metrics_dir)
    all_rows.to_csv(output_dir / "ragas_all_rows.csv", index=False, encoding="utf-8")

    by_strategy = summarize_group(all_rows, ["strategy"])
    by_model = summarize_group(all_rows, ["model"])
    by_model_strategy = summarize_group(all_rows, ["model", "strategy"])
    by_task = summarize_group(all_rows, ["task"])
    by_model_strategy_task = summarize_group(all_rows, ["model", "strategy", "task"])

    by_strategy.to_csv(output_dir / "ragas_summary_by_strategy.csv", index=False, encoding="utf-8")
    by_model.to_csv(output_dir / "ragas_summary_by_model.csv", index=False, encoding="utf-8")
    by_model_strategy.to_csv(output_dir / "ragas_summary_by_model_strategy.csv", index=False, encoding="utf-8")
    by_task.to_csv(output_dir / "ragas_summary_by_task.csv", index=False, encoding="utf-8")
    by_model_strategy_task.to_csv(output_dir / "ragas_summary_by_model_strategy_task.csv", index=False, encoding="utf-8")

    best = {
        "best_strategy": best_row(by_strategy, ["strategy"]),
        "best_model": best_row(by_model, ["model"]),
        "best_model_strategy": best_row(by_model_strategy, ["model", "strategy"]),
        "metric_names": RAGAS_METRICS,
        "ranking_note": (
            "O ranking usa ragas_mean_strict quando as quatro métricas existem; "
            "caso contrário, usa ragas_mean_available e reporta cobertura para evitar interpretação indevida."
        ),
    }
    with open(output_dir / "ragas_best.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    with open(output_dir / "ragas_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "num_rows": int(len(all_rows)),
            "num_files": int(all_rows["source_file"].nunique()) if not all_rows.empty else 0,
            "best": best,
            "by_strategy": dataframe_to_records(by_strategy),
            "by_model": dataframe_to_records(by_model),
            "by_model_strategy": dataframe_to_records(by_model_strategy),
            "by_task": dataframe_to_records(by_task),
        }, f, ensure_ascii=False, indent=2)

    print(f"Resumo RAGAS salvo em: {output_dir}")
    print(f"Arquivos RAGAS agregados: {all_rows['source_file'].nunique() if not all_rows.empty else 0}")
    print(f"Linhas agregadas: {len(all_rows)}")
    if best["best_strategy"]:
        print(f"Melhor estratégia: {best['best_strategy']}")
    if best["best_model"]:
        print(f"Melhor modelo: {best['best_model']}")
    if best["best_model_strategy"]:
        print(f"Melhor combinação modelo+estratégia: {best['best_model_strategy']}")


if __name__ == "__main__":
    main()
