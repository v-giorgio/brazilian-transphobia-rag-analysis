import os
import ast
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
    multilabel_confusion_matrix,
    hamming_loss,
    jaccard_score,
    zero_one_loss,
)
from sklearn.preprocessing import label_binarize

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
import ragas.metrics as ragas_metrics

DEFAULT_BASE_OUTPUT_DIR = "./data/metrics"
DEFAULT_PREDICTIONS_DIR = "./data"
DEFAULT_MODEL = "qwen3:14b"

DEFAULT_STRATEGIES = ["zero_shot", "few_shot", "rag_transcripts", "rag_wiki", "rag_both"]

TASK_ALIASES = {
    "speech": "speech_class",
    "speech_class": "speech_class",
    "offensive": "offensive_class",
    "offensive_class": "offensive_class",
    "aspects": "aspects",
}


SPEECH_LABELS = ["HATEFUL", "SUPPORTIVE", "NEUTRAL"]
OFFENSIVE_LABELS = ["OFFENSIVE", "NON_OFFENSIVE"]
ASPECT_LABELS = [
    "RELIGION",
    "POLITICS",
    "GENDER_DENIAL",
    "HARRASSMENT",
    "THREATENING",
    "PSEUDOSCIENCE",
]


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def safe_parse_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip().upper() for x in value if str(x).strip()]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(x).strip().upper() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return []


def safe_json_loads(value: Any, default: Any = None) -> Any:
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, float) and np.isnan(value):
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            try:
                return ast.literal_eval(value)
            except Exception:
                return default
    return default


def normalize_requested_tasks(tasks: List[str]) -> List[str]:
    normalized = []
    for task in tasks:
        task = str(task).strip().lower()
        if task == "all":
            return ["speech_class", "offensive_class", "aspects"]
        if task not in TASK_ALIASES:
            raise ValueError(f"Tarefa inválida: {task}. Use: speech, offensive, aspects ou all.")
        normalized.append(TASK_ALIASES[task])

    seen = set()
    unique = []
    for task in normalized:
        if task not in seen:
            unique.append(task)
            seen.add(task)
    return unique


def required_columns_for_tasks(tasks: List[str]) -> List[str]:
    required = []
    if "speech_class" in tasks:
        required.extend(["speech_class", "pred_speech_class"])
    if "offensive_class" in tasks:
        required.extend(["offensive_class", "pred_offensive_class"])
    if "aspects" in tasks:
        required.extend(["aspects", "pred_aspects"])
    return required


def aspect_list_from_indicator_row(row: pd.Series) -> List[str]:
    labels = []
    for label in ASPECT_LABELS:
        value = row.get(label, 0)
        if pd.notna(value) and int(value) == 1:
            labels.append(label)
    return labels


def filter_valid_rows(df: pd.DataFrame, tasks: List[str]) -> pd.DataFrame:
    if "aspects" in tasks and "aspects" not in df.columns and all(label in df.columns for label in ASPECT_LABELS):
        df = df.copy()
        df["aspects"] = df.apply(aspect_list_from_indicator_row, axis=1)

    required_cols = required_columns_for_tasks(tasks)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset para as tarefas escolhidas: {missing}")

    out = df.copy()

    if "error" in out.columns:
        out = out[out["error"].isna()].copy()

    if "speech_class" in tasks:
        out = out[out["speech_class"].notna() & out["pred_speech_class"].notna()].copy()
        out["speech_class"] = out["speech_class"].astype(str).str.strip().str.upper()
        out["pred_speech_class"] = out["pred_speech_class"].astype(str).str.strip().str.upper()
        out = out[
            out["speech_class"].isin(SPEECH_LABELS)
            & out["pred_speech_class"].isin(SPEECH_LABELS)
        ].copy()

    if "offensive_class" in tasks:
        out = out[out["offensive_class"].notna() & out["pred_offensive_class"].notna()].copy()
        out["offensive_class"] = out["offensive_class"].astype(str).str.strip().str.upper()
        out["pred_offensive_class"] = out["pred_offensive_class"].astype(str).str.strip().str.upper()
        out = out[
            out["offensive_class"].isin(OFFENSIVE_LABELS)
            & out["pred_offensive_class"].isin(OFFENSIVE_LABELS)
        ].copy()

    if "aspects" in tasks:
        out["aspects_true_list"] = out["aspects"].apply(safe_parse_list)
        out["aspects_pred_list"] = out["pred_aspects"].apply(safe_parse_list)

    if "model_name" not in out.columns:
        out["model_name"] = "unknown_model"
    if "prompt_strategy" not in out.columns:
        out["prompt_strategy"] = "unknown_prompt"

    out["model_name"] = out["model_name"].fillna("unknown_model").astype(str)
    out["prompt_strategy"] = out["prompt_strategy"].fillna("unknown_prompt").astype(str)

    return out


def multilabel_binarize(series: pd.Series, labels: List[str]) -> np.ndarray:
    label_to_idx = {label: i for i, label in enumerate(labels)}
    mat = np.zeros((len(series), len(labels)), dtype=int)

    for row_idx, items in enumerate(series.tolist()):
        for item in items:
            if item in label_to_idx:
                mat[row_idx, label_to_idx[item]] = 1
    return mat


def sanitize_name(text: str) -> str:
    text = str(text)
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def task_folder(task: str) -> str:
    if task == "speech_class":
        return "speech"
    if task == "offensive_class":
        return "offensive"
    if task == "aspects":
        return "aspects"
    raise ValueError(f"Tarefa invalida para caminho: {task}")


def strategy_prediction_path(predictions_dir: str, strategy: str, task: str, model: str) -> str:
    model_slug = sanitize_name(model)
    folder = task_folder(task)
    if strategy == "zero_shot":
        return str(Path(predictions_dir) / "zero_shot" / folder / f"predictions_{model_slug}.csv")
    if strategy == "few_shot":
        return str(Path(predictions_dir) / "few_shot" / folder / f"predictions_{model_slug}.csv")
    if strategy.startswith("rag_"):
        rag_mode = strategy.replace("rag_", "", 1)
        return str(Path(predictions_dir) / "rag" / folder / f"predictions_{model_slug}_{rag_mode}.csv")
    raise ValueError(f"Estrategia invalida: {strategy}")


def tasks_for_resolved_input(strategy_name: str, default_tasks: List[str]) -> List[str]:
    if strategy_name.endswith("_speech"):
        return ["speech_class"]
    if strategy_name.endswith("_offensive"):
        return ["offensive_class"]
    if strategy_name.endswith("_aspects"):
        return ["aspects"]
    return default_tasks


def metric_block_classification(y_true, y_pred, labels: List[str]) -> Dict[str, Any]:
    metrics = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics["precision_micro"] = float(p_micro)
    metrics["recall_micro"] = float(r_micro)
    metrics["f1_micro"] = float(f_micro)

    metrics["precision_macro"] = float(p_macro)
    metrics["recall_macro"] = float(r_macro)
    metrics["f1_macro"] = float(f_macro)

    metrics["precision_weighted"] = float(p_weighted)
    metrics["recall_weighted"] = float(r_weighted)
    metrics["f1_weighted"] = float(f_weighted)

    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    metrics["per_class"] = {
        label: {
            "precision": float(per_class[0][i]),
            "recall": float(per_class[1][i]),
            "f1": float(per_class[2][i]),
            "support": int(per_class[3][i]),
        }
        for i, label in enumerate(labels)
    }

    return metrics


def metric_block_multilabel(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    metrics = {}

    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="weighted", zero_division=0
    )
    p_samples, r_samples, f_samples, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="samples", zero_division=0
    )

    metrics["precision_micro"] = float(p_micro)
    metrics["recall_micro"] = float(r_micro)
    metrics["f1_micro"] = float(f_micro)

    metrics["precision_macro"] = float(p_macro)
    metrics["recall_macro"] = float(r_macro)
    metrics["f1_macro"] = float(f_macro)

    metrics["precision_weighted"] = float(p_weighted)
    metrics["recall_weighted"] = float(r_weighted)
    metrics["f1_weighted"] = float(f_weighted)

    metrics["precision_samples"] = float(p_samples)
    metrics["recall_samples"] = float(r_samples)
    metrics["f1_samples"] = float(f_samples)

    metrics["hamming_loss"] = float(hamming_loss(y_true_bin, y_pred_bin))
    metrics["subset_zero_one_loss"] = float(zero_one_loss(y_true_bin, y_pred_bin))
    metrics["subset_accuracy"] = float(1.0 - zero_one_loss(y_true_bin, y_pred_bin))
    metrics["jaccard_micro"] = float(jaccard_score(y_true_bin, y_pred_bin, average="micro", zero_division=0))
    metrics["jaccard_macro"] = float(jaccard_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))
    metrics["jaccard_samples"] = float(jaccard_score(y_true_bin, y_pred_bin, average="samples", zero_division=0))

    per_class = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average=None, zero_division=0
    )
    supports = y_true_bin.sum(axis=0)

    metrics["per_class"] = {
        label: {
            "precision": float(per_class[0][i]),
            "recall": float(per_class[1][i]),
            "f1": float(per_class[2][i]),
            "support": int(supports[i]),
        }
        for i, label in enumerate(labels)
    }

    return metrics


def save_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="Real",
        xlabel="Predito",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_multilabel_confusion_matrices(mcm: np.ndarray, labels: List[str], output_dir: str, prefix: str) -> None:
    for i, label in enumerate(labels):
        cm = mcm[i]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(2),
            yticks=np.arange(2),
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Real 0", "Real 1"],
            title=f"Matriz multilabel - {label}",
            ylabel="Real",
            xlabel="Predito",
        )
        thresh = cm.max() / 2.0 if cm.size else 0.5
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(
                    c, r, format(cm[r, c], "d"),
                    ha="center", va="center",
                    color="white" if cm[r, c] > thresh else "black"
                )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{prefix}_multilabel_confusion_{label}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_per_class_barplot(per_class: Dict[str, Dict[str, Any]], title: str, output_path: str) -> None:
    labels = list(per_class.keys())
    precisions = [per_class[k]["precision"] for k in labels]
    recalls = [per_class[k]["recall"] for k in labels]
    f1s = [per_class[k]["f1"] for k in labels]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x, recalls, width, label="Recall")
    ax.bar(x + width, f1s, width, label="F1")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_strategy_comparison_plot(df_plot: pd.DataFrame, task: str, metric: str, title: str, output_path: str) -> None:
    sub = df_plot[df_plot["task"] == task].copy()
    if sub.empty or metric not in sub.columns:
        return
    sub = sub.sort_values("prompt_strategy")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sub["prompt_strategy"], sub[metric])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _as_float_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except Exception:
        return None
    if np.isnan(score):
        return None
    return max(0.0, min(1.0, score))


def score_matrix_from_json_column(df: pd.DataFrame, column: str, labels: List[str]) -> Optional[np.ndarray]:
    if column not in df.columns:
        return None

    rows = []
    for value in df[column].tolist():
        parsed = safe_json_loads(value, default={})
        if not isinstance(parsed, dict):
            return None

        row = []
        for label in labels:
            score = _as_float_score(parsed.get(label, parsed.get(label.lower())))
            if score is None:
                return None
            row.append(score)
        rows.append(row)

    return np.array(rows, dtype=float) if rows else None


def score_matrix_from_probability_columns(df: pd.DataFrame, columns: List[str]) -> Optional[np.ndarray]:
    if any(column not in df.columns for column in columns):
        return None

    rows = []
    for _, row_data in df.iterrows():
        row = []
        for column in columns:
            score = _as_float_score(row_data.get(column))
            if score is None:
                return None
            row.append(score)
        rows.append(row)

    return np.array(rows, dtype=float) if rows else None


def score_matrix_from_json_list_column(df: pd.DataFrame, column: str, labels: List[str]) -> Optional[np.ndarray]:
    if column not in df.columns:
        return None

    rows = []
    for value in df[column].tolist():
        parsed = safe_json_loads(value, default=[])
        if not isinstance(parsed, list) or len(parsed) != len(labels):
            return None

        row = []
        for item in parsed:
            score = _as_float_score(item)
            if score is None:
                return None
            row.append(score)
        rows.append(row)

    return np.array(rows, dtype=float) if rows else None


def get_speech_score_matrix(df: pd.DataFrame) -> Optional[np.ndarray]:
    scores = score_matrix_from_json_column(df, "pred_speech_scores", SPEECH_LABELS)
    if scores is not None:
        return scores
    return score_matrix_from_probability_columns(
        df,
        ["prob_hateful", "prob_supportive", "prob_neutral"],
    )


def get_offensive_score_matrix(df: pd.DataFrame) -> Optional[np.ndarray]:
    scores = score_matrix_from_json_column(df, "pred_offensive_scores", OFFENSIVE_LABELS)
    if scores is not None:
        return scores
    return score_matrix_from_probability_columns(
        df,
        ["prob_offensive", "prob_non_offensive"],
    )


def get_aspect_score_matrix(df: pd.DataFrame) -> Optional[np.ndarray]:
    scores = score_matrix_from_json_column(df, "pred_aspect_scores", ASPECT_LABELS)
    if scores is not None:
        return scores
    return score_matrix_from_json_list_column(df, "pred_aspects_probs", ASPECT_LABELS)


def save_precision_recall_curves(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    title: str,
    output_prefix: str,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y_true_bin = label_binarize(y_true, classes=labels)
    y_pred_bin = label_binarize(y_pred, classes=labels)
    score_type = "continuous_score_one_vs_rest" if y_score is not None else "hard_label_one_vs_rest"
    y_score = y_score if y_score is not None else y_pred_bin

    if len(labels) == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        y_pred_bin = np.hstack([1 - y_pred_bin, y_pred_bin])
        if y_score.shape[1] == 1:
            y_score = np.hstack([1 - y_score, y_score])

    rows = []
    summary: Dict[str, Any] = {}

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(labels):
        precision, recall, thresholds = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        avg_precision = average_precision_score(y_true_bin[:, i], y_score[:, i])
        summary[label] = float(avg_precision)

        ax.plot(recall, precision, marker="o", label=f"{label} (AP={avg_precision:.3f})")

        for idx in range(len(precision)):
            rows.append({
                "class": label,
                "point_index": idx,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "threshold": float(thresholds[idx]) if idx < len(thresholds) else None,
                "average_precision": float(avg_precision),
                "score_type": score_type,
            })

    micro_ap = average_precision_score(y_true_bin.ravel(), y_score.ravel())
    precision_micro, recall_micro, thresholds_micro = precision_recall_curve(
        y_true_bin.ravel(), y_score.ravel()
    )
    summary["micro_average"] = float(micro_ap)
    ax.plot(recall_micro, precision_micro, linestyle="--", color="black", label=f"micro (AP={micro_ap:.3f})")

    for idx in range(len(precision_micro)):
        rows.append({
            "class": "micro_average",
            "point_index": idx,
            "precision": float(precision_micro[idx]),
            "recall": float(recall_micro[idx]),
            "threshold": float(thresholds_micro[idx]) if idx < len(thresholds_micro) else None,
            "average_precision": float(micro_ap),
            "score_type": score_type,
        })

    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisão")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize="small")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(f"{output_prefix}.csv", index=False, encoding="utf-8")
    with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
        json.dump({
            "average_precision": summary,
            "score_type": score_type,
            "note": "Curva calculada a partir de scores contínuos quando disponíveis. Na ausência de scores, usa rótulos preditos discretos em esquema one-vs-rest.",
        }, f, ensure_ascii=False, indent=2)
    return summary


def save_multilabel_precision_recall_curves(
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    labels: List[str],
    title: str,
    output_prefix: str,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    rows = []
    summary: Dict[str, Any] = {}
    score_type = "continuous_score_multilabel" if y_score is not None else "hard_label_multilabel"
    y_score = y_score if y_score is not None else y_pred_bin

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(labels):
        precision, recall, thresholds = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        avg_precision = average_precision_score(y_true_bin[:, i], y_score[:, i])
        summary[label] = float(avg_precision)

        ax.plot(recall, precision, marker="o", label=f"{label} (AP={avg_precision:.3f})")

        for idx in range(len(precision)):
            rows.append({
                "class": label,
                "point_index": idx,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "threshold": float(thresholds[idx]) if idx < len(thresholds) else None,
                "average_precision": float(avg_precision),
                "score_type": score_type,
            })

    micro_ap = average_precision_score(y_true_bin.ravel(), y_score.ravel())
    precision_micro, recall_micro, thresholds_micro = precision_recall_curve(
        y_true_bin.ravel(), y_score.ravel()
    )
    summary["micro_average"] = float(micro_ap)
    ax.plot(recall_micro, precision_micro, linestyle="--", color="black", label=f"micro (AP={micro_ap:.3f})")

    for idx in range(len(precision_micro)):
        rows.append({
            "class": "micro_average",
            "point_index": idx,
            "precision": float(precision_micro[idx]),
            "recall": float(recall_micro[idx]),
            "threshold": float(thresholds_micro[idx]) if idx < len(thresholds_micro) else None,
            "average_precision": float(micro_ap),
            "score_type": score_type,
        })

    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisão")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize="x-small")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(f"{output_prefix}.csv", index=False, encoding="utf-8")
    with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
        json.dump({
            "average_precision": summary,
            "score_type": score_type,
            "note": "Curva multilabel calculada a partir de scores contínuos quando disponíveis. Na ausência de scores, usa rótulos preditos discretos.",
        }, f, ensure_ascii=False, indent=2)
    return summary


def build_ragas_answer(row: pd.Series, tasks: List[str]) -> str:
    parts = []
    if "speech_class" in tasks and "pred_speech_class" in row:
        parts.append(f"speech_class={row.get('pred_speech_class')}")
    if "offensive_class" in tasks and "pred_offensive_class" in row:
        parts.append(f"offensive_class={row.get('pred_offensive_class')}")
    if "aspects" in tasks and "pred_aspects" in row:
        parts.append(f"aspects={row.get('pred_aspects')}")
    if "pred_justification" in row and pd.notna(row.get("pred_justification")):
        parts.append(f"justification={row.get('pred_justification')}")
    return "; ".join(parts)


def build_ragas_reference(row: pd.Series, tasks: List[str]) -> str:
    parts = []
    if "speech_class" in tasks and "speech_class" in row:
        parts.append(f"speech_class={row.get('speech_class')}")
    if "offensive_class" in tasks and "offensive_class" in row:
        parts.append(f"offensive_class={row.get('offensive_class')}")
    if "aspects" in tasks and "aspects" in row:
        parts.append(f"aspects={row.get('aspects')}")
    return "; ".join(parts)


def build_ragas_question(row: pd.Series, tasks: List[str]) -> str:
    title = row.get("video_title", row.get("title", ""))
    comment = row.get("comment_text", row.get("comment", row.get("text", "")))
    task_names = ", ".join(tasks)
    return (
        "Classifique o comentário em português brasileiro sobre pessoas trans. "
        f"Tarefas: {task_names}.\n"
        f"Título do vídeo: {title}\n"
        f"Comentário: {comment}"
    )


def export_ragas_dataset(df: pd.DataFrame, tasks: List[str], output_prefix: str, limit: Optional[int] = None) -> Optional[str]:
    if "retrieved_contexts" not in df.columns:
        return None

    ragas_df = df.copy()
    if limit is not None and limit > 0:
        ragas_df = ragas_df.head(limit).copy()

    rows = []
    for _, row in ragas_df.iterrows():
        contexts = safe_json_loads(row.get("retrieved_contexts"), default=[])
        if not contexts:
            continue
        contexts = [str(context) for context in contexts if str(context).strip()]
        if not contexts:
            continue

        question = build_ragas_question(row, tasks)
        answer = build_ragas_answer(row, tasks)
        reference = build_ragas_reference(row, tasks)

        rows.append({
            "id": row.get("id", row.get("comment_id", None)),
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": reference,
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": reference,
            "model_name": row.get("model_name"),
            "prompt_strategy": row.get("prompt_strategy"),
        })

    if not rows:
        return None

    output_path = f"{output_prefix}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pd.DataFrame(rows).to_csv(f"{output_prefix}.csv", index=False, encoding="utf-8")
    return output_path


def resolve_ragas_metrics(metric_names: List[str]) -> List[Any]:

    candidates = {
        "faithfulness": ["faithfulness", "Faithfulness"],
        "answer_relevancy": ["answer_relevancy", "ResponseRelevancy"],
        "response_relevancy": ["answer_relevancy", "ResponseRelevancy"],
        "context_precision": ["context_precision", "LLMContextPrecisionWithoutReference", "LLMContextPrecisionWithReference"],
        "context_recall": ["context_recall", "LLMContextRecall"],
        "noise_sensitivity": ["noise_sensitivity_relevant", "NoiseSensitivity"],
    }

    resolved = []
    missing = []
    for name in metric_names:
        found = None
        for attr in candidates.get(name, [name]):
            if hasattr(ragas_metrics, attr):
                candidate = getattr(ragas_metrics, attr)
                found = candidate() if isinstance(candidate, type) else candidate
                break
        if found is None:
            missing.append(name)
        else:
            resolved.append(found)

    if missing:
        raise ValueError(f"Métricas RAGAS não encontradas nesta versão instalada: {missing}")
    return resolved


def run_ragas_evaluation(
    ragas_jsonl_path: str,
    output_prefix: str,
    metric_names: List[str],
) -> None:
    try:
        rows = load_jsonl(ragas_jsonl_path).to_dict(orient="records")
        dataset = Dataset.from_list(rows)
        metrics = resolve_ragas_metrics(metric_names)
        result = ragas_evaluate(dataset, metrics=metrics)

        if hasattr(result, "to_pandas"):
            result_df = result.to_pandas()
        else:
            result_df = pd.DataFrame(result)

        result_df.to_csv(f"{output_prefix}_results.csv", index=False, encoding="utf-8")
        with open(f"{output_prefix}_results.json", "w", encoding="utf-8") as f:
            json.dump(result_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    except Exception as exc:
        with open(f"{output_prefix}_error.json", "w", encoding="utf-8") as f:
            json.dump({
                "error": str(exc),
                "note": "A avaliação RAGAS exige a biblioteca ragas, datasets e configuração de LLM/embeddings compatível com as métricas escolhidas. O dataset RAGAS foi exportado para execução posterior.",
            }, f, ensure_ascii=False, indent=2)


def save_latex_table(df: pd.DataFrame, output_path: str, caption: str, label: str, float_format="%.4f") -> None:
    latex = df.to_latex(index=False, float_format=float_format, caption=caption, label=label)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)


def evaluate_subset(
    df: pd.DataFrame,
    output_dir: str,
    prefix: str,
    tasks: List[str],
    enable_ragas: bool = False,
    ragas_limit: Optional[int] = None,
    ragas_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    ensure_dir(output_dir)

    metrics_by_task = {}
    json_metrics = {"num_samples": int(len(df))}
    ragas_metrics = ragas_metrics or []

    if "speech_class" in tasks:
        y_true_speech = df["speech_class"].tolist()
        y_pred_speech = df["pred_speech_class"].tolist()
        speech_metrics = metric_block_classification(y_true_speech, y_pred_speech, SPEECH_LABELS)
        speech_cm = confusion_matrix(y_true_speech, y_pred_speech, labels=SPEECH_LABELS)
        save_confusion_matrix(
            speech_cm,
            SPEECH_LABELS,
            f"Matriz de confusão - speech_class ({prefix})",
            os.path.join(output_dir, f"{prefix}_confusion_matrix_speech_class.png"),
        )
        save_per_class_barplot(
            speech_metrics["per_class"],
            f"Métricas por classe - speech_class ({prefix})",
            os.path.join(output_dir, f"{prefix}_per_class_speech_class.png"),
        )
        speech_metrics["average_precision"] = save_precision_recall_curves(
            y_true_speech,
            y_pred_speech,
            SPEECH_LABELS,
            f"Curva precisão-recall - speech_class ({prefix})",
            os.path.join(output_dir, f"{prefix}_precision_recall_speech_class"),
            y_score=get_speech_score_matrix(df),
        )
        metrics_by_task["speech_class"] = speech_metrics
        json_metrics["speech_class"] = speech_metrics

    if "offensive_class" in tasks:
        y_true_off = df["offensive_class"].tolist()
        y_pred_off = df["pred_offensive_class"].tolist()
        offensive_metrics = metric_block_classification(y_true_off, y_pred_off, OFFENSIVE_LABELS)
        offensive_cm = confusion_matrix(y_true_off, y_pred_off, labels=OFFENSIVE_LABELS)
        save_confusion_matrix(
            offensive_cm,
            OFFENSIVE_LABELS,
            f"Matriz de confusão - offensive_class ({prefix})",
            os.path.join(output_dir, f"{prefix}_confusion_matrix_offensive_class.png"),
        )
        save_per_class_barplot(
            offensive_metrics["per_class"],
            f"Métricas por classe - offensive_class ({prefix})",
            os.path.join(output_dir, f"{prefix}_per_class_offensive_class.png"),
        )
        offensive_metrics["average_precision"] = save_precision_recall_curves(
            y_true_off,
            y_pred_off,
            OFFENSIVE_LABELS,
            f"Curva precisão-recall - offensive_class ({prefix})",
            os.path.join(output_dir, f"{prefix}_precision_recall_offensive_class"),
            y_score=get_offensive_score_matrix(df),
        )
        metrics_by_task["offensive_class"] = offensive_metrics
        json_metrics["offensive_class"] = offensive_metrics

    if "aspects" in tasks:
        y_true_aspects = multilabel_binarize(df["aspects_true_list"], ASPECT_LABELS)
        y_pred_aspects = multilabel_binarize(df["aspects_pred_list"], ASPECT_LABELS)
        aspects_metrics = metric_block_multilabel(y_true_aspects, y_pred_aspects, ASPECT_LABELS)
        aspects_mcm = multilabel_confusion_matrix(y_true_aspects, y_pred_aspects)
        save_multilabel_confusion_matrices(aspects_mcm, ASPECT_LABELS, output_dir, prefix)
        save_per_class_barplot(
            aspects_metrics["per_class"],
            f"Métricas por rótulo - aspects ({prefix})",
            os.path.join(output_dir, f"{prefix}_per_class_aspects.png"),
        )
        aspects_metrics["average_precision"] = save_multilabel_precision_recall_curves(
            y_true_aspects,
            y_pred_aspects,
            ASPECT_LABELS,
            f"Curva precisão-recall - aspects ({prefix})",
            os.path.join(output_dir, f"{prefix}_precision_recall_aspects"),
            y_score=get_aspect_score_matrix(df),
        )
        metrics_by_task["aspects"] = aspects_metrics
        json_metrics["aspects"] = aspects_metrics

    ragas_prefix = os.path.join(output_dir, f"{prefix}_ragas")
    ragas_input_path = export_ragas_dataset(df, tasks, ragas_prefix, limit=ragas_limit)
    if ragas_input_path:
        json_metrics["ragas_input"] = ragas_input_path
        if enable_ragas:
            run_ragas_evaluation(ragas_input_path, ragas_prefix, ragas_metrics)
            json_metrics["ragas_results_prefix"] = ragas_prefix

    def row_from_metrics(task: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            "task": task,
            "num_samples": int(len(df)),
            "precision_micro": metrics.get("precision_micro"),
            "recall_micro": metrics.get("recall_micro"),
            "f1_micro": metrics.get("f1_micro"),
            "precision_macro": metrics.get("precision_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "f1_macro": metrics.get("f1_macro"),
            "precision_weighted": metrics.get("precision_weighted"),
            "recall_weighted": metrics.get("recall_weighted"),
            "f1_weighted": metrics.get("f1_weighted"),
        }
        for extra in [
            "accuracy", "balanced_accuracy", "precision_samples", "recall_samples", "f1_samples",
            "hamming_loss", "subset_zero_one_loss", "subset_accuracy",
            "jaccard_micro", "jaccard_macro", "jaccard_samples",
        ]:
            if extra in metrics:
                row[extra] = metrics.get(extra)
        if "average_precision" in metrics:
            row["average_precision_micro"] = metrics["average_precision"].get("micro_average")
        return row

    rows = [row_from_metrics(task, metrics) for task, metrics in metrics_by_task.items()]
    summary_df = pd.DataFrame(rows)

    summary_df.to_csv(os.path.join(output_dir, f"{prefix}_metrics_summary.csv"), index=False, encoding="utf-8")
    save_latex_table(
        summary_df,
        os.path.join(output_dir, f"{prefix}_metrics_summary.tex"),
        caption=f"Resultados para {prefix}",
        label=f"tab:{sanitize_name(prefix)}",
    )

    with open(os.path.join(output_dir, f"{prefix}_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_metrics, f, ensure_ascii=False, indent=2)

    return summary_df


def rebuild_overall_from_existing(by_model_dir: str, overall_dir: str, tasks: List[str]) -> None:
    ensure_dir(overall_dir)

    by_model_path = Path(by_model_dir)
    if not by_model_path.exists():
        return

    for model_dir in by_model_path.iterdir():
        if not model_dir.is_dir():
            continue

        model_name_safe = model_dir.name
        collected = []

        for strategy_dir in model_dir.iterdir():
            if not strategy_dir.is_dir():
                continue

            legacy_summary_name = f"{model_name_safe}_{strategy_dir.name}_metrics_summary.csv"
            csv_files = [
                path
                for path in sorted(strategy_dir.glob("*_metrics_summary.csv"))
                if path.name != legacy_summary_name
            ]
            if not csv_files:
                continue

            summary_frames = []
            for csv_file in csv_files:
                summary_frames.append(pd.read_csv(csv_file))

            summary_df = pd.concat(summary_frames, ignore_index=True)
            summary_df = summary_df[summary_df["task"].isin(tasks)].copy()
            if summary_df.empty:
                continue
            summary_df = summary_df.drop_duplicates(subset=["task"], keep="last")

            summary_df["model_name"] = model_name_safe
            summary_df["prompt_strategy"] = strategy_dir.name
            collected.append(summary_df)

        if not collected:
            continue

        model_comp_df = pd.concat(collected, ignore_index=True)
        model_comp_df = model_comp_df.sort_values(["prompt_strategy", "task"]).reset_index(drop=True)

        model_overall_dir = os.path.join(overall_dir, model_name_safe)
        ensure_dir(model_overall_dir)

        model_comp_df.to_csv(
            os.path.join(model_overall_dir, "strategy_comparison.csv"),
            index=False,
            encoding="utf-8",
        )
        save_latex_table(
            model_comp_df,
            os.path.join(model_overall_dir, "strategy_comparison.tex"),
            caption=f"Comparação entre estratégias para o modelo {model_name_safe}",
            label=f"tab:strategy_comparison_{model_name_safe}",
        )

        for task in tasks:
            save_strategy_comparison_plot(
                model_comp_df,
                task=task,
                metric="f1_macro",
                title=f"{model_name_safe} - comparação entre estratégias ({task}, F1 macro)",
                output_path=os.path.join(model_overall_dir, f"{task}_strategy_comparison_f1_macro.png"),
            )
            save_strategy_comparison_plot(
                model_comp_df,
                task=task,
                metric="recall_macro",
                title=f"{model_name_safe} - comparação entre estratégias ({task}, Recall macro)",
                output_path=os.path.join(model_overall_dir, f"{task}_strategy_comparison_recall_macro.png"),
            )
            save_strategy_comparison_plot(
                model_comp_df,
                task=task,
                metric="precision_macro",
                title=f"{model_name_safe} - comparação entre estratégias ({task}, Precision macro)",
                output_path=os.path.join(model_overall_dir, f"{task}_strategy_comparison_precision_macro.png"),
            )


def resolve_input_files(args: argparse.Namespace, tasks: List[str]) -> Dict[str, str]:
    if args.input:
        strategy_name = args.strategy
        if strategy_name == "all":
            strategy_name = "custom_input"
        return {strategy_name: args.input}

    strategies = DEFAULT_STRATEGIES if args.strategy == "all" else [args.strategy]
    invalid = [s for s in strategies if s not in DEFAULT_STRATEGIES]
    if invalid:
        valid = ", ".join(["all"] + sorted(DEFAULT_STRATEGIES))
        raise ValueError(f"EstratÃ©gia invÃ¡lida: {invalid[0]}. Use uma destas: {valid}")

    resolved = {}
    for strategy in strategies:
        for task in tasks:
            key = f"{strategy}_{task_folder(task)}"
            resolved[key] = strategy_prediction_path(args.predictions_dir, strategy, task, args.model)
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Avaliação parametrizável de speech, offensive e aspects.")
    parser.add_argument(
        "--input",
        default=None,
        help="Caminho opcional do JSONL ou CSV com predições. Se informado, avalia apenas esse arquivo.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_BASE_OUTPUT_DIR, help="Diretorio base de saida")
    parser.add_argument("--predictions-dir", default=DEFAULT_PREDICTIONS_DIR, help="Diretorio base das predicoes")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo usado nas predicoes")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", "speech", "speech_class", "offensive", "offensive_class", "aspects"],
        help="Quais tarefas avaliar. Exemplos: --tasks speech | --tasks offensive aspects | --tasks all",
    )
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all", *DEFAULT_STRATEGIES],
        help="Qual estratégia avaliar. Use all para avaliar todas as entradas padrão.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        default=True,
        help="Ignora arquivos de estratégia que não existirem.",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Executa a avaliação RAGAS para saídas RAG que contenham retrieved_contexts. O dataset RAGAS sempre é exportado quando os contextos existem.",
    )
    parser.add_argument(
        "--ragas-limit",
        type=int,
        default=None,
        help="Limita o número de amostras enviadas/exportadas para RAGAS.",
    )
    parser.add_argument(
        "--ragas-metrics",
        nargs="+",
        default=["faithfulness", "context_precision", "context_recall"],
        help="Métricas RAGAS a executar. Exemplos: faithfulness context_precision context_recall answer_relevancy noise_sensitivity.",
    )
    args = parser.parse_args()

    tasks = normalize_requested_tasks(args.tasks)

    base_output_dir = args.output_dir
    ensure_dir(base_output_dir)

    by_model_dir = os.path.join(base_output_dir, "by_model")
    overall_dir = os.path.join(base_output_dir, "overall")
    ensure_dir(by_model_dir)
    ensure_dir(overall_dir)

    input_files = resolve_input_files(args, tasks)
    total_valid_rows = 0
    evaluated_files = []

    for strategy_name, input_path in input_files.items():
        if not os.path.exists(input_path):
            if args.skip_missing:
                print(f"[AVISO] Arquivo não encontrado, ignorando: {input_path}")
                continue
            raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

        if input_path.lower().endswith(".jsonl"):
            df = load_jsonl(input_path)
        else:
            df = pd.read_csv(input_path)

        file_tasks = tasks_for_resolved_input(strategy_name, tasks)
        df = filter_valid_rows(df, file_tasks)
        total_valid_rows += len(df)
        evaluated_files.append(input_path)

        if "prompt_strategy" not in df.columns or df["prompt_strategy"].isna().all():
            df["prompt_strategy"] = strategy_name

        for model_name, model_df in df.groupby("model_name"):
            safe_model = sanitize_name(model_name)
            model_root = os.path.join(by_model_dir, safe_model)
            ensure_dir(model_root)

            for prompt_strategy, strat_df in model_df.groupby("prompt_strategy"):
                safe_strategy = sanitize_name(prompt_strategy)
                strategy_dir = os.path.join(model_root, safe_strategy)
                safe_task_scope = sanitize_name("_".join(file_tasks))
                evaluate_subset(
                    strat_df,
                    strategy_dir,
                    f"{safe_model}_{safe_strategy}_{safe_task_scope}",
                    file_tasks,
                    enable_ragas=args.ragas,
                    ragas_limit=args.ragas_limit,
                    ragas_metrics=args.ragas_metrics,
                )

    rebuild_overall_from_existing(by_model_dir, overall_dir, tasks)

    print(f"Avaliação salva em: {base_output_dir}")
    print(f"Tarefas avaliadas: {', '.join(tasks)}")
    print(f"Arquivos avaliados: {len(evaluated_files)}")
    for path in evaluated_files:
        print(f" - {path}")
    print(f"Número total de amostras válidas avaliadas: {total_valid_rows}")


if __name__ == "__main__":
    main()

