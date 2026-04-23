import os
import ast
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    multilabel_confusion_matrix,
    hamming_loss,
    jaccard_score,
    zero_one_loss,
)

# DEFAULT_INPUT = "./data/temp_dataset/output/classifier_llm_zero_shot_predictions.jsonl"
# DEFAULT_INPUT = "./data/temp_dataset/output/classifier_llm_few_shot_predictions.jsonl"
# DEFAULT_INPUT = "./data/temp_dataset/output/classifier_llm_rag_transcripts_predictions.jsonl"
# DEFAULT_INPUT = "./data/temp_dataset/output/classifier_llm_rag_wiki_predictions.jsonl"
DEFAULT_INPUT = "./data/temp_dataset/output/classifier_llm_rag_both_predictions.jsonl"
DEFAULT_BASE_OUTPUT_DIR = "./data/temp_dataset/output/metrics"

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


def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "speech_class",
        "offensive_class",
        "aspects",
        "pred_speech_class",
        "pred_offensive_class",
        "pred_aspects",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")

    out = df.copy()

    if "error" in out.columns:
        out = out[out["error"].isna()].copy()

    out = out[
        out["speech_class"].notna()
        & out["offensive_class"].notna()
        & out["pred_speech_class"].notna()
        & out["pred_offensive_class"].notna()
    ].copy()

    out["speech_class"] = out["speech_class"].astype(str).str.strip().str.upper()
    out["pred_speech_class"] = out["pred_speech_class"].astype(str).str.strip().str.upper()
    out["offensive_class"] = out["offensive_class"].astype(str).str.strip().str.upper()
    out["pred_offensive_class"] = out["pred_offensive_class"].astype(str).str.strip().str.upper()

    out = out[
        out["speech_class"].isin(SPEECH_LABELS)
        & out["pred_speech_class"].isin(SPEECH_LABELS)
        & out["offensive_class"].isin(OFFENSIVE_LABELS)
        & out["pred_offensive_class"].isin(OFFENSIVE_LABELS)
    ].copy()

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


def save_latex_table(df: pd.DataFrame, output_path: str, caption: str, label: str, float_format="%.4f") -> None:
    latex = df.to_latex(index=False, float_format=float_format, caption=caption, label=label)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)


def evaluate_subset(df: pd.DataFrame, output_dir: str, prefix: str) -> pd.DataFrame:
    ensure_dir(output_dir)

    y_true_speech = df["speech_class"].tolist()
    y_pred_speech = df["pred_speech_class"].tolist()
    speech_metrics = metric_block_classification(y_true_speech, y_pred_speech, SPEECH_LABELS)
    speech_cm = confusion_matrix(y_true_speech, y_pred_speech, labels=SPEECH_LABELS)
    save_confusion_matrix(
        speech_cm, SPEECH_LABELS,
        f"Matriz de confusão - speech_class ({prefix})",
        os.path.join(output_dir, f"{prefix}_confusion_matrix_speech_class.png"),
    )
    save_per_class_barplot(
        speech_metrics["per_class"],
        f"Métricas por classe - speech_class ({prefix})",
        os.path.join(output_dir, f"{prefix}_per_class_speech_class.png"),
    )

    y_true_off = df["offensive_class"].tolist()
    y_pred_off = df["pred_offensive_class"].tolist()
    offensive_metrics = metric_block_classification(y_true_off, y_pred_off, OFFENSIVE_LABELS)
    offensive_cm = confusion_matrix(y_true_off, y_pred_off, labels=OFFENSIVE_LABELS)
    save_confusion_matrix(
        offensive_cm, OFFENSIVE_LABELS,
        f"Matriz de confusão - offensive_class ({prefix})",
        os.path.join(output_dir, f"{prefix}_confusion_matrix_offensive_class.png"),
    )
    save_per_class_barplot(
        offensive_metrics["per_class"],
        f"Métricas por classe - offensive_class ({prefix})",
        os.path.join(output_dir, f"{prefix}_per_class_offensive_class.png"),
    )

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

    rows = []

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
        for extra in ["accuracy", "balanced_accuracy", "precision_samples", "recall_samples", "f1_samples",
                      "hamming_loss", "subset_zero_one_loss", "subset_accuracy",
                      "jaccard_micro", "jaccard_macro", "jaccard_samples"]:
            if extra in metrics:
                row[extra] = metrics.get(extra)
        return row

    rows.append(row_from_metrics("speech_class", speech_metrics))
    rows.append(row_from_metrics("offensive_class", offensive_metrics))
    rows.append(row_from_metrics("aspects", aspects_metrics))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(output_dir, f"{prefix}_metrics_summary.csv"), index=False, encoding="utf-8")
    save_latex_table(
        summary_df,
        os.path.join(output_dir, f"{prefix}_metrics_summary.tex"),
        caption=f"Resultados para {prefix}",
        label=f"tab:{sanitize_name(prefix)}",
    )

    with open(os.path.join(output_dir, f"{prefix}_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "speech_class": speech_metrics,
            "offensive_class": offensive_metrics,
            "aspects": aspects_metrics,
            "num_samples": int(len(df)),
        }, f, ensure_ascii=False, indent=2)

    return summary_df


def rebuild_overall_from_existing(by_model_dir: str, overall_dir: str) -> None:
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

            csv_files = sorted(strategy_dir.glob("*_metrics_summary.csv"))
            if not csv_files:
                continue

            # pega o summary gerado nessa pasta de estratégia
            summary_df = pd.read_csv(csv_files[0])
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

        for task in ["speech_class", "offensive_class", "aspects"]:
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


def main():
    parser = argparse.ArgumentParser(description="Avaliação de classificação multiclass, binária e multilabel.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Caminho do JSONL ou CSV com predições")
    parser.add_argument("--output-dir", default=DEFAULT_BASE_OUTPUT_DIR, help="Diretório base de saída")
    args = parser.parse_args()

    base_output_dir = args.output_dir
    ensure_dir(base_output_dir)

    if args.input.lower().endswith(".jsonl"):
        df = load_jsonl(args.input)
    else:
        df = pd.read_csv(args.input)

    df = filter_valid_rows(df)

    by_model_dir = os.path.join(base_output_dir, "by_model")
    overall_dir = os.path.join(base_output_dir, "overall")
    ensure_dir(by_model_dir)
    ensure_dir(overall_dir)

    # Atualiza só os diretórios impactados pelo arquivo atual
    for model_name, model_df in df.groupby("model_name"):
        safe_model = sanitize_name(model_name)
        model_root = os.path.join(by_model_dir, safe_model)
        ensure_dir(model_root)

        for prompt_strategy, strat_df in model_df.groupby("prompt_strategy"):
            safe_strategy = sanitize_name(prompt_strategy)
            strategy_dir = os.path.join(model_root, safe_strategy)
            evaluate_subset(strat_df, strategy_dir, f"{safe_model}_{safe_strategy}")

    # Reconstroi overall a partir de tudo que já existe em by_model
    rebuild_overall_from_existing(by_model_dir, overall_dir)

    print(f"Avaliação salva em: {base_output_dir}")
    print(f"Número de amostras válidas: {len(df)}")


if __name__ == "__main__":
    main()
