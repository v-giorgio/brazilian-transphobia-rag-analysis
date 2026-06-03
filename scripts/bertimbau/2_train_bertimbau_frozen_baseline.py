"""
Baseline com BERTimbau congelado como extrator de embeddings.

Este script nao faz fine-tuning do BERTimbau. Ele extrai representacoes
do token [CLS] e treina um SVM linear calibrado sobre os embeddings.

Exemplos:
python 2_train_bertimbau_frozen_baseline.py --task speech --input-format title_comment
python 2_train_bertimbau_frozen_baseline.py --task offensive
python 2_train_bertimbau_frozen_baseline.py --task aspects
"""

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer, set_seed

DEFAULT_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

TEXT_COL = "text"
TITLE_COL = "video_title"
COMMENT_COL = "comment"
LABEL_COL = "label"

SPEECH_ID2LABEL = {
    0: "HATEFUL",
    1: "SUPPORTIVE",
    2: "NEUTRAL",
}

OFFENSIVE_ID2LABEL = {
    0: "NON_OFFENSIVE",
    1: "OFFENSIVE",
}

ASPECT_LABELS = [
    "RELIGION",
    "POLITICS",
    "GENDER_DENIAL",
    "HARRASSMENT",
    "THREATENING",
    "PSEUDOSCIENCE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina baseline com BERTimbau congelado + classificador tradicional."
    )
    parser.add_argument("--task", choices=["speech", "offensive", "aspects"], required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input-format", choices=["comment_only", "title_comment"], default="comment_only")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_default_paths(task: str, input_format: str) -> Tuple[Path, Path, Path, Path, Path]:
    data_dir = Path("./pipeline/data") / task
    suffix = f"{task}_frozen_{input_format}"
    output_dir = Path("./pipeline/outputs") / suffix
    model_dir = Path("./pipeline/models") / suffix
    return (
        data_dir / "train.csv",
        data_dir / "val.csv",
        data_dir / "test.csv",
        output_dir,
        model_dir,
    )


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, Path]:
    default_train, default_val, default_test, default_output, default_model = get_default_paths(
        args.task,
        args.input_format,
    )
    return (
        Path(args.train_path) if args.train_path else default_train,
        Path(args.val_path) if args.val_path else default_val,
        Path(args.test_path) if args.test_path else default_test,
        Path(args.output_dir) if args.output_dir else default_output,
        Path(args.model_dir) if args.model_dir else default_model,
    )


def validate_splits(task: str, input_format: str, splits: Dict[str, pd.DataFrame]) -> None:
    required = {COMMENT_COL}
    if task in {"speech", "offensive"}:
        required.add(LABEL_COL)
    else:
        required.update(ASPECT_LABELS)
    if input_format == "title_comment":
        required.add(TITLE_COL)

    for split_name, df in splits.items():
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Split {split_name} sem colunas obrigatorias: {sorted(missing)}")
        if df.empty:
            raise ValueError(f"Split {split_name} esta vazio.")


def build_input_text(row: pd.Series, input_format: str) -> str:
    comment = str(row[COMMENT_COL])
    if input_format == "comment_only":
        return comment
    title = str(row[TITLE_COL])
    return f"Titulo do video: {title}\nComentario: {comment}"


def extract_embeddings(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    input_format: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    texts = df.apply(lambda row: build_input_text(row, input_format), axis=1).astype(str).tolist()
    embeddings = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


def build_calibrated_linear_svm(max_iter: int):
    svm = LinearSVC(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=42,
    )
    return CalibratedClassifierCV(
        svm,
        method="sigmoid",
        cv=3,
    )


def classifier_multiclass(max_iter: int):
    return make_pipeline(
        StandardScaler(),
        build_calibrated_linear_svm(max_iter),
    )


def classifier_multilabel(max_iter: int):
    base = build_calibrated_linear_svm(max_iter)
    return make_pipeline(StandardScaler(), OneVsRestClassifier(base))


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int], label_names: List[str]) -> Dict[str, Any]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    for idx, name in enumerate(label_names):
        key = name.lower()
        metrics[f"precision_{key}"] = float(precision_per[idx])
        metrics[f"recall_{key}"] = float(recall_per[idx])
        metrics[f"f1_{key}"] = float(f1_per[idx])
        metrics[f"support_{key}"] = int(support_per[idx])

    return metrics


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    confusion_by_label = {}
    metrics: Dict[str, Any] = {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }

    for idx, label in enumerate(ASPECT_LABELS):
        cm = confusion_matrix(y_true[:, idx], y_pred[:, idx], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        confusion_by_label[label] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
        }
        metrics[f"f1_{label.lower()}"] = float(f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0))
        metrics[f"precision_{label.lower()}"] = float(
            precision_score(y_true[:, idx], y_pred[:, idx], zero_division=0)
        )
        metrics[f"recall_{label.lower()}"] = float(recall_score(y_true[:, idx], y_pred[:, idx], zero_division=0))
        metrics[f"support_{label.lower()}"] = int(y_true[:, idx].sum())

    metrics["confusion_matrix_by_label"] = confusion_by_label
    return metrics


def save_multilabel_confusion_matrix_csv(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    rows = []
    for idx, label in enumerate(ASPECT_LABELS):
        cm = confusion_matrix(y_true[:, idx], y_pred[:, idx], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "label": label,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")


def save_speech_predictions(df: pd.DataFrame, pred: np.ndarray, prob: np.ndarray, output_path: Path) -> None:
    out = df.copy()
    out["pred_label"] = pred
    out["pred_speech_class"] = out["pred_label"].map(SPEECH_ID2LABEL)
    for label_id, label_name in SPEECH_ID2LABEL.items():
        out[f"prob_{label_name.lower()}"] = prob[:, label_id]
    out.to_csv(output_path, index=False, encoding="utf-8")


def save_offensive_predictions(df: pd.DataFrame, pred: np.ndarray, prob: np.ndarray, output_path: Path) -> None:
    out = df.copy()
    out["pred_label"] = pred
    out["pred_offensive_class"] = out["pred_label"].map(OFFENSIVE_ID2LABEL)
    out["prob_non_offensive"] = prob[:, 0]
    out["prob_offensive"] = prob[:, 1]
    out.to_csv(output_path, index=False, encoding="utf-8")


def tune_offensive_threshold(
    y_true: np.ndarray,
    prob_offensive: np.ndarray,
    metric: str = "macro_f1",
) -> Dict[str, Any]:
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    best = {
        "threshold": 0.5,
        "score": -1.0,
    }
    candidates = []

    for threshold in thresholds:
        pred = (prob_offensive >= threshold).astype(int)

        if metric == "f1_offensive":
            score = f1_score(y_true, pred, pos_label=1, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, pred)
        elif metric == "macro_f1":
            score = f1_score(y_true, pred, average="macro", zero_division=0)
        else:
            raise ValueError(f"Metrica de threshold invalida: {metric}")

        precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
            y_true,
            pred,
            labels=[0, 1],
            average=None,
            zero_division=0,
        )
        candidate = {
            "threshold": float(threshold),
            "score": float(score),
            "precision_non_offensive": float(precision_per[0]),
            "recall_non_offensive": float(recall_per[0]),
            "f1_non_offensive": float(f1_per[0]),
            "support_non_offensive": int(support_per[0]),
            "precision_offensive": float(precision_per[1]),
            "recall_offensive": float(recall_per[1]),
            "f1_offensive": float(f1_per[1]),
            "support_offensive": int(support_per[1]),
        }
        candidates.append(candidate)

        if score > best["score"]:
            best = {
                "threshold": float(threshold),
                "score": float(score),
            }

    return {
        "metric": metric,
        "default_threshold": 0.5,
        "selected_threshold": best["threshold"],
        "validation_score": best["score"],
        "candidates": candidates,
    }


def save_aspects_predictions(df: pd.DataFrame, pred: np.ndarray, prob: np.ndarray, output_path: Path) -> None:
    out = df.copy()
    out["pred_aspects"] = [
        json.dumps([label for label, value in zip(ASPECT_LABELS, row) if int(value) == 1], ensure_ascii=False)
        for row in pred
    ]
    out["pred_aspects_binary"] = [json.dumps(row.astype(int).tolist()) for row in pred]
    out["pred_aspects_probs"] = [json.dumps(row.tolist()) for row in prob]
    out.to_csv(output_path, index=False, encoding="utf-8")


def tune_aspect_thresholds(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    thresholds = np.round(np.arange(0.01, 0.951, 0.01), 2)
    selected_thresholds = {}
    validation_scores = {}
    candidates = {}

    for idx, label in enumerate(ASPECT_LABELS):
        label_true = y_true[:, idx]
        label_prob = probs[:, idx]
        best_threshold = 0.5
        best_score = -1.0
        label_candidates = []

        for threshold in thresholds:
            pred = (label_prob >= threshold).astype(int)
            precision = precision_score(label_true, pred, zero_division=0)
            recall = recall_score(label_true, pred, zero_division=0)
            f1 = f1_score(label_true, pred, zero_division=0)
            candidate = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(label_true.sum()),
                "predicted_positive": int(pred.sum()),
            }
            label_candidates.append(candidate)

            if f1 > best_score:
                best_score = float(f1)
                best_threshold = float(threshold)

        selected_thresholds[label] = best_threshold
        validation_scores[label] = best_score
        candidates[label] = label_candidates

    return {
        "metric": "per_label_f1",
        "default_threshold": 0.5,
        "selected_thresholds": selected_thresholds,
        "validation_scores": validation_scores,
        "candidates": candidates,
    }


def apply_aspect_thresholds(probs: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    threshold_values = np.array([float(thresholds[label]) for label in ASPECT_LABELS])
    return (probs >= threshold_values).astype(int)


def predict_proba_multilabel(model, embeddings: np.ndarray) -> np.ndarray:
    clf = model.named_steps["onevsrestclassifier"]
    transformed = model.named_steps["standardscaler"].transform(embeddings)
    return np.array([est.predict_proba(transformed)[:, 1] for est in clf.estimators_]).T


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    train_path, val_path, test_path, output_dir, model_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    validate_splits(args.task, args.input_format, splits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name).to(device)

    print(f"Extraindo embeddings em {device}...")
    train_x = extract_embeddings(train_df, tokenizer, encoder, device, args.input_format, args.max_length, args.batch_size)
    val_x = extract_embeddings(val_df, tokenizer, encoder, device, args.input_format, args.max_length, args.batch_size)
    test_x = extract_embeddings(test_df, tokenizer, encoder, device, args.input_format, args.max_length, args.batch_size)

    if args.task == "aspects":
        train_y = train_df[ASPECT_LABELS].astype(int).values
        val_y = val_df[ASPECT_LABELS].astype(int).values
        test_y = test_df[ASPECT_LABELS].astype(int).values
        clf = classifier_multilabel(args.max_iter)
    else:
        train_y = train_df[LABEL_COL].astype(int).values
        val_y = val_df[LABEL_COL].astype(int).values
        test_y = test_df[LABEL_COL].astype(int).values
        clf = classifier_multiclass(args.max_iter)

    print("Treinando classificador tradicional sobre embeddings congelados...")
    clf.fit(train_x, train_y)

    with open(model_dir / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    run_config = {
        "task": args.task,
        "model_name": args.model_name,
        "input_format": args.input_format,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "classifier": "Calibrated LinearSVC" if args.task != "aspects" else "OneVsRest(Calibrated LinearSVC)",
        "bertimbau_trainable": False,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "output_dir": str(output_dir),
        "model_dir": str(model_dir),
    }
    save_json(run_config, output_dir / "run_config.json")

    offensive_threshold = 0.5
    if args.task == "offensive":
        val_prob = clf.predict_proba(val_x)
        threshold_info = tune_offensive_threshold(
            y_true=val_y,
            prob_offensive=val_prob[:, 1],
            metric="macro_f1",
        )
        offensive_threshold = float(threshold_info["selected_threshold"])
        save_json(threshold_info, output_dir / "threshold_tuning.json")
        print(
            "Threshold offensive selecionado na validacao: "
            f"{offensive_threshold:.2f} "
            f"(macro-F1={threshold_info['validation_score']:.4f})"
        )

    aspect_thresholds = {label: 0.5 for label in ASPECT_LABELS}
    if args.task == "aspects":
        val_prob = predict_proba_multilabel(clf, val_x)
        threshold_info = tune_aspect_thresholds(
            y_true=val_y,
            probs=val_prob,
        )
        aspect_thresholds = {
            label: float(value)
            for label, value in threshold_info["selected_thresholds"].items()
        }
        save_json(threshold_info, output_dir / "threshold_tuning.json")
        print("Thresholds de aspects selecionados na validacao:")
        for label in ASPECT_LABELS:
            print(
                f"  {label}: {aspect_thresholds[label]:.2f} "
                f"(F1={threshold_info['validation_scores'][label]:.4f})"
            )

    for split_name, df, x, y in [
        ("val", val_df, val_x, val_y),
        ("test", test_df, test_x, test_y),
    ]:
        if args.task == "aspects":
            prob = predict_proba_multilabel(clf, x)
            pred = apply_aspect_thresholds(prob, aspect_thresholds)
            save_aspects_predictions(df, pred, prob, output_dir / f"{split_name}_predictions.csv")
            save_multilabel_confusion_matrix_csv(
                y_true=y,
                y_pred=pred,
                output_path=output_dir / f"confusion_matrix_{split_name}.csv",
            )
            metrics = multilabel_metrics(y, pred)
            metrics["decision_thresholds"] = aspect_thresholds
            metrics["threshold_metric"] = "per_label_f1"
        else:
            pred = clf.predict(x)
            prob = clf.predict_proba(x)
            if args.task == "speech":
                save_speech_predictions(df, pred, prob, output_dir / f"{split_name}_predictions.csv")
                metrics = multiclass_metrics(y, pred, [0, 1, 2], [SPEECH_ID2LABEL[i] for i in [0, 1, 2]])
            else:
                pred = (prob[:, 1] >= offensive_threshold).astype(int)
                save_offensive_predictions(df, pred, prob, output_dir / f"{split_name}_predictions.csv")
                metrics = multiclass_metrics(y, pred, [0, 1], [OFFENSIVE_ID2LABEL[i] for i in [0, 1]])
                metrics["decision_threshold"] = offensive_threshold
                metrics["threshold_metric"] = "macro_f1"

        save_json(metrics, output_dir / f"metrics_{split_name}_predictions.json")
        print(f"{split_name}: {json.dumps(metrics, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
