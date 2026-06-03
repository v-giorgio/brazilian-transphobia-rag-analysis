"""
Treino supervisionado de aspects (multilabel) com BERTimbau.

Entradas esperadas:
  ./pipeline/data/aspects/train.csv
  ./pipeline/data/aspects/val.csv
  ./pipeline/data/aspects/test.csv
  ./pipeline/data/metadata/aspects_metadata.json

Saídas:
  ./pipeline/models/aspects/
  ./pipeline/outputs/aspects/

Dependências:
  pip install torch transformers scikit-learn pandas matplotlib
"""

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
)
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ==========================================================
# CONFIG
# ==========================================================

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

TRAIN_PATH = "./pipeline/data/aspects/train.csv"
VAL_PATH = "./pipeline/data/aspects/val.csv"
TEST_PATH = "./pipeline/data/aspects/test.csv"
META_PATH = "./pipeline/data/metadata/aspects_metadata.json"

MODEL_BASE_DIR = "./pipeline/models"
OUTPUT_BASE_DIR = "./pipeline/outputs"

TEXT_COL = "text"
TITLE_COL = "video_title"
COMMENT_COL = "comment"
LABEL_COLS = [
    "RELIGION",
    "POLITICS",
    "GENDER_DENIAL",
    "HARRASSMENT",
    "THREATENING",
    "PSEUDOSCIENCE",
]

MAX_LENGTH = 256
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 6
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# threshold inicial; depois você pode calibrar por label no val
DEFAULT_THRESHOLD = 0.5
PER_LABEL_THRESHOLDS = {
    "RELIGION": 0.5,
    "POLITICS": 0.5,
    "GENDER_DENIAL": 0.5,
    "HARRASSMENT": 0.5,
    "THREATENING": 0.5,
    "PSEUDOSCIENCE": 0.5,
}


# ==========================================================
# HELPERS
# ==========================================================

def ensure_dirs(model_dir: str, output_dir: str) -> None:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_input_text(row: pd.Series, input_format: str) -> str:
    comment = str(row[COMMENT_COL])
    if input_format == "comment_only":
        return comment
    if input_format == "title_comment":
        title = str(row[TITLE_COL])
        return f"Título do vídeo: {title}\nComentário: {comment}"
    raise ValueError(f"input_format inválido: {input_format}")


def validate_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, input_format: str) -> None:
    required_cols = {COMMENT_COL, *LABEL_COLS}
    if input_format == "title_comment":
        required_cols.add(TITLE_COL)

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Split {split_name} sem colunas obrigatórias: {sorted(missing)}")
        if df.empty:
            raise ValueError(f"Split {split_name} está vazio.")


class AspectsDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, input_format: str):
        self.texts = df.apply(lambda row: build_input_text(row, input_format), axis=1).astype(str).tolist()
        self.labels = df[LABEL_COLS].astype(float).values
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class BertForMultiLabel(torch.nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


class WeightedMultiLabelTrainer(Trainer):
    def __init__(self, pos_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def apply_thresholds(probs: np.ndarray, thresholds_by_label: Dict[str, float] = None) -> np.ndarray:
    thresholds_source = thresholds_by_label or PER_LABEL_THRESHOLDS
    thresholds = np.array([thresholds_source[label] for label in LABEL_COLS], dtype=np.float32)
    return (probs >= thresholds).astype(int)


def multilabel_metrics(labels: np.ndarray, preds: np.ndarray, include_confusion: bool = True) -> Dict[str, Any]:
    result = {
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(labels, preds, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(labels, preds, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(labels, preds)),
        "subset_accuracy": float(np.all(labels == preds, axis=1).mean()),
    }

    confusion_by_label = {}
    for i, label in enumerate(LABEL_COLS):
        cm = confusion_matrix(labels[:, i], preds[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        confusion_by_label[label] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
        }
        result[f"precision_{label.lower()}"] = float(precision_score(labels[:, i], preds[:, i], zero_division=0))
        result[f"recall_{label.lower()}"] = float(recall_score(labels[:, i], preds[:, i], zero_division=0))
        result[f"f1_{label.lower()}"] = float(f1_score(labels[:, i], preds[:, i], zero_division=0))
        result[f"support_{label.lower()}"] = int(labels[:, i].sum())

    if include_confusion:
        result["confusion_matrix_by_label"] = confusion_by_label
    return result


def tune_aspect_thresholds(labels: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    thresholds = np.round(np.arange(0.01, 0.951, 0.01), 2)
    selected_thresholds = {}
    validation_scores = {}
    candidates = {}

    for idx, label in enumerate(LABEL_COLS):
        label_true = labels[:, idx]
        label_prob = probs[:, idx]
        best_threshold = 0.5
        best_score = -1.0
        label_candidates = []

        for threshold in thresholds:
            pred = (label_prob >= threshold).astype(int)
            precision = precision_score(label_true, pred, zero_division=0)
            recall = recall_score(label_true, pred, zero_division=0)
            f1 = f1_score(label_true, pred, zero_division=0)
            label_candidates.append(
                {
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "support": int(label_true.sum()),
                    "predicted_positive": int(pred.sum()),
                }
            )
            if f1 > best_score:
                best_score = float(f1)
                best_threshold = float(threshold)

        selected_thresholds[label] = best_threshold
        validation_scores[label] = best_score
        candidates[label] = label_candidates

    return {
        "metric": "per_label_f1",
        "default_threshold": DEFAULT_THRESHOLD,
        "selected_thresholds": selected_thresholds,
        "validation_scores": validation_scores,
        "candidates": candidates,
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = apply_thresholds(probs)
    return multilabel_metrics(labels, preds, include_confusion=False)


def build_predictions_df(
    df: pd.DataFrame,
    logits: np.ndarray,
    input_texts: List[str],
    thresholds_by_label: Dict[str, float],
) -> pd.DataFrame:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = apply_thresholds(probs, thresholds_by_label=thresholds_by_label)

    out = df.copy()
    out["model_input_text"] = input_texts
    out["pred_aspects"] = [
        json.dumps([LABEL_COLS[j] for j, v in enumerate(row) if v == 1], ensure_ascii=False)
        for row in preds
    ]
    out["pred_aspects_binary"] = [json.dumps(row.tolist()) for row in preds]
    out["pred_aspects_probs"] = [json.dumps(row) for row in probs.tolist()]
    return out


def save_multilabel_confusion_matrix_csv(labels: np.ndarray, preds: np.ndarray, path: str) -> None:
    rows = []
    for idx, label in enumerate(LABEL_COLS):
        cm = confusion_matrix(labels[:, idx], preds[:, idx], labels=[0, 1])
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
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def save_label_barplots(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, path: str) -> None:
    train_counts = [int(train_df[col].sum()) for col in LABEL_COLS]
    val_counts = [int(val_df[col].sum()) for col in LABEL_COLS]
    test_counts = [int(test_df[col].sum()) for col in LABEL_COLS]

    x = np.arange(len(LABEL_COLS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, train_counts, width, label="train")
    ax.bar(x, val_counts, width, label="val")
    ax.bar(x + width, test_counts, width, label="test")

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_COLS, rotation=25)
    ax.set_ylabel("Positive examples")
    ax.set_title("Aspect label distribution by split")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==========================================================
# MAIN
# ==========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning BERTimbau para aspects multilabel.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--val-path", default=VAL_PATH)
    parser.add_argument("--test-path", default=TEST_PATH)
    parser.add_argument("--meta-path", default=META_PATH)
    parser.add_argument(
        "--input-format",
        choices=["comment_only", "title_comment"],
        required=True,
        help="comment_only usa apenas o comentario; title_comment concatena titulo do video + comentario.",
    )
    parser.add_argument("--model-base-dir", default=MODEL_BASE_DIR)
    parser.add_argument("--output-base-dir", default=OUTPUT_BASE_DIR)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--num-epochs", type=float, default=NUM_EPOCHS)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument(
        "--predict-only",
        action="store_true",
        default=False,
        help="Carrega o modelo salvo em --model-dir e gera avaliacoes/predicoes sem retreinar.",
    )
    return parser.parse_args()


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    supports = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    kwargs = {
        "output_dir": args.model_dir,
        "overwrite_output_dir": True,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.num_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": args.seed,
        "report_to": "none",
    }

    if "eval_strategy" in supports:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    filtered = {k: v for k, v in kwargs.items() if k in supports}
    return TrainingArguments(**filtered)


def load_saved_model_weights(model: torch.nn.Module, model_dir: str) -> None:
    model_path = Path(model_dir) / "model.safetensors"
    if not model_path.exists():
        model_path = Path(model_dir) / "pytorch_model.bin"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Nenhum peso salvo encontrado em {model_dir}. "
            "Esperado model.safetensors ou pytorch_model.bin."
        )

    if model_path.suffix == ".safetensors":
        state_dict = load_safetensors(str(model_path))
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[AVISO] Pesos ausentes ao carregar modelo salvo: {missing}")
    if unexpected:
        print(f"[AVISO] Pesos inesperados ao carregar modelo salvo: {unexpected}")


def main():
    args = parse_args()

    run_name = f"aspects_finetuned_{args.input_format}"
    if args.model_dir is None:
        args.model_dir = str(Path(args.model_base_dir) / run_name)
    if args.output_dir is None:
        args.output_dir = str(Path(args.output_base_dir) / run_name)

    ensure_dirs(args.model_dir, args.output_dir)
    set_all_seeds(args.seed)

    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    test_df = pd.read_csv(args.test_path)
    metadata = load_json(args.meta_path)

    validate_splits(train_df, val_df, test_df, args.input_format)
    required_cols = set()
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Split {split_name} sem colunas obrigatórias: {sorted(missing)}")

    pos_weights_dict = metadata.get("pos_weights", {})
    pos_weight = torch.tensor(
        [float(pos_weights_dict[label]) for label in LABEL_COLS],
        dtype=torch.float,
    )

    tokenizer_source = args.model_dir if args.predict_only and (Path(args.model_dir) / "tokenizer.json").exists() else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    train_dataset = AspectsDataset(train_df, tokenizer, args.max_length, args.input_format)
    val_dataset = AspectsDataset(val_df, tokenizer, args.max_length, args.input_format)
    test_dataset = AspectsDataset(test_df, tokenizer, args.max_length, args.input_format)

    model = BertForMultiLabel(args.model_name, num_labels=len(LABEL_COLS))
    if args.predict_only:
        print(f"Carregando modelo salvo de: {args.model_dir}")
        load_saved_model_weights(model, args.model_dir)

    training_args = make_training_args(args)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    }

    trainer_init_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedMultiLabelTrainer(pos_weight=pos_weight, **trainer_kwargs)

    if not args.predict_only:
        trainer.train()
        trainer.save_model(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
    else:
        print("Modo predict-only ativo: treino ignorado.")

    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    val_metrics = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in val_metrics.items()}
    val_metrics["input_format"] = args.input_format
    save_json(val_metrics, f"{args.output_dir}/metrics_val.json")

    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    test_metrics = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in test_metrics.items()}
    test_metrics["input_format"] = args.input_format
    save_json(test_metrics, f"{args.output_dir}/metrics_test.json")

    val_output = trainer.predict(val_dataset)
    val_logits = val_output.predictions
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    val_labels = val_df[LABEL_COLS].astype(int).values
    threshold_info = tune_aspect_thresholds(val_labels, val_probs)
    selected_thresholds = {
        label: float(value)
        for label, value in threshold_info["selected_thresholds"].items()
    }
    save_json(threshold_info, f"{args.output_dir}/threshold_tuning.json")

    val_preds = apply_thresholds(val_probs, thresholds_by_label=selected_thresholds)
    val_pred_df = build_predictions_df(
        val_df,
        val_logits,
        input_texts=val_dataset.texts,
        thresholds_by_label=selected_thresholds,
    )
    val_pred_df.to_csv(f"{args.output_dir}/val_predictions.csv", index=False, encoding="utf-8")
    val_prediction_metrics = multilabel_metrics(val_labels, val_preds, include_confusion=True)
    val_prediction_metrics["decision_thresholds"] = selected_thresholds
    val_prediction_metrics["threshold_metric"] = "per_label_f1"
    save_json(val_prediction_metrics, f"{args.output_dir}/metrics_val_predictions.json")
    save_multilabel_confusion_matrix_csv(
        val_labels,
        val_preds,
        f"{args.output_dir}/confusion_matrix_val.csv",
    )

    test_output = trainer.predict(test_dataset)
    test_logits = test_output.predictions
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    test_labels = test_df[LABEL_COLS].astype(int).values
    test_preds = apply_thresholds(test_probs, thresholds_by_label=selected_thresholds)

    pred_df = build_predictions_df(
        test_df,
        test_logits,
        input_texts=test_dataset.texts,
        thresholds_by_label=selected_thresholds,
    )
    pred_df.to_csv(f"{args.output_dir}/test_predictions.csv", index=False, encoding="utf-8")
    test_prediction_metrics = multilabel_metrics(test_labels, test_preds, include_confusion=True)
    test_prediction_metrics["decision_thresholds"] = selected_thresholds
    test_prediction_metrics["threshold_metric"] = "per_label_f1"
    save_json(test_prediction_metrics, f"{args.output_dir}/metrics_test_predictions.json")
    save_multilabel_confusion_matrix_csv(
        test_labels,
        test_preds,
        f"{args.output_dir}/confusion_matrix_test.csv",
    )

    save_label_barplots(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        path=f"{args.output_dir}/label_distribution.png",
    )

    summary = {
        "model_name": args.model_name,
        "task": "aspects",
        "input_format": args.input_format,
        "label_order": LABEL_COLS,
        "text_column": COMMENT_COL,
        "title_column": TITLE_COL if args.input_format == "title_comment" else None,
        "model_dir": args.model_dir,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_epochs": args.num_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "default_threshold": DEFAULT_THRESHOLD,
        "per_label_thresholds": selected_thresholds,
        "pos_weight": {label: float(pos_weight[i]) for i, label in enumerate(LABEL_COLS)},
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "best_metric": "macro_f1",
    }
    save_json(summary, f"{args.output_dir}/training_summary.json")

    print("Treino concluído com sucesso.")
    print(f"Modelo salvo em: {args.model_dir}")
    print(f"Saídas salvas em: {args.output_dir}")


if __name__ == "__main__":
    main()
