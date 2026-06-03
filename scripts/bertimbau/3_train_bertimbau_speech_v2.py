"""
Treino supervisionado de speech_class com BERTimbau.

Permite comparar dois formatos de entrada:
1. comment_only: usa apenas o comentário.
2. title_comment: usa título do vídeo + comentário.

Entradas esperadas:
  ./pipeline/data/speech/train.csv
  ./pipeline/data/speech/val.csv
  ./pipeline/data/speech/test.csv
  ./pipeline/data/metadata/speech_metadata.json

Saídas padrão:
  ./pipeline/models/speech_<input_format>/
  ./pipeline/outputs/speech_<input_format>/

python train_bertimbau_speech_v2.py \
--input-format comment_only \
--learning-rate 2e-5 \
--num-epochs 5 \
--train-batch-size 16 \
--eval-batch-size 16 \
--class-weight-mode balanced

python train_bertimbau_speech_v2.py \
  --input-format title_comment \
  --learning-rate 2e-5 \
  --num-epochs 5 \
  --train-batch-size 16 \
  --eval-batch-size 16 \
  --class-weight-mode balanced
"""

import argparse
import inspect
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
)
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed,
)

DEFAULT_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

DEFAULT_TRAIN_PATH = "./pipeline/data/speech/train.csv"
DEFAULT_VAL_PATH = "./pipeline/data/speech/val.csv"
DEFAULT_TEST_PATH = "./pipeline/data/speech/test.csv"
DEFAULT_META_PATH = "./pipeline/data/metadata/speech_metadata.json"

DEFAULT_MODEL_BASE_DIR = "./pipeline/models"
DEFAULT_OUTPUT_BASE_DIR = "./pipeline/outputs"

TEXT_COL = "text"
TITLE_COL = "video_title"
COMMENT_COL = "comment"
LABEL_COL = "label"

ID2LABEL = {
    0: "HATEFUL",
    1: "SUPPORTIVE",
    2: "NEUTRAL",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def ensure_dirs(model_dir: str, output_dir: str) -> None:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_report() -> Dict[str, Any]:
    report = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "device_name": None,
        "cuda_version": getattr(torch.version, "cuda", None),
        "hip_version": getattr(torch.version, "hip", None),
    }

    if torch.cuda.is_available():
        report["device_name"] = torch.cuda.get_device_name(0)
        try:
            props = torch.cuda.get_device_properties(0)
            report["total_vram_gb"] = round(props.total_memory / (1024 ** 3), 3)
        except Exception:
            pass

    return report


def print_device_report(report: Dict[str, Any]) -> None:
    print("\n========== DEVICE REPORT ==========")
    for k, v in report.items():
        print(f"{k}: {v}")
    print("===================================\n")

    if not report["cuda_available"]:
        print(
            "[AVISO] PyTorch não detectou GPU CUDA/ROCm. "
            "O fine-tuning deve rodar em CPU. "
            "No Windows com AMD Radeon, isso é esperado na maioria dos ambientes.\n"
        )


def validate_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, input_format: str) -> None:
    required_cols = {COMMENT_COL, LABEL_COL}

    if input_format == "title_comment":
        required_cols.add(TITLE_COL)

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Split {split_name} sem colunas obrigatórias: {sorted(missing)}")
        if df.empty:
            raise ValueError(f"Split {split_name} está vazio.")


def build_input_text(row: pd.Series, input_format: str) -> str:
    comment = str(row[COMMENT_COL])

    if input_format == "comment_only":
        return comment

    if input_format == "title_comment":
        title = str(row[TITLE_COL])
        return f"Título do vídeo: {title}\nComentário: {comment}"

    raise ValueError(f"input_format inválido: {input_format}")


def build_class_weights(
    metadata: Dict[str, Any],
    train_df: pd.DataFrame,
    class_weight_mode: str,
    custom_class_weights: Optional[List[float]],
) -> torch.Tensor:
    if class_weight_mode == "none":
        return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float)

    if class_weight_mode == "custom":
        if custom_class_weights is None or len(custom_class_weights) != 3:
            raise ValueError(
                "Para --class-weight-mode custom, informe 3 valores: "
                "--custom-class-weights HATEFUL SUPPORTIVE NEUTRAL"
            )
        return torch.tensor([float(x) for x in custom_class_weights], dtype=torch.float)

    if class_weight_mode != "balanced":
        raise ValueError(f"class_weight_mode inválido: {class_weight_mode}")

    class_weights_dict = metadata.get("class_weights", {})
    if all(label in class_weights_dict for label in ["HATEFUL", "SUPPORTIVE", "NEUTRAL"]):
        weights = [
            float(class_weights_dict["HATEFUL"]),
            float(class_weights_dict["SUPPORTIVE"]),
            float(class_weights_dict["NEUTRAL"]),
        ]
    else:
        counts = train_df[LABEL_COL].astype(int).value_counts().to_dict()
        n = len(train_df)
        num_classes = 3
        weights = []
        for label_id in [0, 1, 2]:
            c = counts.get(label_id, 0)
            weights.append(1.0 if c == 0 else n / (num_classes * c))

    return torch.tensor(weights, dtype=torch.float)


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, input_format: str):
        self.texts = df.apply(lambda row: build_input_text(row, input_format), axis=1).astype(str).tolist()
        self.labels = df[LABEL_COL].astype(int).tolist()
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {k: self.encodings[k][idx] for k in self.encodings}
        item["labels"] = self.labels[idx]
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], zero_division=0
    )

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "precision_hateful": float(precision_per_class[0]),
        "recall_hateful": float(recall_per_class[0]),
        "f1_hateful": float(f1_per_class[0]),
        "support_hateful": int(support_per_class[0]),
        "precision_supportive": float(precision_per_class[1]),
        "recall_supportive": float(recall_per_class[1]),
        "f1_supportive": float(f1_per_class[1]),
        "support_supportive": int(support_per_class[1]),
        "precision_neutral": float(precision_per_class[2]),
        "recall_neutral": float(recall_per_class[2]),
        "f1_neutral": float(f1_per_class[2]),
        "support_neutral": int(support_per_class[2]),
    }


def save_confusion_matrix(y_true: List[int], y_pred: List[int], path: str, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    matrix_df = pd.DataFrame(
        cm,
        index=["true_HATEFUL", "true_SUPPORTIVE", "true_NEUTRAL"],
        columns=["pred_HATEFUL", "pred_SUPPORTIVE", "pred_NEUTRAL"],
    )
    matrix_df.to_csv(str(Path(path).with_suffix(".csv")), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)

    labels = ["HATEFUL", "SUPPORTIVE", "NEUTRAL"]
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_predictions_df(df: pd.DataFrame, preds: np.ndarray, probs: np.ndarray, input_texts: List[str]) -> pd.DataFrame:
    out = df.copy()
    out["model_input_text"] = input_texts
    out["pred_label"] = preds.astype(int)
    out["pred_speech_class"] = out["pred_label"].map(ID2LABEL)

    for label_id, label_name in ID2LABEL.items():
        out[f"prob_{label_name.lower()}"] = probs[:, label_id]

    return out


def safe_float_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            clean[k] = float(v)
        else:
            clean[k] = v
    return clean


def predict_and_save_split(
    trainer: Trainer,
    dataset: SpeechDataset,
    df: pd.DataFrame,
    output_dir: str,
    split_name: str,
    input_format: str,
) -> Dict[str, Any]:
    preds_output = trainer.predict(dataset)
    logits = preds_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(logits, axis=1)

    pred_df = build_predictions_df(df, preds, probs, input_texts=dataset.texts)
    pred_path = f"{output_dir}/{split_name}_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    metrics = compute_metrics((logits, df[LABEL_COL].astype(int).to_numpy()))
    metrics = safe_float_metrics({f"{split_name}_{k}": v for k, v in metrics.items()})
    metrics["input_format"] = input_format

    metrics_path = f"{output_dir}/metrics_{split_name}_predictions.json"
    save_json(metrics, metrics_path)

    return {
        "predictions_path": pred_path,
        "metrics_path": metrics_path,
    }


def make_training_args(args: argparse.Namespace, device_report: Dict[str, Any], n_train: int) -> TrainingArguments:
    supports = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    kwargs = {
        "output_dir": args.model_dir,
        "overwrite_output_dir": True,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.num_epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": args.seed,
        "report_to": "none",
        "dataloader_pin_memory": bool(device_report["cuda_available"]),
        "fp16": bool(args.fp16 and device_report["cuda_available"]),
    }

    if "warmup_steps" in supports:
        steps_per_epoch = int(np.ceil(n_train / args.train_batch_size))
        total_steps = int(np.ceil(steps_per_epoch * args.num_epochs))
        kwargs["warmup_steps"] = int(total_steps * args.warmup_ratio)
    elif "warmup_ratio" in supports:
        kwargs["warmup_ratio"] = args.warmup_ratio

    if "eval_strategy" in supports:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    if "use_cpu" in supports:
        kwargs["use_cpu"] = not bool(device_report["cuda_available"])

    filtered = {k: v for k, v in kwargs.items() if k in supports}
    return TrainingArguments(**filtered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning BERTimbau para speech_class com formatos de entrada comparáveis.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--meta-path", default=DEFAULT_META_PATH)

    parser.add_argument(
        "--input-format",
        choices=["comment_only", "title_comment"],
        required=True,
        help="comment_only usa apenas o comentário; title_comment concatena título do vídeo + comentário.",
    )

    parser.add_argument("--model-base-dir", default=DEFAULT_MODEL_BASE_DIR)
    parser.add_argument("--output-base-dir", default=DEFAULT_OUTPUT_BASE_DIR)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=float, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", default=False)

    parser.add_argument(
        "--class-weight-mode",
        choices=["balanced", "none", "custom"],
        default="balanced",
        help="balanced usa metadata; none usa [1,1,1]; custom usa --custom-class-weights.",
    )
    parser.add_argument(
        "--custom-class-weights",
        nargs=3,
        type=float,
        default=None,
        metavar=("HATEFUL", "SUPPORTIVE", "NEUTRAL"),
        help="Exemplo: --custom-class-weights 1.0 1.5 1.0",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_name = f"speech_finetuned_{args.input_format}"

    if args.model_dir is None:
        args.model_dir = str(Path(args.model_base_dir) / run_name)
    if args.output_dir is None:
        args.output_dir = str(Path(args.output_base_dir) / run_name)

    ensure_dirs(args.model_dir, args.output_dir)
    set_all_seeds(args.seed)

    run_start = time.perf_counter()

    device_report = get_device_report()
    print_device_report(device_report)
    save_json(device_report, f"{args.output_dir}/device_report.json")

    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    test_df = pd.read_csv(args.test_path)
    metadata = load_json(args.meta_path)

    validate_splits(train_df, val_df, test_df, args.input_format)

    class_weights = build_class_weights(
        metadata=metadata,
        train_df=train_df,
        class_weight_mode=args.class_weight_mode,
        custom_class_weights=args.custom_class_weights,
    )

    print(f"Input format: {args.input_format}")
    print(f"Model dir: {args.model_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Class weight mode: {args.class_weight_mode}")
    print(f"Class weights: {class_weights.tolist()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = SpeechDataset(train_df, tokenizer, args.max_length, args.input_format)
    val_dataset = SpeechDataset(val_df, tokenizer, args.max_length, args.input_format)
    test_dataset = SpeechDataset(test_df, tokenizer, args.max_length, args.input_format)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = make_training_args(args, device_report, n_train=len(train_df))

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    }

    trainer_init_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(class_weights=class_weights, **trainer_kwargs)

    train_result = trainer.train()

    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    val_metrics = safe_float_metrics(trainer.evaluate(eval_dataset=val_dataset))
    val_metrics["input_format"] = args.input_format
    save_json(val_metrics, f"{args.output_dir}/metrics_val.json")

    test_metrics = safe_float_metrics(trainer.evaluate(eval_dataset=test_dataset))
    test_metrics["input_format"] = args.input_format
    save_json(test_metrics, f"{args.output_dir}/metrics_test.json")

    val_prediction_files = predict_and_save_split(
        trainer=trainer,
        dataset=val_dataset,
        df=val_df,
        output_dir=args.output_dir,
        split_name="val",
        input_format=args.input_format,
    )

    test_prediction_files = predict_and_save_split(
        trainer=trainer,
        dataset=test_dataset,
        df=test_df,
        output_dir=args.output_dir,
        split_name="test",
        input_format=args.input_format,
    )

    test_pred_df = pd.read_csv(f"{args.output_dir}/test_predictions.csv")
    test_preds = test_pred_df["pred_label"].astype(int).to_numpy()

    save_confusion_matrix(
        y_true=test_df[LABEL_COL].astype(int).tolist(),
        y_pred=test_preds.tolist(),
        path=f"{args.output_dir}/confusion_matrix_test.png",
        title=f"Confusion Matrix - Speech - {args.input_format}",
    )

    total_seconds = time.perf_counter() - run_start

    summary = {
        "model_name": args.model_name,
        "task": "speech",
        "input_format": args.input_format,
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
        "class_weight_mode": args.class_weight_mode,
        "custom_class_weights": args.custom_class_weights,
        "class_weights": {
            "HATEFUL": float(class_weights[0]),
            "SUPPORTIVE": float(class_weights[1]),
            "NEUTRAL": float(class_weights[2]),
        },
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "best_metric": "f1_macro",
        "train_runtime_seconds": float(train_result.metrics.get("train_runtime", 0.0)) if hasattr(train_result, "metrics") else None,
        "total_script_runtime_seconds": round(float(total_seconds), 6),
        "val_prediction_files": val_prediction_files,
        "test_prediction_files": test_prediction_files,
        "device_report": device_report,
    }
    save_json(summary, f"{args.output_dir}/training_summary.json")

    print("Treino concluído com sucesso.")
    print(f"Modelo salvo em: {args.model_dir}")
    print(f"Saídas salvas em: {args.output_dir}")
    print(f"Tempo total do script: {total_seconds:.2f}s")


if __name__ == "__main__":
    main()
