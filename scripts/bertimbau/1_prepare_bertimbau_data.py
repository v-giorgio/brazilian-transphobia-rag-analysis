"""
Prepara datasets supervisionados para fine-tuning com BERTimbau a partir de
./data/temp_dataset/dataset_unificado.csv

Saídas:
  ./pipeline/data/master/clean_dataset.csv
  ./pipeline/data/speech/{train,val,test}.csv
  ./pipeline/data/offensive/{train,val,test}.csv
  ./pipeline/data/aspects/{train,val,test}.csv
  ./pipeline/data/metadata/*.json

Decisões metodológicas implementadas:
- Texto de entrada = título + comentário
- Tarefas separadas:
    1) speech_class (multiclasse)
    2) offensive_class (binária)
    3) aspects (multilabel)
- Speech/offensive: split estratificado com scikit-learn
- Aspects: split com iterative stratification (multilabel)
- Cálculo de class weights / pos_weight para mitigar desbalanceamento
"""

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError as e:
    raise ImportError(
        "Pacote 'iterative-stratification' não encontrado. "
        "Instale com: pip install iterative-stratification"
    ) from e


# ==========================================================
# CONFIG
# ==========================================================

INPUT_PATH = "./data/dataset_final.csv"
PIPELINE_DIR = Path("./pipeline")

MASTER_DIR = PIPELINE_DIR / "data" / "master"
SPEECH_DIR = PIPELINE_DIR / "data" / "speech"
OFFENSIVE_DIR = PIPELINE_DIR / "data" / "offensive"
ASPECTS_DIR = PIPELINE_DIR / "data" / "aspects"
META_DIR = PIPELINE_DIR / "data" / "metadata"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10
TEXT_TEMPLATE = "[TÍTULO] {video_title} [COMENTÁRIO] {comment}"

ALLOWED_SPEECH = ["HATEFUL", "SUPPORTIVE", "NEUTRAL"]
ALLOWED_OFFENSIVE = ["OFFENSIVE", "NON_OFFENSIVE"]
ASPECT_LABELS = [
    "RELIGION",
    "POLITICS",
    "GENDER_DENIAL",
    "HARRASSMENT",
    "THREATENING",
    "PSEUDOSCIENCE",
]

MIN_SPEECH_CONFIDENCE = None
MIN_OFFENSIVE_CONFIDENCE = None
MIN_TOTAL_ANNOTATIONS = None


# ==========================================================
# HELPERS
# ==========================================================

def ensure_dirs() -> None:
    for d in [MASTER_DIR, SPEECH_DIR, OFFENSIVE_DIR, ASPECTS_DIR, META_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def normalize_aspects(aspects: Any) -> List[str]:
    if aspects is None:
        return []
    if isinstance(aspects, list):
        items = aspects
    elif isinstance(aspects, str):
        try:
            parsed = ast.literal_eval(aspects)
            items = parsed if isinstance(parsed, list) else [aspects]
        except Exception:
            items = [aspects]
    else:
        return []

    clean = []
    seen = set()
    for a in items:
        a = str(a).strip().upper()
        if a in ASPECT_LABELS and a not in seen:
            clean.append(a)
            seen.add(a)
    return clean


def validate_columns(df: pd.DataFrame) -> None:
    required = {
        "dataset_comment_id",
        "video_title",
        "comment",
        "speech_class",
        "offensive_class",
        "aspects",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")


def make_input_text(video_title: str, comment: str) -> str:
    return TEXT_TEMPLATE.format(
        video_title=str(video_title).strip(),
        comment=str(comment).strip(),
    )


def apply_optional_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if MIN_SPEECH_CONFIDENCE is not None and "speech_confidence" in out.columns:
        out = out[out["speech_confidence"] >= MIN_SPEECH_CONFIDENCE]

    if MIN_OFFENSIVE_CONFIDENCE is not None and "offensive_confidence" in out.columns:
        out = out[out["offensive_confidence"] >= MIN_OFFENSIVE_CONFIDENCE]

    if MIN_TOTAL_ANNOTATIONS is not None and "num_annotations" in out.columns:
        out = out[out["num_annotations"] >= MIN_TOTAL_ANNOTATIONS]

    return out


def prepare_master_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["speech_class"] = out["speech_class"].astype(str).str.strip().str.upper()
    out["offensive_class"] = out["offensive_class"].astype(str).str.strip().str.upper()
    out["aspects_list"] = out["aspects"].apply(normalize_aspects)
    out["text"] = out.apply(
        lambda row: make_input_text(row["video_title"], row["comment"]),
        axis=1,
    )

    out = out[out["speech_class"].isin(ALLOWED_SPEECH)]
    out = out[out["offensive_class"].isin(ALLOWED_OFFENSIVE)]
    out = out.drop_duplicates(subset=["dataset_comment_id"], keep="last").copy()
    out = out[out["text"].astype(str).str.strip().str.len() > 0].copy()

    return out


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def label_distribution(series: pd.Series) -> Dict[str, int]:
    counts = series.value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def multilabel_distribution(df: pd.DataFrame, label_cols: List[str]) -> Dict[str, int]:
    return {label: int(df[label].sum()) for label in label_cols}


def compute_class_weights_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    k = len(counts)
    weights = {}
    for label, n_c in counts.items():
        if n_c <= 0:
            weights[label] = 0.0
        else:
            weights[label] = float(total / (k * n_c))
    return weights


def compute_multilabel_pos_weights(df: pd.DataFrame, label_cols: List[str]) -> Dict[str, float]:
    weights = {}
    n = len(df)
    for label in label_cols:
        pos = int(df[label].sum())
        neg = n - pos
        if pos <= 0:
            weights[label] = 0.0
        else:
            weights[label] = float(neg / pos)
    return weights


def train_val_test_split_stratified(
    df: pd.DataFrame,
    stratify_col: str,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_col],
    )

    val_relative = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        random_state=random_state,
        stratify=train_val_df[stratify_col],
    )

    return train_df.copy(), val_df.copy(), test_df.copy()


def multilabel_iterative_split(
    df: pd.DataFrame,
    label_cols: List[str],
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[label_cols].values
    X_idx = np.arange(len(df)).reshape(-1, 1)

    msss_test = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_val_idx, test_idx = next(msss_test.split(X_idx, y))

    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    y_train_val = train_val_df[label_cols].values
    X_train_val_idx = np.arange(len(train_val_df)).reshape(-1, 1)

    val_relative = val_size / (1.0 - test_size)

    msss_val = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=val_relative,
        random_state=random_state,
    )
    train_idx_local, val_idx_local = next(msss_val.split(X_train_val_idx, y_train_val))

    train_df = train_val_df.iloc[train_idx_local].copy()
    val_df = train_val_df.iloc[val_idx_local].copy()

    return train_df, val_df, test_df


def save_split_csvs(task_dir: Path, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_df.to_csv(task_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(task_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(task_dir / "test.csv", index=False, encoding="utf-8")


# ==========================================================
# TASK BUILDERS
# ==========================================================

def build_speech_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset_comment_id",
        "text",
        "speech_class",
        "video_title",
        "comment",
        "num_annotations",
        "num_trans_annotations",
        "num_cis_annotations",
        "speech_confidence",
        "speech_rule",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    label_map = {
        "HATEFUL": 0,
        "SUPPORTIVE": 1,
        "NEUTRAL": 2,
    }
    out["label"] = out["speech_class"].map(label_map)
    return out


def build_offensive_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset_comment_id",
        "text",
        "offensive_class",
        "video_title",
        "comment",
        "num_annotations",
        "num_trans_annotations",
        "num_cis_annotations",
        "offensive_confidence",
        "offensive_rule",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    label_map = {
        "NON_OFFENSIVE": 0,
        "OFFENSIVE": 1,
    }
    out["label"] = out["offensive_class"].map(label_map)
    return out


def build_aspects_dataset(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "dataset_comment_id",
        "text",
        "video_title",
        "comment",
        "num_annotations",
        "num_trans_annotations",
        "num_cis_annotations",
        "aspect_support",
    ]
    base_cols = [c for c in base_cols if c in df.columns]
    out = df[base_cols].copy()

    for label in ASPECT_LABELS:
        out[label] = df["aspects_list"].apply(lambda xs: int(label in xs))

    out["num_positive_labels"] = out[ASPECT_LABELS].sum(axis=1)
    return out


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    ensure_dirs()

    df = pd.read_csv(INPUT_PATH)
    validate_columns(df)

    filtered_df = apply_optional_filters(df)
    master_df = prepare_master_dataframe(filtered_df)

    master_df.to_csv(MASTER_DIR / "clean_dataset.csv", index=False, encoding="utf-8")

    # SPEECH
    speech_df = build_speech_dataset(master_df)
    speech_train, speech_val, speech_test = train_val_test_split_stratified(
        speech_df,
        stratify_col="speech_class",
    )
    save_split_csvs(SPEECH_DIR, speech_train, speech_val, speech_test)

    speech_counts = label_distribution(speech_df["speech_class"])
    speech_weights = compute_class_weights_from_counts(speech_counts)

    save_json(
        {
            "task": "speech",
            "n_total": int(len(speech_df)),
            "class_distribution": speech_counts,
            "class_weights": speech_weights,
            "split_sizes": {
                "train": int(len(speech_train)),
                "val": int(len(speech_val)),
                "test": int(len(speech_test)),
            },
            "train_distribution": label_distribution(speech_train["speech_class"]),
            "val_distribution": label_distribution(speech_val["speech_class"]),
            "test_distribution": label_distribution(speech_test["speech_class"]),
            "label_map": {"HATEFUL": 0, "SUPPORTIVE": 1, "NEUTRAL": 2},
        },
        META_DIR / "speech_metadata.json",
    )

    # OFFENSIVE
    offensive_df = build_offensive_dataset(master_df)
    off_train, off_val, off_test = train_val_test_split_stratified(
        offensive_df,
        stratify_col="offensive_class",
    )
    save_split_csvs(OFFENSIVE_DIR, off_train, off_val, off_test)

    offensive_counts = label_distribution(offensive_df["offensive_class"])
    offensive_weights = compute_class_weights_from_counts(offensive_counts)

    save_json(
        {
            "task": "offensive",
            "n_total": int(len(offensive_df)),
            "class_distribution": offensive_counts,
            "class_weights": offensive_weights,
            "split_sizes": {
                "train": int(len(off_train)),
                "val": int(len(off_val)),
                "test": int(len(off_test)),
            },
            "train_distribution": label_distribution(off_train["offensive_class"]),
            "val_distribution": label_distribution(off_val["offensive_class"]),
            "test_distribution": label_distribution(off_test["offensive_class"]),
            "label_map": {"NON_OFFENSIVE": 0, "OFFENSIVE": 1},
        },
        META_DIR / "offensive_metadata.json",
    )

    # ASPECTS
    aspects_df = build_aspects_dataset(master_df)
    asp_train, asp_val, asp_test = multilabel_iterative_split(
        aspects_df,
        label_cols=ASPECT_LABELS,
    )
    save_split_csvs(ASPECTS_DIR, asp_train, asp_val, asp_test)

    pos_weights = compute_multilabel_pos_weights(aspects_df, ASPECT_LABELS)

    save_json(
        {
            "task": "aspects",
            "n_total": int(len(aspects_df)),
            "label_distribution": multilabel_distribution(aspects_df, ASPECT_LABELS),
            "pos_weights": pos_weights,
            "split_sizes": {
                "train": int(len(asp_train)),
                "val": int(len(asp_val)),
                "test": int(len(asp_test)),
            },
            "train_distribution": multilabel_distribution(asp_train, ASPECT_LABELS),
            "val_distribution": multilabel_distribution(asp_val, ASPECT_LABELS),
            "test_distribution": multilabel_distribution(asp_test, ASPECT_LABELS),
            "num_positive_labels_distribution": {
                "all": label_distribution(aspects_df["num_positive_labels"]),
                "train": label_distribution(asp_train["num_positive_labels"]),
                "val": label_distribution(asp_val["num_positive_labels"]),
                "test": label_distribution(asp_test["num_positive_labels"]),
            },
            "label_order": ASPECT_LABELS,
        },
        META_DIR / "aspects_metadata.json",
    )

    save_json(
        {
            "input_path": INPUT_PATH,
            "n_input_rows": int(len(df)),
            "n_after_optional_filters": int(len(filtered_df)),
            "n_clean_rows": int(len(master_df)),
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "random_state": RANDOM_STATE,
            "text_template": TEXT_TEMPLATE,
            "tasks": ["speech", "offensive", "aspects"],
            "scientific_choices": {
                "speech_offensive_split": "stratified random split",
                "aspects_split": "iterative stratification for multilabel",
                "imbalance_metadata": [
                    "class weights for single-label tasks",
                    "pos_weight for multilabel task",
                ],
            },
        },
        META_DIR / "dataset_preparation_summary.json",
    )

    print("Preparação concluída com sucesso.")
    print(f"Master dataset: {MASTER_DIR / 'clean_dataset.csv'}")
    print(f"Speech splits: {SPEECH_DIR}")
    print(f"Offensive splits: {OFFENSIVE_DIR}")
    print(f"Aspects splits: {ASPECTS_DIR}")
    print(f"Metadata: {META_DIR}")


if __name__ == "__main__":
    main()
