import re
import pandas as pd
from pathlib import Path

# =========================
# PATHS
# =========================

RAW_DIR = Path("../../../data/raw_comments")
CLEAN_DIR = Path("../../../data/cleaned_comments")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# TEXT CLEANING FUNCTIONS
# =========================

def to_lowercase(text: str) -> str:
    return text.lower()


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\S+", "", text)


def remove_timestamps(text: str) -> str:
    return re.sub(r"\b\d{1,2}:\d{2}\b", "", text)


def normalize_punctuation(text: str) -> str:
    text = re.sub(r"([!?\.]){2,}", r"\1", text)
    return text


def normalize_elongated_words(text: str) -> str:
    return re.sub(r"(.)\1{3,}", r"\1\1", text)


def remove_non_linguistic_symbols(text: str) -> str:
    return re.sub(r"[^\w\sà-úÀ-Ú.,!?]", "", text)


def remove_short_comments(text: str, min_chars: int = 10) -> str | None:
    if len(text) < min_chars:
        return None
    return text


# =========================
# MAIN PIPELINE
# =========================

def clean_text(text: str) -> str | None:
    if not isinstance(text, str):
        return None

    # -- NORMALIZATION --
    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_timestamps(text)

    # -- CLEANING --
    text = normalize_whitespace(text)
    text = normalize_punctuation(text)
    text = normalize_elongated_words(text)
    text = remove_non_linguistic_symbols(text)

    # -- FILTER --
    text = normalize_whitespace(text)
    text = remove_short_comments(text)

    return text


# =========================
# FILE PROCESSING
# =========================

def process_file(csv_path: Path):
    print(f"Processando {csv_path.name}")

    df = pd.read_csv(csv_path)

    if "comment" not in df.columns:
        print("Skip (sem 'comment')")
        return

    df["comment_clean"] = df["comment"].apply(clean_text)
    df = df.dropna(subset=["comment_clean"])

    output_path = CLEAN_DIR / csv_path.name
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Salvo {len(df)} comentários limpos -> {output_path.name}")


def main():
    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print("Nenhum CSV encontrado.")
        return

    for csv_file in csv_files:
        process_file(csv_file)


if __name__ == "__main__":
    main()
