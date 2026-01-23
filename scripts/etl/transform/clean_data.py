import re
import pandas as pd
from pathlib import Path

# =========================
# PATHS
# =========================

RAW_DIR = Path("../../../data/manual_selection")
CLEAN_DIR = Path("../../../data/cleaned_comments")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# TEXT CLEANING FUNCTIONS
# =========================

def to_lowercase(text: str) -> str:
    """
    Converte todo o texto para letras minúsculas. 
    Obj.: Reduzir a variabilidade textual causada por diferenças de capitalização, facilitando a comparação e o aprendizado dos modelos de PLN.
    """
    return text.lower()


def normalize_whitespace(text: str) -> str:
    """
    Normaliza espaços em branco no texto.
    Obj.: Substituir múltiplos espaços, quebras de linha e tabulações por um único espaço, além de remover espaços no início e no fim do texto.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_urls(text: str) -> str:
    """
    Remove URLs do texto.
    Obj.: Eliminar links externos que não contribuem semanticamente para a análise do conteúdo textual e podem introduzir ruído no modelo.
    """
    return re.sub(r"http\S+|www\S+", "", text)


def remove_timestamps(text: str) -> str:
    """
    Remove URLs do texto.
    Obj.: Eliminar links externos que não contribuem semanticamente para a análise do conteúdo textual e podem introduzir ruído no modelo.
    """
    return re.sub(r"\b\d{1,2}:\d{2}\b", "", text)


def normalize_punctuation(text: str) -> str:
    """
    Normaliza pontuação repetida.
    Obj.: Reduzir sequências exageradas de pontuação (ex.: '!!!', '???', '...') para apenas um caractere, mantendo a intenção discursiva sem ruído.
    """
    text = re.sub(r"([!?\.]){2,}", r"\1", text)
    return text


def normalize_elongated_words(text: str) -> str:
    """
    Normaliza palavras com caracteres excessivamente repetidos.
    Obj.: Reduzir alongamentos artificiais de palavras (ex.: 'nooooossa', 'ameeeei') para uma forma mais padronizada, 
    preservando o tom emocional sem prejudicar a análise lexical.
    """
    return re.sub(r"(.)\1{3,}", r"\1\1", text)


def remove_non_linguistic_symbols(text: str) -> str:
    """
    Remove símbolos não linguísticos do texto.
    Obj.: Eliminar caracteres que não fazem parte da linguagem escrita natural (emojis, símbolos gráficos, etc.), 
    mantendo letras acentuadas do português e sinais básicos de pontuação.
    """
    return re.sub(r"[^\w\sà-úÀ-Ú.,!?]", "", text)


def remove_short_comments(text: str, min_chars: int = 15) -> str | None:
    """
    Remove comentários excessivamente curtos.
    Obj.: Filtrar textos com baixa densidade informacional, que tendem a ser semanticamente ambíguos ou pouco úteis
    para tarefas de classificação (ex.: 'kk', 'nao', 'ruim').
    """
    if len(text) < min_chars:
        return None
    return text


def limit_comment_length(
    text: str,
    max_sentences: int = 5,
    max_words: int = 200
) -> str:
    """
    Limita o tamanho de comentários textuais excessivamente longos.
    Obj.: Reduzir ruído e padronizar o tamanho dos textos, preservando unidades linguísticas completas 
    (frases) e o conteúdo semântico principal. A estratégia combina limitação por número de frases 
    e, adicionalmente, por número máximo de palavras, evitando perda abrupta de contexto.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    text_limited = " ".join(sentences[:max_sentences])

    words = text_limited.split()
    if len(words) > max_words:
        text_limited = " ".join(words[:max_words])

    return text_limited


# =========================
# DATAFRAME NORMALIZATION
# =========================

def normalize_and_add_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia comment_clean para comment e adiciona colunas faltantes conforme o schema final do dataset.
    Obj.: Garantir que o DataFrame final possua todas as colunas necessárias para análises subsequentes,
    mesmo que algumas delas não estejam presentes nos dados originais.
    """
    if "comment_clean" in df.columns:
        df = df.rename(columns={"comment_clean": "comment"})

    required_columns = {
        # Anotação manual (anotadores humanos)
        "attitude_manual": pd.NA,
        "offensive_manual": pd.NA,
        "subject_manual": pd.NA,

        # Distribuição de probabilidades – atitude discursiva
        "attitude_dist_supportive": pd.NA,
        "attitude_dist_neutral": pd.NA,
        "attitude_dist_hateful": pd.NA,

        # Distribuição de probabilidades – ofensividade
        "offensive_dist_non_offensive": pd.NA,
        "offensive_dist_offensive": pd.NA,

        # Distribuição de probabilidades – cunho discursivo
        "subject_dist_religion": pd.NA,
        "subject_dist_politics": pd.NA,
        "subject_dist_gender_denial": pd.NA,
        "subject_dist_harrassment": pd.NA,
        "subject_dist_threatening": pd.NA,
        "subject_dist_pseudoscience": pd.NA,
    }

    for col, default_value in required_columns.items():
        if col not in df.columns:
            df[col] = default_value

    return df


# =========================
# MAIN PIPELINE
# =========================

def clean_text(text: str) -> str | None:
    if not isinstance(text, str):
        return None

    # -- NORMALIZATION --
    text = remove_urls(text)
    text = remove_timestamps(text)
    text = remove_non_linguistic_symbols(text)

    # -- CLEANING --
    text = to_lowercase(text)
    text = normalize_whitespace(text)
    text = normalize_punctuation(text)
    text = normalize_elongated_words(text)

    # -- FILTER & LIMIT LENGTH --
    text = limit_comment_length(text)
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

    df = df.drop(columns=["source", "video_id", "comment"])

    df = normalize_and_add_columns(df)

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
