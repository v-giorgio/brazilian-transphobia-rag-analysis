import os
import re
import json
import math
import time
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

# =====================================
# CONFIG
# =====================================

TITLES_PATH = "../data/titles/video_titles.csv"
METADATA_PATH = "../data/wiki/metadata.jsonl"

OUTPUT_DOC_LEVEL_PATH = "../data/wiki/eval_doc_level.csv"
OUTPUT_VIDEO_LEVEL_PATH = "../data/wiki/eval_video_level.csv"

TOP_K_VALUES = [1, 3, 5, 10]

# thresholds para silver labels
HIGH_RELEVANCE_THRESHOLD = 0.75
PARTIAL_RELEVANCE_THRESHOLD = 0.45

# peso dos sinais heurísticos
WEIGHTS = {
    "entity_in_title": 0.35,
    "entity_in_text": 0.25,
    "title_overlap": 0.15,
    "token_overlap": 0.10,
    "score_signal": 0.15,
}

# opcional: LLM judge
USE_LLM_JUDGE = False
LLM_MODEL = "gpt-4.1-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =====================================
# TEXT UTILS
# =====================================

STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "no", "na",
    "nos", "nas", "um", "uma", "uns", "umas", "por", "para", "com", "sem", "sobre",
    "que", "ser", "é", "foi", "são", "ao", "à", "às", "ou", "como", "mais", "menos",
    "muito", "muita", "muitos", "muitas", "já", "não", "sim", "se", "sua", "seu",
    "suas", "seus", "ele", "ela", "eles", "elas", "pessoas", "trans"
}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    text = normalize_text(text)
    return [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]


def unique_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =====================================
# ENTITY EXTRACTION (leve, sem spaCy)
# =====================================

def extract_candidate_entities(video_title: str):
    """
    Heurística leve:
    - captura sequências com maiúsculas / nomes próprios
    - preserva coisas como 'J. K. Rowling'
    - remove fragmentos muito curtos
    """
    if not isinstance(video_title, str):
        return []

    raw = video_title.strip()

    patterns = [
        r"(?:[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][\w\.\-']+(?:\s+|$)){1,5}",
    ]

    candidates = []
    for pattern in patterns:
        for m in re.finditer(pattern, raw):
            ent = m.group(0).strip(" ,:;.!?\"'()[]{}")
            if len(ent) >= 3:
                candidates.append(ent)

    # remove sequências muito genéricas
    bad_starts = {"Pré", "Quero", "Como", "Porque", "Quando", "Onde", "Quem"}
    filtered = []
    for ent in candidates:
        first = ent.split()[0]
        if first in bad_starts:
            continue
        if len(ent.split()) <= 5:
            filtered.append(ent)

    filtered = unique_preserve_order(filtered)

    # fallback: nomes curtos relevantes citados na frase
    # ex: Thammy, Jotta A, Rowling
    extra = []
    tokens = video_title.split()
    for i, tok in enumerate(tokens):
        clean = tok.strip(" ,:;.!?\"'()[]{}")
        if clean[:1].isupper() and len(clean) > 2:
            extra.append(clean)

    filtered.extend(extra)
    filtered = unique_preserve_order(filtered)

    return filtered[:5]


# =====================================
# LOAD DATA
# =====================================

def load_titles(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"video_id", "video_title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV de títulos sem colunas obrigatórias: {missing}")
    return df


def load_metadata_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["video_id", "title", "chunk_index", "text", "score", "source"])
    df = pd.DataFrame(rows)

    if "source" not in df.columns:
        df["source"] = "unknown"

    return df


# =====================================
# DOC AGGREGATION
# =====================================

def aggregate_to_doc_level(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte chunk-level para doc-level.
    Mantém score máximo do documento e concatena um preview.
    """
    if metadata_df.empty:
        return pd.DataFrame(columns=["video_id", "doc_title", "source", "doc_score", "doc_text"])

    grouped = (
        metadata_df
        .groupby(["video_id", "title", "source"], as_index=False)
        .agg({
            "score": "max",
            "text": lambda xs: " ".join(list(xs)[:3])  # preview curto
        })
        .rename(columns={
            "title": "doc_title",
            "score": "doc_score",
            "text": "doc_text"
        })
    )

    return grouped


# =====================================
# HEURISTIC RELEVANCE / SILVER LABELS
# =====================================

def jaccard(a, b):
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def overlap_ratio(a, b):
    if not a:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa), 1)


def compute_silver_features(video_title: str, doc_title: str, doc_text: str, doc_score: float, entities):
    vt_norm = normalize_text(video_title)
    dt_norm = normalize_text(doc_title)
    dx_norm = normalize_text(doc_text[:1200])

    vt_tokens = tokenize(video_title)
    dt_tokens = tokenize(doc_title)
    dx_tokens = tokenize(doc_text[:1200])

    entity_in_title = 0.0
    entity_in_text = 0.0

    for ent in entities:
        ent_norm = normalize_text(ent)
        if ent_norm and ent_norm in dt_norm:
            entity_in_title = 1.0
        if ent_norm and ent_norm in dx_norm:
            entity_in_text = 1.0

    title_overlap = jaccard(vt_tokens, dt_tokens)
    token_overlap = overlap_ratio(vt_tokens, dx_tokens)

    # normalização simples do score já vindo do retriever
    score_signal = max(0.0, min(float(doc_score), 1.0))

    return {
        "entity_in_title": entity_in_title,
        "entity_in_text": entity_in_text,
        "title_overlap": title_overlap,
        "token_overlap": token_overlap,
        "score_signal": score_signal,
    }


def combine_features(features):
    score = 0.0
    for k, w in WEIGHTS.items():
        score += w * features.get(k, 0.0)
    return max(0.0, min(score, 1.0))


def silver_label_from_score(score: float) -> int:
    if score >= HIGH_RELEVANCE_THRESHOLD:
        return 2
    if score >= PARTIAL_RELEVANCE_THRESHOLD:
        return 1
    return 0


# =====================================
# OPTIONAL LLM JUDGE
# =====================================

def llm_judge_relevance(video_title: str, doc_title: str, doc_text: str):
    """
    Retorna 0, 1 ou 2.
    Requer OPENAI_API_KEY e pacote openai instalado.
    """
    if not USE_LLM_JUDGE:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""
Você está avaliando relevância documental para RAG.

Título do vídeo:
{video_title}

Título do documento:
{doc_title}

Trecho do documento:
{doc_text[:1500]}

Critério:
2 = altamente relevante: ajuda diretamente a entender a pessoa, evento, controvérsia ou contexto central do vídeo
1 = parcialmente relevante: fornece contexto útil, mas não é central
0 = irrelevante: pouco ou nada ajuda a entender o vídeo

Responda APENAS com um JSON válido no formato:
{{"relevance": 0}}
ou
{{"relevance": 1}}
ou
{{"relevance": 2}}
""".strip()

        resp = client.responses.create(
            model=LLM_MODEL,
            input=prompt,
        )

        text = resp.output_text.strip()
        data = json.loads(text)
        rel = int(data["relevance"])
        if rel not in {0, 1, 2}:
            return None
        return rel

    except Exception as e:
        print(f"[LLM JUDGE ERROR] {e}")
        return None


# =====================================
# METRICS
# =====================================

def dcg(rels):
    total = 0.0
    for i, rel in enumerate(rels, start=1):
        total += (2**rel - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(rels, k):
    rels_k = rels[:k]
    ideal = sorted(rels, reverse=True)[:k]
    denom = dcg(ideal)
    if denom == 0:
        return 0.0
    return dcg(rels_k) / denom


def precision_at_k(rels, k):
    rels_k = rels[:k]
    if not rels_k:
        return 0.0
    return sum(1 for r in rels_k if r > 0) / len(rels_k)


def recall_at_k(rels, k):
    total_relevant = sum(1 for r in rels if r > 0)
    if total_relevant == 0:
        return 0.0
    found = sum(1 for r in rels[:k] if r > 0)
    return found / total_relevant


def mrr(rels):
    for i, rel in enumerate(rels, start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


# =====================================
# MAIN EVALUATION
# =====================================

def evaluate():
    titles_df = load_titles(TITLES_PATH)
    metadata_df = load_metadata_jsonl(METADATA_PATH)
    docs_df = aggregate_to_doc_level(metadata_df)

    if docs_df.empty:
        print("Nenhum documento encontrado em metadata.jsonl")
        return

    merged = docs_df.merge(
        titles_df[["video_id", "video_title"]],
        on="video_id",
        how="left"
    )

    doc_level_rows = []
    video_level_rows = []

    for video_id, group in merged.groupby("video_id"):
        group = group.copy()

        video_title = group["video_title"].iloc[0]
        entities = extract_candidate_entities(video_title)

        rows = []
        for _, row in group.iterrows():
            features = compute_silver_features(
                video_title=video_title,
                doc_title=row["doc_title"],
                doc_text=row["doc_text"],
                doc_score=row["doc_score"],
                entities=entities
            )

            silver_score = combine_features(features)
            silver_label = silver_label_from_score(silver_score)

            llm_label = None
            final_label = silver_label

            if USE_LLM_JUDGE:
                llm_label = llm_judge_relevance(
                    video_title=video_title,
                    doc_title=row["doc_title"],
                    doc_text=row["doc_text"]
                )
                if llm_label is not None:
                    final_label = llm_label
                    time.sleep(0.2)

            rows.append({
                "video_id": video_id,
                "video_title": video_title,
                "doc_title": row["doc_title"],
                "source": row["source"],
                "doc_score": row["doc_score"],
                "entities": " | ".join(entities),
                **features,
                "silver_score": silver_score,
                "silver_label": silver_label,
                "llm_label": llm_label,
                "final_label": final_label,
            })

        ranked = pd.DataFrame(rows).sort_values(
            by=["doc_score", "silver_score"],
            ascending=False
        ).reset_index(drop=True)

        rels = ranked["final_label"].tolist()

        video_metrics = {
            "video_id": video_id,
            "video_title": video_title,
            "num_docs": len(ranked),
            "num_relevant_docs": int(sum(1 for r in rels if r > 0)),
            "num_highly_relevant_docs": int(sum(1 for r in rels if r == 2)),
            "mrr": mrr(rels),
        }

        for k in TOP_K_VALUES:
            video_metrics[f"precision@{k}"] = precision_at_k(rels, k)
            video_metrics[f"recall@{k}"] = recall_at_k(rels, k)
            video_metrics[f"ndcg@{k}"] = ndcg_at_k(rels, k)

        video_level_rows.append(video_metrics)
        doc_level_rows.extend(ranked.to_dict(orient="records"))

    doc_eval_df = pd.DataFrame(doc_level_rows)
    video_eval_df = pd.DataFrame(video_level_rows)

    doc_eval_df.to_csv(OUTPUT_DOC_LEVEL_PATH, index=False, encoding="utf8")
    video_eval_df.to_csv(OUTPUT_VIDEO_LEVEL_PATH, index=False, encoding="utf8")

    print("\n===== RESULTADOS GERAIS =====")
    print(f"Vídeos avaliados: {len(video_eval_df)}")
    print(f"Docs avaliados: {len(doc_eval_df)}")

    metric_cols = ["mrr"] + [f"{m}@{k}" for k in TOP_K_VALUES for m in ["precision", "recall", "ndcg"]]
    summary = video_eval_df[metric_cols].mean(numeric_only=True)

    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")

    coverage = (video_eval_df["num_highly_relevant_docs"] > 0).mean()
    useful_coverage = (video_eval_df["num_relevant_docs"] > 0).mean()

    print(f"Coverage (>=1 doc altamente relevante): {coverage:.4f}")
    print(f"Useful coverage (>=1 doc relevante): {useful_coverage:.4f}")
    print(f"\nArquivo doc-level salvo em: {OUTPUT_DOC_LEVEL_PATH}")
    print(f"Arquivo video-level salvo em: {OUTPUT_VIDEO_LEVEL_PATH}")


if __name__ == "__main__":
    evaluate()