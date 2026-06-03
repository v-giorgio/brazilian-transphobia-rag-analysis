import os
import re
import json
import math
import time
import unicodedata

import pandas as pd

# =====================================
# CONFIG
# =====================================

TITLES_PATH = "./data/titles/video_titles.csv"
METADATA_PATH = "./data/wiki/metadata.jsonl"

OUTPUT_DOC_LEVEL_PATH = "./data/wiki/eval_doc_level.csv"
OUTPUT_VIDEO_LEVEL_PATH = "./data/wiki/eval_video_level.csv"
OUTPUT_SOURCE_LEVEL_PATH = "./data/wiki/eval_source_level.csv"
OUTPUT_VIDEO_SOURCE_LEVEL_PATH = "./data/wiki/eval_video_source_level.csv"

TOP_K_VALUES = [1, 3, 5, 10]

HIGH_RELEVANCE_THRESHOLD = 0.72
PARTIAL_RELEVANCE_THRESHOLD = 0.42

# pesos mais equilibrados para comparar wikipedia/antra/gov
WEIGHTS = {
    "entity_in_title": 0.22,
    "entity_in_text": 0.18,
    "title_overlap": 0.10,
    "token_overlap": 0.08,
    "score_signal": 0.12,
    "theme_overlap": 0.18,
    "source_bonus": 0.12,
}

USE_LLM_JUDGE = False
LLM_MODEL = "gpt-4.1-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "no", "na",
    "nos", "nas", "um", "uma", "uns", "umas", "por", "para", "com", "sem", "sobre",
    "que", "ser", "é", "foi", "são", "ao", "à", "às", "ou", "como", "mais", "menos",
    "muito", "muita", "muitos", "muitas", "já", "não", "sim", "se", "sua", "seu",
    "suas", "seus", "ele", "ela", "eles", "elas", "pessoas", "trans"
}

THEME_KEYWORDS = {
    "violence": {
        "violencia", "violência", "transfobia", "agressao", "agressão", "assassinato",
        "crime", "odio", "ódio", "discriminacao", "discriminação", "ataque"
    },
    "rights_policy": {
        "direito", "direitos", "lei", "decreto", "portaria", "resolucao", "resolução",
        "politica", "política", "politicas", "pública", "publica", "nome", "social",
        "retificacao", "retificação", "registro", "cidadania"
    },
    "health": {
        "saude", "saúde", "sus", "ambulatorio", "ambulatório", "psicologo", "psicólogo",
        "psiquiatra", "atendimento", "hormonio", "hormônio", "hormonioterapia",
        "cirurgia", "processo", "transexualizador"
    },
    "education_work": {
        "educacao", "educação", "escola", "universidade", "trabalho", "emprego",
        "empregabilidade", "mercado", "renda", "profissional"
    },
    "biographical": {
        "quem", "biografia", "vida", "carreira", "cantor", "atriz", "ator", "modelo",
        "deputado", "deputada", "vereador", "vereadora"
    },
}

SOURCE_THEME_AFFINITY = {
    "wikipedia": {"violence": 0.45, "rights_policy": 0.45, "health": 0.40, "education_work": 0.40, "biographical": 1.0},
    "antra": {"violence": 1.0, "rights_policy": 0.80, "health": 0.70, "education_work": 0.75, "biographical": 0.20},
    "gov": {"violence": 0.80, "rights_policy": 1.0, "health": 1.0, "education_work": 0.85, "biographical": 0.15},
    "gov_br": {"violence": 0.80, "rights_policy": 1.0, "health": 1.0, "education_work": 0.85, "biographical": 0.15},
    "planalto": {"violence": 0.70, "rights_policy": 1.0, "health": 0.70, "education_work": 0.65, "biographical": 0.05},
    "mdhc": {"violence": 0.85, "rights_policy": 0.95, "health": 0.70, "education_work": 0.70, "biographical": 0.10},
    "unknown": {"violence": 0.40, "rights_policy": 0.40, "health": 0.40, "education_work": 0.40, "biographical": 0.40},
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


def extract_candidate_entities(video_title: str):
    if not isinstance(video_title, str):
        return []

    raw = video_title.strip()
    patterns = [r"(?:[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][\w\.\-']+(?:\s+|$)){1,5}"]

    candidates = []
    for pattern in patterns:
        for m in re.finditer(pattern, raw):
            ent = m.group(0).strip(" ,:;.!?\"'()[]{}")
            if len(ent) >= 3:
                candidates.append(ent)

    bad_starts = {"Pré", "Quero", "Como", "Porque", "Quando", "Onde", "Quem"}
    filtered = []
    for ent in candidates:
        first = ent.split()[0]
        if first in bad_starts:
            continue
        if len(ent.split()) <= 5:
            filtered.append(ent)

    filtered = unique_preserve_order(filtered)

    extra = []
    for tok in video_title.split():
        clean = tok.strip(" ,:;.!?\"'()[]{}")
        if clean[:1].isupper() and len(clean) > 2:
            extra.append(clean)

    filtered.extend(extra)
    filtered = unique_preserve_order(filtered)
    return filtered[:5]


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


def aggregate_to_doc_level(metadata_df: pd.DataFrame) -> pd.DataFrame:
    if metadata_df.empty:
        return pd.DataFrame(columns=["video_id", "doc_title", "source", "doc_score", "doc_text"])

    grouped = (
        metadata_df
        .groupby(["video_id", "title", "source"], as_index=False)
        .agg({
            "score": "max",
            "text": lambda xs: " ".join(list(xs)[:3])
        })
        .rename(columns={
            "title": "doc_title",
            "score": "doc_score",
            "text": "doc_text"
        })
    )
    return grouped


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


def detect_video_themes(video_title: str):
    norm = normalize_text(video_title)
    themes = {}
    for theme, words in THEME_KEYWORDS.items():
        count = sum(1 for w in words if w in norm)
        themes[theme] = min(count / 2.0, 1.0)
    if all(v == 0 for v in themes.values()):
        themes["biographical"] = 0.35
    return themes


def source_bonus_for_video(source: str, video_themes: dict):
    src = normalize_text(source) or "unknown"
    affinity = SOURCE_THEME_AFFINITY.get(src, SOURCE_THEME_AFFINITY["unknown"])
    best = 0.0
    for theme, strength in video_themes.items():
        best = max(best, strength * affinity.get(theme, 0.4))
    return max(0.0, min(best, 1.0))


def theme_overlap_score(video_title: str, doc_title: str, doc_text: str):
    video_themes = detect_video_themes(video_title)
    norm_doc = normalize_text(f"{doc_title} {doc_text[:1400]}")

    doc_theme_hits = {}
    for theme, words in THEME_KEYWORDS.items():
        count = sum(1 for w in words if w in norm_doc)
        doc_theme_hits[theme] = min(count / 3.0, 1.0)

    best = 0.0
    for theme, v_strength in video_themes.items():
        best = max(best, v_strength * doc_theme_hits.get(theme, 0.0))
    return max(0.0, min(best, 1.0)), video_themes, doc_theme_hits


def compute_silver_features(video_title: str, doc_title: str, doc_text: str, doc_score: float, entities, source: str):
    dt_norm = normalize_text(doc_title)
    dx_norm = normalize_text(doc_text[:1400])

    vt_tokens = tokenize(video_title)
    dt_tokens = tokenize(doc_title)
    dx_tokens = tokenize(doc_text[:1400])

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

    raw_score = float(doc_score)
    if raw_score > 1.0:
        score_signal = raw_score / (raw_score + 1.0)
    else:
        score_signal = max(0.0, min(raw_score, 1.0))

    theme_overlap, video_themes, doc_theme_hits = theme_overlap_score(video_title, doc_title, doc_text)
    source_bonus = source_bonus_for_video(source, video_themes)

    return {
        "entity_in_title": entity_in_title,
        "entity_in_text": entity_in_text,
        "title_overlap": title_overlap,
        "token_overlap": token_overlap,
        "score_signal": score_signal,
        "theme_overlap": theme_overlap,
        "source_bonus": source_bonus,
        "video_theme_main": max(video_themes, key=video_themes.get),
        "doc_theme_main": max(doc_theme_hits, key=doc_theme_hits.get),
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


def llm_judge_relevance(video_title: str, doc_title: str, doc_text: str):
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
        resp = client.responses.create(model=LLM_MODEL, input=prompt)
        text = resp.output_text.strip()
        data = json.loads(text)
        rel = int(data["relevance"])
        if rel not in {0, 1, 2}:
            return None
        return rel
    except Exception as e:
        print(f"[LLM JUDGE ERROR] {e}")
        return None


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


def build_metrics_row(ranked_df: pd.DataFrame, prefix: str = ""):
    rels = ranked_df["final_label"].tolist()
    row = {
        f"{prefix}num_docs": len(ranked_df),
        f"{prefix}num_relevant_docs": int(sum(1 for r in rels if r > 0)),
        f"{prefix}num_highly_relevant_docs": int(sum(1 for r in rels if r == 2)),
        f"{prefix}mrr": mrr(rels),
    }
    for k in TOP_K_VALUES:
        row[f"{prefix}precision@{k}"] = precision_at_k(rels, k)
        row[f"{prefix}recall@{k}"] = recall_at_k(rels, k)
        row[f"{prefix}ndcg@{k}"] = ndcg_at_k(rels, k)
    return row


def summarize_metrics(df: pd.DataFrame, label: str):
    if df.empty:
        print(f"\n===== {label} =====")
        print("Sem dados")
        return
    metric_cols = [c for c in df.columns if c in (["mrr"] + [f"{m}@{k}" for k in TOP_K_VALUES for m in ["precision", "recall", "ndcg"]])]
    print(f"\n===== {label} =====")
    print(f"Vídeos avaliados: {len(df)}")
    summary = df[metric_cols].mean(numeric_only=True)
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")
    coverage = (df["num_highly_relevant_docs"] > 0).mean()
    useful_coverage = (df["num_relevant_docs"] > 0).mean()
    print(f"Coverage (>=1 doc altamente relevante): {coverage:.4f}")
    print(f"Useful coverage (>=1 doc relevante): {useful_coverage:.4f}")


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
    video_source_level_rows = []

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
                entities=entities,
                source=row["source"],
            )

            silver_score = combine_features(features)
            silver_label = silver_label_from_score(silver_score)

            llm_label = None
            final_label = silver_label
            if USE_LLM_JUDGE:
                llm_label = llm_judge_relevance(video_title=video_title, doc_title=row["doc_title"], doc_text=row["doc_text"])
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

        ranked = pd.DataFrame(rows).sort_values(by=["doc_score", "silver_score"], ascending=False).reset_index(drop=True)
        video_metrics = {"video_id": video_id, "video_title": video_title, **build_metrics_row(ranked)}
        video_level_rows.append(video_metrics)
        doc_level_rows.extend(ranked.to_dict(orient="records"))

        for source_name, source_group in ranked.groupby("source"):
            source_ranked = source_group.sort_values(by=["doc_score", "silver_score"], ascending=False).reset_index(drop=True)
            src_metrics = {
                "video_id": video_id,
                "video_title": video_title,
                "source": source_name,
                **build_metrics_row(source_ranked)
            }
            video_source_level_rows.append(src_metrics)

    doc_eval_df = pd.DataFrame(doc_level_rows)
    video_eval_df = pd.DataFrame(video_level_rows)
    video_source_eval_df = pd.DataFrame(video_source_level_rows)

    source_level_rows = []
    for source_name, group in video_source_eval_df.groupby("source"):
        metric_cols = ["mrr"] + [f"{m}@{k}" for k in TOP_K_VALUES for m in ["precision", "recall", "ndcg"]]
        row = {
            "source": source_name,
            "videos_evaluated": len(group),
            "avg_num_docs": group["num_docs"].mean(),
            "coverage_highly_relevant": (group["num_highly_relevant_docs"] > 0).mean(),
            "coverage_relevant": (group["num_relevant_docs"] > 0).mean(),
        }
        for c in metric_cols:
            row[c] = group[c].mean()
        source_level_rows.append(row)
    source_eval_df = pd.DataFrame(source_level_rows).sort_values(by="coverage_relevant", ascending=False)

    doc_eval_df.to_csv(OUTPUT_DOC_LEVEL_PATH, index=False, encoding="utf8")
    video_eval_df.to_csv(OUTPUT_VIDEO_LEVEL_PATH, index=False, encoding="utf8")
    source_eval_df.to_csv(OUTPUT_SOURCE_LEVEL_PATH, index=False, encoding="utf8")
    video_source_eval_df.to_csv(OUTPUT_VIDEO_SOURCE_LEVEL_PATH, index=False, encoding="utf8")

    print("\n===== RESULTADOS GERAIS =====")
    print(f"Vídeos avaliados: {len(video_eval_df)}")
    print(f"Docs avaliados: {len(doc_eval_df)}")
    summarize_metrics(video_eval_df, "RESULTADOS GERAIS")

    if not source_eval_df.empty:
        print("\n===== RESULTADOS POR SOURCE =====")
        for _, row in source_eval_df.iterrows():
            print(f"\nSOURCE: {row['source']}")
            print(f"videos_evaluated: {int(row['videos_evaluated'])}")
            print(f"avg_num_docs: {row['avg_num_docs']:.4f}")
            print(f"mrr: {row['mrr']:.4f}")
            for k in TOP_K_VALUES:
                print(f"precision@{k}: {row[f'precision@{k}']:.4f}")
                print(f"recall@{k}: {row[f'recall@{k}']:.4f}")
                print(f"ndcg@{k}: {row[f'ndcg@{k}']:.4f}")
            print(f"Coverage (>=1 doc altamente relevante): {row['coverage_highly_relevant']:.4f}")
            print(f"Useful coverage (>=1 doc relevante): {row['coverage_relevant']:.4f}")

    print(f"\nArquivo doc-level salvo em: {OUTPUT_DOC_LEVEL_PATH}")
    print(f"Arquivo video-level salvo em: {OUTPUT_VIDEO_LEVEL_PATH}")
    print(f"Arquivo source-level salvo em: {OUTPUT_SOURCE_LEVEL_PATH}")
    print(f"Arquivo video-source-level salvo em: {OUTPUT_VIDEO_SOURCE_LEVEL_PATH}")


if __name__ == "__main__":
    evaluate()
