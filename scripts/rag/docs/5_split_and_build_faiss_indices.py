import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIG
# =========================================================

INPUT_PATH = Path("./data/wiki/metadata.jsonl")
OUTPUT_DIR = Path("./data/retrieval_split")

EMBEDDING_MODEL = "BAAI/bge-m3"
DEVICE = "cpu"  # troque para "cuda" se quiser usar GPU

ENTITIES_METADATA_PATH = OUTPUT_DIR / "metadata_entities.jsonl"
EXPLANATIONS_METADATA_PATH = OUTPUT_DIR / "metadata_explanations.jsonl"

ENTITIES_INDEX_PATH = OUTPUT_DIR / "faiss_entities.bin"
EXPLANATIONS_INDEX_PATH = OUTPUT_DIR / "faiss_explanations.bin"

REPORT_PATH = OUTPUT_DIR / "split_and_prune_report.json"

MAX_CHUNKS_ENTITY = 2
MAX_CHUNKS_EXPLANATORY = 3
MAX_CHUNKS_NORMATIVE = 4
MAX_CHUNKS_DEFAULT = 3

DIVERSITY_MAX_JACCARD = 0.75

TAIL_SECTION_TERMS = {
    "referências", "referencias", "bibliografia", "ligações externas", "ligacoes externas",
    "ver também", "ver tambem", "galeria", "notas", "fontes", "leia também", "leia tambem",
    "compartilhe isso", "curtir carregando", "os comentários estão desativados", "pingback"
}

TASK_TERMS = {
    "trans", "travesti", "travestis", "transexual", "transexuais", "transgenero", "transgênero",
    "transfobia", "identidade", "gênero", "genero", "discriminação", "discriminacao",
    "violência", "violencia", "ódio", "odio", "direitos", "ameaça", "ameaca",
    "assassinato", "assassinatos", "política", "politica", "saúde", "saude",
    "nome", "social", "banheiro", "emprego", "hiv", "escola", "lei", "jurídico", "juridico",
    "feminicídio", "feminicidio", "misoginia", "racismo", "religião", "religiao",
    "hormonização", "hormonizacao", "cidh", "onu", "stf", "dossiê", "dossie"
}

LIGHT_STOPWORDS = {
    "de", "da", "do", "das", "dos", "e", "em", "para", "com", "sem", "na", "no", "nas", "nos",
    "um", "uma", "uns", "umas", "que", "como", "por", "ao", "aos", "às", "ate", "até",
    "sobre", "entre", "contra", "se", "é", "o", "a", "os", "as"
}

TRASH_TITLE_PATTERNS = [
    r"^\d+\s+coment[aá]rio[s]?$",
    r"^coment[aá]rio[s]?$",
    r"^pingback",
    r"^trackback",
    r"^sem t[ií]tulo$",
]

# =========================================================
# UTILS
# =========================================================

def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", str(text).lower())


def normalized_tokens(text: str) -> List[str]:
    toks = [normalize_text(t) for t in tokenize(text)]
    return [t for t in toks if t and t not in LIGHT_STOPWORDS and len(t) >= 3]


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def is_tail_chunk(text: str) -> bool:
    norm = normalize_text(text)
    return any(term in norm for term in TAIL_SECTION_TERMS)


def is_trash_title(title: str) -> bool:
    norm = normalize_text(title)
    if not norm:
        return True
    return any(re.match(pattern, norm) for pattern in TRASH_TITLE_PATTERNS)


def is_entity_title(title: str) -> bool:
    raw = str(title).strip()
    norm = normalize_text(raw)

    if any(term in norm for term in [
        "direitos", "violencia", "violência", "feminismo", "psicologia", "dia internacional",
        "politica", "política", "saude", "saúde", "conferencia", "conferência",
        "nota técnica", "nota tecnica", "relatorio", "relatório", "dossie", "dossiê",
        "voto", "monitoramento", "programa", "implementacao", "implementação", "carta", "denúncia", "denuncia"
    ]):
        return False

    words = raw.split()
    if len(words) > 5:
        return False

    capitalized = sum(1 for w in words if w[:1].isupper())
    return capitalized >= max(1, math.ceil(len(words) * 0.6))


def is_normative_or_policy_title(title: str) -> bool:
    norm = normalize_text(title)
    return any(term in norm for term in [
        "direitos", "lei", "portaria", "politica", "política", "nota técnica", "nota tecnica",
        "conferencia", "conferência", "saude", "saúde", "voto", "onu", "stf", "cidh",
        "relatora", "programa", "implementacao", "implementação", "documento", "monitoramento",
        "dossiê", "dossie", "denúncia", "denuncia", "carta", "proposta", "memória", "memoria"
    ])


def classify_title_type(title: str, source: str) -> str:
    if is_entity_title(title):
        return "entity"
    if is_normative_or_policy_title(title):
        return "normative"
    return "explanatory"


def max_chunks_for_title(title: str, source: str) -> int:
    kind = classify_title_type(title, source)
    if kind == "entity":
        return MAX_CHUNKS_ENTITY
    if kind == "normative":
        return MAX_CHUNKS_NORMATIVE
    if kind == "explanatory":
        return MAX_CHUNKS_EXPLANATORY
    return MAX_CHUNKS_DEFAULT


# =========================================================
# SCORING
# =========================================================

def compute_title_keywords(rows: List[Dict[str, Any]]) -> List[str]:
    title = rows[0].get("title", "")
    toks = normalized_tokens(title)
    out = []
    seen = set()
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out[:10]


def compute_document_keywords(rows: List[Dict[str, Any]], title_terms: List[str]) -> List[str]:
    counter = Counter()
    for row in rows:
        toks = normalized_tokens(row.get("text", ""))
        counter.update(toks)

    ranked = [tok for tok, _ in counter.most_common(30)]
    merged = []

    for t in title_terms:
        if t not in merged:
            merged.append(t)

    for t in ranked:
        if t in TASK_TERMS and t not in merged:
            merged.append(t)

    for t in ranked:
        if t not in merged:
            merged.append(t)

    return merged[:20]


def centrality_score(idx: int, total: int, text: str) -> float:
    if total <= 1:
        base = 1.0
    else:
        rel = idx / max(total - 1, 1)
        base = 1.0 - 0.85 * rel

    if is_tail_chunk(text):
        base -= 0.8

    return max(base, 0.0)


def length_score(text: str) -> float:
    n = len(normalized_tokens(text))
    if n < 40:
        return 0.1
    if n < 80:
        return 0.5
    if n <= 220:
        return 1.0
    if n <= 320:
        return 0.75
    return 0.4


def overlap_ratio(text: str, terms: List[str]) -> float:
    if not terms:
        return 0.0
    toks = set(normalized_tokens(text))
    hits = sum(1 for t in terms if t in toks)
    return hits / len(terms)


def task_relevance_score(text: str) -> float:
    toks = set(normalized_tokens(text))
    hits = sum(1 for t in TASK_TERMS if t in toks)
    if hits == 0:
        return 0.0
    return min(hits / 6.0, 1.0)


def definition_bonus(title: str, text: str) -> float:
    norm_title = normalize_text(title)
    norm_text = normalize_text(text[:600])

    bonus = 0.0
    if norm_title and norm_title.split(".")[0] in norm_text:
        bonus += 0.2

    if any(p in norm_text for p in [" é ", " sao ", " são ", " refere-se", " consiste em", " diz respeito"]):
        bonus += 0.2

    return min(bonus, 0.4)


def score_chunk(row: Dict[str, Any], idx: int, total: int, title_terms: List[str], doc_terms: List[str]) -> float:
    text = row.get("text", "")
    title = row.get("title", "")

    c_score = centrality_score(idx, total, text)
    d_score = overlap_ratio(text, doc_terms)
    t_score = task_relevance_score(text)
    l_score = length_score(text)
    b_score = definition_bonus(title, text)

    final = (
        0.35 * c_score +
        0.30 * d_score +
        0.25 * t_score +
        0.10 * l_score +
        b_score
    )

    if is_tail_chunk(text):
        final -= 0.5

    return final


# =========================================================
# PRUNING / SPLIT
# =========================================================

def diversify_select(scored_rows: List[Tuple[float, Dict[str, Any]]], limit: int) -> List[Dict[str, Any]]:
    selected = []
    selected_sets = []

    for score, row in scored_rows:
        text_set = set(normalized_tokens(row.get("text", "")))
        too_close = False

        for prev in selected_sets:
            if jaccard(text_set, prev) > DIVERSITY_MAX_JACCARD:
                too_close = True
                break

        if too_close:
            continue

        selected.append(row)
        selected_sets.append(text_set)

        if len(selected) >= limit:
            break

    return selected


def prune_and_split_metadata(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    grouped = defaultdict(list)
    trash_rows = 0

    for row in rows:
        title = row.get("title", "")
        if is_trash_title(title):
            trash_rows += 1
            continue

        key = (
            normalize_text(row.get("source", "")),
            normalize_text(title),
        )
        grouped[key].append(row)

    entities = []
    explanations = []

    report = {
        "input_rows": len(rows),
        "trash_rows_removed": trash_rows,
        "titles_total": len(grouped),
        "entities_rows": 0,
        "explanations_rows": 0,
        "titles": [],
    }

    for (_, _), group in grouped.items():
        group = sorted(group, key=lambda x: int(x.get("chunk_index", 0)))
        title = group[0].get("title", "")
        source = group[0].get("source", "")
        total = len(group)

        title_type = classify_title_type(title, source)
        limit = max_chunks_for_title(title, source)

        title_terms = compute_title_keywords(group)
        doc_terms = compute_document_keywords(group, title_terms)

        scored = []
        for i, row in enumerate(group):
            s = score_chunk(row, i, total, title_terms, doc_terms)
            new_row = dict(row)
            new_row["_prune_score"] = round(float(s), 6)
            new_row["_title_type"] = title_type
            scored.append((s, new_row))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = diversify_select(scored, limit)
        selected = sorted(selected, key=lambda x: int(x.get("chunk_index", 0)))

        if title_type == "entity":
            entities.extend(selected)
        else:
            explanations.extend(selected)

        report["titles"].append({
            "title": title,
            "source": source,
            "title_type": title_type,
            "input_chunks": total,
            "kept_chunks": len(selected),
            "kept_chunk_indexes": [int(x.get("chunk_index", 0)) for x in selected],
            "title_terms": title_terms,
            "doc_terms_preview": doc_terms[:10],
        })

    report["entities_rows"] = len(entities)
    report["explanations_rows"] = len(explanations)
    report["output_rows_total"] = len(entities) + len(explanations)
    report["removed_rows_total"] = len(rows) - report["output_rows_total"]
    return entities, explanations, report


# =========================================================
# IO
# =========================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Linha inválida {line_no}: {e}")
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


# =========================================================
# FAISS BUILD
# =========================================================

def build_faiss_index(rows: List[Dict[str, Any]], index_path: Path, model: SentenceTransformer) -> None:
    if not rows:
        print(f"[WARN] Nenhuma linha para indexar em {index_path}")
        return

    texts = [str(r.get("text", "")).strip() for r in rows]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )

    emb = np.asarray(embeddings, dtype="float32")
    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    print(f"[OK] Índice salvo em {index_path} com {index.ntotal} vetores.")


# =========================================================
# MAIN
# =========================================================

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Lendo metadata...")
    rows = load_jsonl(INPUT_PATH)

    print("[2/4] Fazendo pruning e separação...")
    entities, explanations, report = prune_and_split_metadata(rows)

    print("[3/4] Salvando metadata separado...")
    write_jsonl(entities, ENTITIES_METADATA_PATH)
    write_jsonl(explanations, EXPLANATIONS_METADATA_PATH)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[4/4] Construindo índices FAISS...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    build_faiss_index(entities, ENTITIES_INDEX_PATH, model)
    build_faiss_index(explanations, EXPLANATIONS_INDEX_PATH, model)

    print("\nConcluído.")
    print(f"Entities metadata:      {ENTITIES_METADATA_PATH}")
    print(f"Explanations metadata:  {EXPLANATIONS_METADATA_PATH}")
    print(f"Entities index:         {ENTITIES_INDEX_PATH}")
    print(f"Explanations index:     {EXPLANATIONS_INDEX_PATH}")
    print(f"Report:                 {REPORT_PATH}")
    print(f"Entities rows:          {len(entities)}")
    print(f"Explanations rows:      {len(explanations)}")


if __name__ == "__main__":
    main()
