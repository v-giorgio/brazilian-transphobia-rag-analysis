import os
import re
import json
import time
import faiss
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# ==========================
# CONFIG
# ==========================

TITLES_PATH = "./data/titles/video_titles.csv"
OUTPUT_DIR = "./data/wiki"
METADATA_PATH = f"{OUTPUT_DIR}/metadata.jsonl"
INDEX_PATH = f"{OUTPUT_DIR}/faiss_index.bin"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32
REQUEST_DELAY = 0.8
REQUEST_TIMEOUT = 30
MAX_RESULTS_PER_QUERY = 4
MAX_DOCS_FROM_ANTRA = 20
MAX_ANTRA_DOCS_PER_VIDEO = 2
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40
SIMILARITY_THRESHOLD = 0.30
ENTITY_BOOST = 1.22
TITLE_BOOST = 1.10
THEME_BOOST = 1.35
QUERY_BOOST = 1.18

# bge-m3: embeddings normalizados + IP
NORMALIZE_EMBEDDINGS = True

ANTRA_BASE = "https://antrabrasil.org"
ANTRA_SEARCH_URL = "https://antrabrasil.org/?s={}"

FIXED_QUERIES = [
    "dossie antra assassinatos",
    "violencia contra pessoas trans antra",
    "transfobia antra",
    "nome social pessoas trans antra",
    "empregabilidade pessoas trans antra",
    "educacao pessoas trans antra",
    "saude pessoas trans antra",
]

VIDEO_THEME_KEYWORDS = {
    "violence": [
        "violencia", "violência", "transfobia", "assassinato", "morte", "agressao", "agressão",
        "crime", "odio", "ódio", "ataque", "discriminacao", "discriminação", "ambulatório", "ambulatorio"
    ],
    "rights": [
        "nome social", "retificacao", "retificação", "direito", "direitos", "politica publica",
        "política pública", "stf", "sus", "ambulatório", "ambulatorio", "escola", "educacao", "educação",
        "saude", "saúde", "psicologo", "psiquiatra", "atendimento"
    ],
    "work": ["trabalho", "emprego", "empregabilidade", "mercado de trabalho", "renda"],
}

DOC_THEME_KEYWORDS = {
    "violence": [
        "violencia", "violência", "transfobia", "assassinato", "morte", "crime", "ódio", "agressão", "agressao", "violacao", "violação"
    ],
    "rights": [
        "nome social", "direitos", "saúde", "saude", "educação", "educacao", "atendimento",
        "ambulatório", "ambulatorio", "política pública", "politica publica", "sus", "retificação", "retificacao"
    ],
    "work": ["trabalho", "emprego", "empregabilidade", "mercado de trabalho", "renda", "inclusão produtiva", "inclusao produtiva"],
}

STRONG_VIDEO_TERMS = [
    "ambulatório", "ambulatorio", "nome social", "transfobia", "assassinato", "violência", "violencia",
    "discriminação", "discriminacao", "empregabilidade", "mercado de trabalho", "atendimento", "sus",
    "saúde", "saude", "educação", "educacao", "retificação", "retificacao"
]

WEAK_OR_BROAD_VIDEO_TERMS = [
    "vereador", "deputado", "prefeito", "senador", "pastor", "cantor", "atriz", "ator", "jornalista",
    "polêmica", "polemica", "disse", "fala", "critica", "criticou", "debate"
]

GENERIC_INSTITUTIONAL_TERMS = [
    "antra", "associação nacional", "associacao nacional", "quem somos", "sobre nós", "sobre nos",
    "manifesto", "nota pública", "nota publica", "institucional", "campanha"
]

TRANS_TERMS = [
    "trans", "travesti", "transexual", "identidade de genero", "identidade de gênero",
    "nome social", "transfobia", "nao binario", "não binário", "lgbt"
]

EXCLUDED_URL_PARTS = [
    "/category/", "/tag/", "/author/", "/feed/", "/page/", "wp-json", "format=amp"
]

session = requests.Session()
session.headers.update({"User-Agent": "TransContextCollector/1.0"})
embedder = SentenceTransformer(MODEL_NAME)

# ==========================
# UTILS
# ==========================

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[\u2018\u2019\u201c\u201d\"']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str):
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def encode_texts(texts):
    return embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        show_progress_bar=False,
    )


def semantic_similarity(vec_a, vec_b):
    if NORMALIZE_EMBEDDINGS:
        return float(np.dot(vec_a, vec_b))
    return float(np.dot(vec_a, vec_b) / ((np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-9))


def sleep_request():
    time.sleep(REQUEST_DELAY)


def safe_get(url, **kwargs):
    for attempt in range(5):
        try:
            sleep_request()
            r = session.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"[429] {url} wait={wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1)
    raise RuntimeError(f"failed request: {url}")


def extract_entities_from_title(title: str):
    ents = []
    pattern = r"(?:[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][\w\.\-']+(?:\s+|$)){1,5}"
    for m in re.finditer(pattern, title or ""):
        ent = m.group(0).strip(" ,:;.!?\"'()[]{}")
        if len(ent) >= 3:
            ents.append(ent)
    seen = set()
    out = []
    for ent in ents:
        key = normalize_text(ent)
        if key and key not in seen:
            seen.add(key)
            out.append(ent)
    return out[:5]


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_text(content: bytes) -> str:
    if PdfReader is None:
        return ""
    tmp_path = "/tmp/antra_temp.pdf"
    with open(tmp_path, "wb") as f:
        f.write(content)
    reader = PdfReader(tmp_path)
    texts = []
    for page in reader.pages[:40]:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            pass
    return re.sub(r"\s+", " ", " ".join(texts)).strip()


def chunk_text(text: str, max_words: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + max_words]).strip()
        if chunk:
            chunks.append(chunk)
        if start + max_words >= len(words):
            break
    return chunks


def is_trans_relevant(text: str) -> bool:
    norm = normalize_text(text)
    return any(term in norm for term in TRANS_TERMS)


def keyword_hits(text: str, keywords):
    norm = normalize_text(text)
    return sum(1 for kw in keywords if normalize_text(kw) in norm)


def detect_video_themes(video_title: str):
    detected = set()
    norm = normalize_text(video_title)
    for theme, kws in VIDEO_THEME_KEYWORDS.items():
        if any(normalize_text(kw) in norm for kw in kws):
            detected.add(theme)
    return detected


def detect_doc_themes(title: str, text: str):
    detected = set()
    norm = normalize_text(f"{title}. {text[:4000]}")
    for theme, kws in DOC_THEME_KEYWORDS.items():
        if any(normalize_text(kw) in norm for kw in kws):
            detected.add(theme)
    return detected


def theme_scores(text: str, theme_map):
    norm = normalize_text(text)
    return {
        theme: sum(1 for kw in kws if normalize_text(kw) in norm)
        for theme, kws in theme_map.items()
    }


def dominant_theme_from_scores(scores):
    if not scores:
        return None
    theme, value = max(scores.items(), key=lambda x: x[1])
    return theme if value > 0 else None


def should_query_antra(video_title: str):
    norm = normalize_text(video_title)
    strong_hits = sum(1 for term in STRONG_VIDEO_TERMS if normalize_text(term) in norm)
    thematic_scores = theme_scores(video_title, VIDEO_THEME_KEYWORDS)
    total_theme_hits = sum(thematic_scores.values())
    broad_hits = sum(1 for term in WEAK_OR_BROAD_VIDEO_TERMS if normalize_text(term) in norm)

    if strong_hits >= 1:
        return True
    if total_theme_hits >= 2 and broad_hits == 0:
        return True
    return False


def doc_specificity_score(title: str, text: str):
    norm = normalize_text(f"{title}. {text[:2500]}")
    thematic_hits = sum(theme_scores(norm, DOC_THEME_KEYWORDS).values())
    trans_hits = sum(1 for term in TRANS_TERMS if normalize_text(term) in norm)
    generic_hits = sum(1 for term in GENERIC_INSTITUTIONAL_TERMS if normalize_text(term) in norm)
    return thematic_hits + 0.5 * trans_hits - generic_hits


def doc_is_overly_generic(title: str, text: str) -> bool:
    norm_title = normalize_text(title)
    norm_head = normalize_text(text[:2000])
    generic_count = sum(1 for s in GENERIC_INSTITUTIONAL_TERMS if s in norm_title or s in norm_head)
    thematic_count = keyword_hits(f"{title}. {text[:4000]}", sum(DOC_THEME_KEYWORDS.values(), []))
    specificity = doc_specificity_score(title, text)
    return generic_count >= 2 and (thematic_count == 0 or specificity < 1.5)


# ==========================
# ANTRA COLLECTION
# ==========================

def search_antra(query: str):
    url = ANTRA_SEARCH_URL.format(requests.utils.quote(query))
    try:
        r = safe_get(url)
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    seen = set()

    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        text = a.get_text(" ", strip=True)
        if not href:
            continue
        if href.startswith("/"):
            href = ANTRA_BASE + href
        if not href.startswith(ANTRA_BASE):
            continue
        if any(part in href for part in EXCLUDED_URL_PARTS):
            continue
        looks_like_doc = href.endswith(".pdf") or "/20" in href or "/wp-content/uploads/" in href
        if not looks_like_doc:
            continue
        title = text or href.split("/")[-2]
        key = href
        if key not in seen:
            seen.add(key)
            results.append({"title": title[:250], "url": href, "seed_query": query})
        if len(results) >= MAX_RESULTS_PER_QUERY:
            break

    return results


def fetch_antra_doc(candidate):
    url = candidate["url"]
    title = candidate["title"]
    try:
        r = safe_get(url)
    except Exception:
        return None

    ctype = r.headers.get("Content-Type", "").lower()
    text = ""

    if "pdf" in ctype or url.lower().endswith(".pdf"):
        text = extract_pdf_text(r.content)
        if not title or title.startswith("http"):
            title = url.split("/")[-1]
    else:
        soup = BeautifulSoup(r.text, "html.parser")
        if soup.title and (not title or len(title) < 5):
            title = soup.title.get_text(" ", strip=True)[:250]
        text = html_to_text(r.text)

    text = text.strip()
    if len(text) < 500:
        return None
    if not is_trans_relevant(f"{title} {text[:4000]}"):
        return None
    if doc_is_overly_generic(title, text):
        return None

    doc_themes = detect_doc_themes(title, text)
    if not doc_themes:
        return None

    scores = theme_scores(f"{title}. {text[:4000]}", DOC_THEME_KEYWORDS)
    specificity = doc_specificity_score(title, text)
    if specificity < 1.5:
        return None

    return {
        "title": title[:250],
        "url": url,
        "source": "antra",
        "seed_query": candidate["seed_query"],
        "doc_themes": sorted(doc_themes),
        "dominant_theme": dominant_theme_from_scores(scores),
        "specificity": float(specificity),
        "text": text[:30000],
    }


def collect_antra_docs():
    print("Collecting ANTRA docs from fixed specialized queries...")
    candidates = []
    seen_urls = set()

    for query in FIXED_QUERIES:
        results = search_antra(query)
        print(f"[ANTRA search] query={query!r} results={len(results)}")
        for item in results:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                candidates.append(item)

    docs = []
    seen_doc_keys = set()
    for cand in candidates:
        doc = fetch_antra_doc(cand)
        if not doc:
            continue
        key = (normalize_text(doc["title"]), doc["url"])
        if key in seen_doc_keys:
            continue
        seen_doc_keys.add(key)
        docs.append(doc)
        print(f"[ANTRA doc] themes={','.join(doc['doc_themes'])} | {doc['title'][:80]}")
        if len(docs) >= MAX_DOCS_FROM_ANTRA:
            break

    print(f"ANTRA docs collected: {len(docs)}")
    return docs


# ==========================
# RANK FOR EACH VIDEO
# ==========================

def rank_antra_docs_for_video(video_title: str, video_entities, antra_docs):
    if not antra_docs:
        return []

    if not should_query_antra(video_title):
        return []

    video_themes = detect_video_themes(video_title)
    if not video_themes:
        return []

    video_theme_scores = theme_scores(video_title, VIDEO_THEME_KEYWORDS)

    filtered_docs = []
    for d in antra_docs:
        doc_themes = set(d.get("doc_themes", []))
        if not (doc_themes & video_themes):
            continue
        overlap_strength = sum(video_theme_scores.get(t, 0) for t in doc_themes)
        if overlap_strength <= 0:
            continue
        if d.get("specificity", 0.0) < 1.5:
            continue
        filtered_docs.append(d)

    if not filtered_docs:
        return []

    doc_texts = [f"{d['title']}. {d['text'][:4000]}" for d in filtered_docs]
    title_embedding = encode_texts([video_title])[0]
    doc_embeddings = encode_texts(doc_texts)

    semantic_scores = [semantic_similarity(title_embedding, emb) for emb in doc_embeddings]

    tokenized = [tokenize(text) for text in doc_texts]
    bm25 = BM25Okapi(tokenized)
    lexical_scores = bm25.get_scores(tokenize(video_title))
    lexical_scores = lexical_scores / (np.max(lexical_scores) + 1e-6)

    ranked = []
    title_tokens = tokenize(video_title)
    for i, doc in enumerate(filtered_docs):
        title_norm = normalize_text(doc["title"])
        text_head = normalize_text(doc["text"][:3500])
        doc_themes = set(doc.get("doc_themes", []))

        has_entity = any(normalize_text(ent) in title_norm or normalize_text(ent) in text_head for ent in video_entities)
        title_overlap_hits = sum(1 for tok in title_tokens[:12] if tok in text_head)
        has_title_overlap = title_overlap_hits >= 3
        theme_overlap = len(doc_themes & video_themes)
        overlap_strength = sum(video_theme_scores.get(t, 0) for t in doc_themes)
        query_match = keyword_hits(f"{doc['title']}. {doc['text'][:2500]}", FIXED_QUERIES) > 0
        specificity = float(doc.get("specificity", 0.0))

        score = 0.60 * semantic_scores[i] + 0.40 * lexical_scores[i]
        if has_entity:
            score *= ENTITY_BOOST
        if has_title_overlap:
            score *= TITLE_BOOST
        if theme_overlap:
            score *= THEME_BOOST
        if query_match:
            score *= QUERY_BOOST
        score *= (1.0 + min(overlap_strength, 3) * 0.10)
        score *= (1.0 + min(specificity, 4.0) * 0.08)

        if doc_is_overly_generic(doc["title"], doc["text"]):
            score *= 0.55

        ranked.append({
            **doc,
            "score": float(score),
            "has_entity": has_entity,
            "theme_match": bool(theme_overlap),
            "theme_overlap": theme_overlap,
            "overlap_strength": overlap_strength,
            "has_title_overlap": has_title_overlap,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    used_titles = set()
    used_dominant_themes = set()

    for doc in ranked:
        dominant_theme = doc.get("dominant_theme")
        norm_title = normalize_text(doc["title"])

        strong_match = (
            doc["theme_match"] and
            doc["overlap_strength"] >= 1 and
            (doc["score"] >= SIMILARITY_THRESHOLD or doc["has_entity"] or doc["has_title_overlap"])
        )
        if not strong_match:
            continue
        if norm_title in used_titles:
            continue

        if dominant_theme in used_dominant_themes and len(selected) >= 1:
            continue

        used_titles.add(norm_title)
        if dominant_theme:
            used_dominant_themes.add(dominant_theme)
        selected.append(doc)

        if len(selected) >= MAX_ANTRA_DOCS_PER_VIDEO:
            break

    return selected


# ==========================
# METADATA / FAISS
# ==========================

def load_existing_metadata():
    rows = []
    if not os.path.exists(METADATA_PATH):
        return rows
    with open(METADATA_PATH, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def existing_chunk_keys(rows):
    keys = set()
    for row in rows:
        key = (
            str(row.get("video_id", "")),
            normalize_text(row.get("title", "")),
            row.get("source", ""),
            int(row.get("chunk_index", 0)),
        )
        keys.add(key)
    return keys


def append_metadata(rows):
    if not rows:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(METADATA_PATH, "a", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rebuild_faiss_from_metadata(rows):
    texts = [row["text"] for row in rows if row.get("text")]
    if not texts:
        print("No texts found to rebuild FAISS.")
        return
    print("Rebuilding FAISS from full metadata...")
    embeddings = encode_texts(texts)
    embeddings = np.asarray(embeddings).astype("float32")
    if NORMALIZE_EMBEDDINGS:
        index = faiss.IndexFlatIP(embeddings.shape[1])
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS rebuilt with {len(texts)} chunks")


# ==========================
# MAIN
# ==========================

def main():
    titles_df = pd.read_csv(TITLES_PATH)
    antra_docs = collect_antra_docs()
    if not antra_docs:
        print("No ANTRA docs collected. Exiting.")
        return

    metadata_rows = load_existing_metadata()
    seen_keys = existing_chunk_keys(metadata_rows)
    new_rows = []

    for _, row in titles_df.iterrows():
        video_id = row["video_id"]
        video_title = row["video_title"]
        entities = extract_entities_from_title(video_title)
        video_themes = detect_video_themes(video_title)

        if not video_themes or not should_query_antra(video_title):
            continue

        selected_docs = rank_antra_docs_for_video(video_title, entities, antra_docs)
        print(f"[VIDEO] {video_id} themes={','.join(sorted(video_themes))} selected_antra_docs={len(selected_docs)} should_query={should_query_antra(video_title)}")

        for doc in selected_docs:
            chunks = chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                entry = {
                    "video_id": video_id,
                    "title": doc["title"],
                    "source": "antra",
                    "url": doc["url"],
                    "chunk_index": i,
                    "text": f"{doc['title']}. {chunk}",
                    "score": float(doc["score"]),
                }
                key = (str(video_id), normalize_text(doc["title"]), "antra", i)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                new_rows.append(entry)

    append_metadata(new_rows)
    metadata_rows.extend(new_rows)
    rebuild_faiss_from_metadata(metadata_rows)
    print(f"Done. Appended {len(new_rows)} ANTRA chunks.")


if __name__ == "__main__":
    main()
