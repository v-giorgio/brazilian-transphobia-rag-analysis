import os
import re
import json
import faiss
import requests
import numpy as np
import pandas as pd
import spacy
import time
import nltk
import unicodedata

from rake_nltk import Rake
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================
# CONFIG
# ==========================

TITLES_PATH = "./data/titles/video_titles.csv"
OUTPUT_DIR = "./data/wiki"

INDEX_PATH = f"{OUTPUT_DIR}/faiss_index.bin"
METADATA_PATH = f"{OUTPUT_DIR}/metadata.jsonl"
EVAL_PATH = f"{OUTPUT_DIR}/eval.jsonl"

ENTITY_LIMIT = 5
CHUNK_SIZE = 250
SIMILARITY_THRESHOLD = 0.02
TOP_K_SEARCH = 10
MAX_ARTICLES_PER_VIDEO = 5
REQUEST_DELAY = 0.8
BATCH_SIZE = 32
MODEL_NAME = "BAAI/bge-m3"
WIKI_API = "https://pt.wikipedia.org/w/api.php"

NORMALIZE_EMBEDDINGS = True

search_cache = {}
session = requests.Session()
session.headers.update({"User-Agent": "MyResearchBot/1.0"})

# ==========================
# LOAD MODELS
# ==========================

print("Loading models...")

nlp = spacy.load("pt_core_news_lg")
embedder = SentenceTransformer(MODEL_NAME)

nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("punkt")

STOPWORDS = set(stopwords.words("portuguese"))

rake = Rake(language="portuguese")

# ==========================
# CONTEXT MAP
# ==========================

CONTEXT_MAP = {
    "political": [
        "direitos LGBT",
        "identidade de gênero",
        "política",
        "legislação",
        "controversia"
    ],
    "violence": [
        "transfobia",
        "violência contra pessoas trans",
        "crimes de ódio",
    ],
    "medical": [
        "transexualidade",
        "transição de gênero",
        "hormonioterapia",
        "saúde pública",
    ],
    "biographical": [
        "biografia",
        "carreira",
        "vida pessoal",
        "cultura pop",
    ],
    "cultural": [
        "cultura",
        "sociedade",
        "antropologia",
    ],
    "general": [
        "pessoa trans",
        "identidade de gênero",
    ]
}

TRANS_THEME_TERMS = {
    "trans", "travesti", "travestis", "transexual", "transexuais", "transgenero",
    "transgênero", "transfobia", "identidade de genero", "identidade de gênero",
    "nome social", "mulher trans", "homem trans", "pessoas trans", "população trans",
    "processo transexualizador", "gênero", "genero", "não binário", "nao binario"
}

# ==========================
# HELPERS
# ==========================

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
    return [t for t in normalize_text(text).split() if t and t not in STOPWORDS and len(t) > 1]


def detect_context_type(title):
    t = title.lower()

    if any(w in t for w in ["lei", "stf", "stj", "mpf", "governo", "câmara", "projeto", "trump", "nikolas"]):
        return "political"
    if any(w in t for w in ["morte", "assassinato", "agredida", "violência", "presa"]):
        return "violence"
    if any(w in t for w in ["hormônio", "cirurgia", "câncer", "saúde", "transição", "ambulatório"]):
        return "medical"
    if any(w in t for w in ["ator", "atriz", "cantor", "modelo", "deputada", "quem é"]):
        return "biographical"
    if any(w in t for w in ["povos", "comunidade", "sociedade", "educação"]):
        return "cultural"
    return "general"


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


# ==========================
# NLP
# ==========================

def extract_entities(text):
    doc = nlp(text)
    entities = [
        e.text.strip()
        for e in doc.ents
        if e.label_ in ["PER", "ORG", "LOC", "MISC"]
    ]
    return list(dict.fromkeys(entities))[:ENTITY_LIMIT]


def extract_keyphrases(text):
    doc = nlp(text)
    phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
    rake.extract_keywords_from_text(text)
    phrases += rake.get_ranked_phrases()

    cleaned = []
    for p in phrases:
        if len(p.split()) < 2:
            continue
        if any(w in STOPWORDS for w in p.split()):
            continue
        cleaned.append(p)
    return list(dict.fromkeys(cleaned))[:5]


def extract_theme_terms(video_title, keyphrases):
    terms = set()
    title_norm = normalize_text(video_title)
    for term in TRANS_THEME_TERMS:
        if term in title_norm:
            terms.add(term)
    for kp in keyphrases:
        kp_norm = normalize_text(kp)
        if any(t in kp_norm for t in ["trans", "travesti", "transexual", "genero", "gênero", "nome social", "transfobia", "ambulatório", "psicologo", "psiquiatra", "saude", "saúde"]):
            terms.add(kp_norm)
    context_type = detect_context_type(video_title)
    terms.update(normalize_text(x) for x in CONTEXT_MAP.get(context_type, []))
    terms.update(normalize_text(x) for x in CONTEXT_MAP["general"])
    return {t for t in terms if t}


# ==========================
# QUERY BUILDER
# ==========================

def build_queries(video_title, entities, keyphrases):
    queries = [video_title]
    context_type = detect_context_type(video_title)
    context_queries = CONTEXT_MAP[context_type]

    for ent in entities:
        queries.append(ent)
        queries.append(f"{ent} trans")
        for ctx in context_queries:
            queries.append(f"{ent} {ctx}")

    for kp in keyphrases:
        for ctx in context_queries:
            queries.append(f"{kp} {ctx}")

    queries += CONTEXT_MAP["general"]
    return list(dict.fromkeys(queries))[:15]

# ==========================
# WIKIPEDIA SEARCH
# ==========================

def wikipedia_search(query):
    if query in search_cache:
        return search_cache[query]

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json"
    }

    for attempt in range(5):
        try:
            time.sleep(REQUEST_DELAY)
            r = session.get(WIKI_API, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if r.status_code != 200:
                return []
            data = r.json()
            results = [x["title"] for x in data.get("query", {}).get("search", [])[:TOP_K_SEARCH]]
            search_cache[query] = results
            return results
        except Exception:
            time.sleep(1)
    return []


# ==========================
# FETCH ARTICLES
# ==========================

def wikipedia_page(title):
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "explaintext": True,
        "format": "json"
    }
    try:
        time.sleep(REQUEST_DELAY)
        r = session.get(WIKI_API, params=params, timeout=30)
        if r.status_code == 429:
            return None
        if r.status_code != 200:
            return None
        pages = r.json().get("query", {}).get("pages", {})
        if not pages:
            return None
        page = list(pages.values())[0]
        return page.get("extract", None)
    except Exception:
        return None


def fetch_wikipedia_articles(titles):
    articles = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(wikipedia_page, t): t for t in titles}
        for future in as_completed(futures):
            t = futures[future]
            text = future.result()
            if text and len(text) > 100:
                articles[t] = text
    return articles

# ==========================
# FILTER / RERANK
# ==========================

def title_has_entity(title, entities):
    title_norm = normalize_text(title)
    return any(normalize_text(ent) in title_norm for ent in entities if ent)


def text_has_entity(text, entities):
    text_norm = normalize_text(text[:1200])
    return any(normalize_text(ent) in text_norm for ent in entities if ent)


def has_theme_match(text, title, theme_terms):
    joined = normalize_text(f"{title}. {text[:1600]}")
    if any(term in joined for term in theme_terms):
        return True
    return False


def is_offtopic_associated_doc(title, text, entities, theme_terms):
    if title_has_entity(title, entities):
        return False
    if text_has_entity(text, entities):
        return False
    if has_theme_match(text, title, theme_terms):
        return False
    return True


def filter_and_rank_candidates(articles, video_title, entities, keyphrases):
    docs = list(articles.values())
    doc_titles = list(articles.keys())
    if not docs:
        return []

    theme_terms = extract_theme_terms(video_title, keyphrases)

    kept_titles = []
    kept_docs = []
    for title, text in zip(doc_titles, docs):
        if is_offtopic_associated_doc(title, text, entities, theme_terms):
            continue
        kept_titles.append(title)
        kept_docs.append(text)

    if not kept_docs:
        for title, text in zip(doc_titles, docs):
            if title_has_entity(title, entities):
                kept_titles.append(title)
                kept_docs.append(text)

    if not kept_docs:
        return []

    title_embedding = encode_texts([video_title])[0]
    doc_embeddings = encode_texts([d[:2000] for d in kept_docs])
    semantic_scores = [semantic_similarity(title_embedding, emb) for emb in doc_embeddings]

    tokenized = [d.lower().split() for d in kept_docs]
    bm25 = BM25Okapi(tokenized)
    q_tokens = video_title.lower().split()
    lexical_scores = bm25.get_scores(q_tokens)
    lexical_scores = lexical_scores / (np.max(lexical_scores) + 1e-6)

    scores = []
    ENTITY_TITLE_BOOST = 3.0
    ENTITY_TEXT_BOOST = 1.8
    THEME_BOOST = 1.5

    for i, title in enumerate(kept_titles):
        base_score = 0.7 * semantic_scores[i] + 0.3 * lexical_scores[i]
        score = base_score

        if title_has_entity(title, entities):
            score *= ENTITY_TITLE_BOOST
        elif text_has_entity(kept_docs[i], entities):
            score *= ENTITY_TEXT_BOOST

        if has_theme_match(kept_docs[i], title, theme_terms):
            score *= THEME_BOOST

        scores.append((title, kept_docs[i], score))

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:MAX_ARTICLES_PER_VIDEO]

# ==========================
# METRICS
# ==========================

def compute_metrics(video_title, retrieved_titles):
    gt_tokens = set(video_title.lower().split())
    relevant = [t for t in retrieved_titles if any(tok in t.lower() for tok in gt_tokens)]
    recall = len(relevant) / len(gt_tokens) if gt_tokens else 0
    precision = len(relevant) / len(retrieved_titles) if retrieved_titles else 0
    return recall, precision

# ==========================
# RESUME HELPERS
# ==========================

def load_existing_metadata(path):
    existing = []
    seen_chunks = set()
    processed_video_ids = set()

    if not os.path.exists(path):
        return existing, seen_chunks, processed_video_ids

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            existing.append(row)
            source = row.get("source", "wikipedia")
            if source in {"wikipedia", "wikidata"}:
                processed_video_ids.add(row.get("video_id"))
            chunk_id = (row.get("video_id"), row.get("title"), row.get("chunk_index"), source)
            seen_chunks.add(chunk_id)
    return existing, seen_chunks, processed_video_ids


def load_processed_eval_video_ids(path):
    processed = set()
    if not os.path.exists(path):
        return processed
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                processed.add(row.get("video_id"))
            except Exception:
                continue
    return processed


# ==========================
# PIPELINE
# ==========================

def main():
    titles_df = pd.read_csv(TITLES_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata, seen_chunks, processed_from_metadata = load_existing_metadata(METADATA_PATH)
    processed_from_eval = load_processed_eval_video_ids(EVAL_PATH)
    already_processed = {x for x in processed_from_metadata | processed_from_eval if x}

    if already_processed:
        titles_df = titles_df[~titles_df["video_id"].isin(already_processed)].copy()
        print(f"Skipping {len(already_processed)} already processed Wikipedia/Wikidata videos")

    if titles_df.empty:
        print("No new videos to process")
        print("Rebuilding FAISS from existing metadata...")
        rebuild_faiss_from_metadata(METADATA_PATH)
        return

    for _, row in titles_df.iterrows():
        video_id = row["video_id"]
        video_title = row["video_title"]

        entities = extract_entities(video_title)
        keyphrases = extract_keyphrases(video_title)
        queries = build_queries(video_title, entities, keyphrases)

        titles = set()
        for q in queries:
            titles.update(wikipedia_search(q))

        articles = fetch_wikipedia_articles(list(titles))
        scores = filter_and_rank_candidates(articles, video_title, entities, keyphrases)

        retrieved_titles = [s[0] for s in scores]
        recall, precision = compute_metrics(video_title, retrieved_titles)

        eval_entry = {
            "video_id": video_id,
            "recall": recall,
            "precision": precision,
            "num_docs": len(retrieved_titles)
        }
        with open(EVAL_PATH, "a", encoding="utf8") as f:
            f.write(json.dumps(eval_entry, ensure_ascii=False) + "\n")

        new_metadata = []
        for wiki_title, article, score in scores:
            if score < SIMILARITY_THRESHOLD:
                continue

            chunks = [
                " ".join(article.split()[i:i+CHUNK_SIZE])
                for i in range(0, len(article.split()), CHUNK_SIZE)
            ]

            for i, chunk in enumerate(chunks):
                entry = {
                    "video_id": video_id,
                    "title": wiki_title,
                    "source": "wikipedia",
                    "chunk_index": i,
                    "text": f"{wiki_title}. {chunk}",
                    "score": float(score)
                }
                chunk_id = (video_id, wiki_title, i, "wikipedia")
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    metadata.append(entry)
                    new_metadata.append(entry)

        with open(METADATA_PATH, "a", encoding="utf8") as f:
            for entry in new_metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[VIDEO DONE] {video_id} | docs={len(retrieved_titles)} | chunks={len(new_metadata)} | top_titles={retrieved_titles[:3]}")

    rebuild_faiss_from_metadata(METADATA_PATH)
    print("Done")


def rebuild_faiss_from_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        print("metadata.jsonl not found; skipping FAISS rebuild")
        return

    all_rows = []
    with open(metadata_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_rows.append(json.loads(line))
                except Exception:
                    continue

    texts = [row.get("text", "") for row in all_rows if row.get("text")]
    if not texts:
        print("No texts found to index")
        return

    print("Building FAISS index from full metadata...")
    embeddings = encode_texts(texts)
    embeddings = np.array(embeddings).astype("float32")

    if NORMALIZE_EMBEDDINGS:
        index = faiss.IndexFlatIP(embeddings.shape[1])
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS rebuilt with {len(texts)} chunks")


if __name__ == "__main__":
    main()
