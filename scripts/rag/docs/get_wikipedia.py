import os
import json
import faiss
import requests
import numpy as np
import pandas as pd
import spacy
import time
import nltk
import re

from rake_nltk import Rake
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords

# ==========================
# CONFIG
# ==========================

TITLES_PATH = "./data/titles/video_titles.csv"
OUTPUT_DIR = "./data/wiki"

INDEX_PATH = f"{OUTPUT_DIR}/faiss_index.bin"
METADATA_PATH = f"{OUTPUT_DIR}/metadata.json"
EVAL_PATH = f"{OUTPUT_DIR}/eval.json"

ENTITY_LIMIT = 5
CHUNK_SIZE = 250

SIMILARITY_THRESHOLD = 0.02
TOP_K_SEARCH = 5

REQUEST_DELAY = 1.2  # evita 429
MAX_ARTICLES_PER_VIDEO = 5

BATCH_SIZE = 32
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

WIKI_API = "https://pt.wikipedia.org/w/api.php"

search_cache = {}

# ==========================
# LOAD MODELS
# ==========================

print("Loading models...")

nlp = spacy.load("pt_core_news_lg")
embedder = SentenceTransformer(MODEL_NAME)

nltk.download("stopwords")
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

# ==========================
# CONTEXT DETECTOR
# ==========================

def detect_context_type(title):

    t = title.lower()

    if any(w in t for w in ["lei", "stf", "stj", "mpf", "governo", "câmara", "projeto", "trump", "nikolas"]):
        return "political"

    if any(w in t for w in ["morte", "assassinato", "agredida", "violência", "presa"]):
        return "violence"

    if any(w in t for w in ["hormônio", "cirurgia", "câncer", "saúde", "transição"]):
        return "medical"

    if any(w in t for w in ["ator", "atriz", "cantor", "modelo", "deputada", "quem é"]):
        return "biographical"

    if any(w in t for w in ["povos", "comunidade", "sociedade", "educação"]):
        return "cultural"

    return "general"

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

# ==========================
# QUERY BUILDER
# ==========================

def build_queries(video_title, entities, keyphrases):

    queries = [video_title]

    context_type = detect_context_type(video_title)
    context_queries = CONTEXT_MAP[context_type]

    for ent in entities:
        queries.append(f"{ent} pessoa trans")
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

    headers = {
        "User-Agent": "MyResearchBot/1.0"
    }

    for attempt in range(5):

        try:
            time.sleep(REQUEST_DELAY)

            r = requests.get(WIKI_API, params=params, headers=headers)

            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"429 → sleep {wait}s")
                time.sleep(wait)
                continue

            if r.status_code != 200:
                return []

            data = r.json()

            results = [
                x["title"]
                for x in data["query"]["search"][:TOP_K_SEARCH]
            ]

            search_cache[query] = results
            return results

        except:
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

    headers = {"User-Agent": "MyResearchBot/1.0"}

    try:
        time.sleep(REQUEST_DELAY)

        r = requests.get(WIKI_API, params=params, headers=headers)

        if r.status_code != 200:
            return None

        pages = r.json()["query"]["pages"]
        page = list(pages.values())[0]

        return page.get("extract", None)

    except:
        return None


def fetch_wikipedia_articles(titles):

    articles = {}

    for t in titles:
        text = wikipedia_page(t)
        if text and len(text) > 100:
            articles[t] = text

    return articles

# ==========================
# METRICS
# ==========================

def compute_metrics(video_title, retrieved_titles):

    gt_tokens = set(video_title.lower().split())

    relevant = [
        t for t in retrieved_titles
        if any(tok in t.lower() for tok in gt_tokens)
    ]

    recall = len(relevant) / len(gt_tokens) if gt_tokens else 0
    precision = len(relevant) / len(retrieved_titles) if retrieved_titles else 0

    return recall, precision

# ==========================
# PIPELINE
# ==========================

titles_df = pd.read_csv(TITLES_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata = []
eval_results = []

for _, row in tqdm(titles_df.iterrows(), total=len(titles_df)):

    video_id = row["video_id"]
    video_title = row["video_title"]

    entities = extract_entities(video_title)
    keyphrases = extract_keyphrases(video_title)

    queries = build_queries(video_title, entities, keyphrases)

    print("\nTITLE:", video_title)
    print("Queries:", queries)

    titles = set()

    for q in queries:
        results = wikipedia_search(q)
        titles.update(results)

    titles = list(titles)

    articles = fetch_wikipedia_articles(titles)

    print("Num titles:", len(titles))
    print("Num articles:", len(articles))

    docs = list(articles.values())
    doc_titles = list(articles.keys())

    if not docs:
        continue

    title_embedding = embedder.encode(video_title)

    doc_embeddings = embedder.encode(
        [d[:2000] for d in docs],
        batch_size=BATCH_SIZE
    )

    semantic_scores = [
        cosine_similarity([title_embedding], [emb])[0][0]
        for emb in doc_embeddings
    ]

    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)

    q_tokens = video_title.lower().split()
    lexical_scores = bm25.get_scores(q_tokens)
    lexical_scores = lexical_scores / (np.max(lexical_scores) + 1e-6)

    scores = []

    for i, title in enumerate(doc_titles):
        score = 0.7 * semantic_scores[i] + 0.3 * lexical_scores[i]
        scores.append((title, docs[i], score))

    scores.sort(key=lambda x: x[2], reverse=True)
    scores = scores[:MAX_ARTICLES_PER_VIDEO]

    # ==========================
    # METRICS
    # ==========================

    retrieved_titles = [s[0] for s in scores]
    recall, precision = compute_metrics(video_title, retrieved_titles)

    eval_results.append({
        "video_id": video_id,
        "recall": recall,
        "precision": precision,
        "num_docs": len(retrieved_titles)
    })

    # ==========================
    # CHUNKS
    # ==========================

    for wiki_title, article, score in scores:

        if score < SIMILARITY_THRESHOLD:
            continue

        chunks = [
            " ".join(article.split()[i:i+CHUNK_SIZE])
            for i in range(0, len(article.split()), CHUNK_SIZE)
        ]

        for i, chunk in enumerate(chunks):

            metadata.append({
                "video_id": video_id,
                "title": wiki_title,
                "chunk_index": i,
                "text": f"{wiki_title}. {chunk}",
                "score": float(score)
            })

# ==========================
# SAVE
# ==========================

with open(METADATA_PATH, "w", encoding="utf8") as f:
    json.dump(metadata, f, ensure_ascii=False)

with open(EVAL_PATH, "w", encoding="utf8") as f:
    json.dump(eval_results, f, ensure_ascii=False)

# ==========================
# FAISS
# ==========================

print("Building FAISS index...")

texts = [m["text"] for m in metadata]

if texts:

    embeddings = embedder.encode(texts, batch_size=BATCH_SIZE)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

print("Done")
print("Chunks:", len(metadata))