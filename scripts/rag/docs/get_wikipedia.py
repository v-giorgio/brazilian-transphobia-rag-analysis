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
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# ==========================
# CONFIG
# ==========================

TITLES_PATH = "./data/titles/video_titles.csv"

OUTPUT_DIR = "./data/wiki"

INDEX_PATH = f"{OUTPUT_DIR}/faiss_index.bin"
METADATA_PATH = f"{OUTPUT_DIR}/metadata.json"

ENTITY_LIMIT = 6
CHUNK_SIZE = 250

SIMILARITY_THRESHOLD = 0.02

TOP_K_SEARCH = 5
MAX_WORKERS = 5
REQUEST_DELAY = 0.2

MAX_ARTICLES_PER_VIDEO = 5

BATCH_SIZE = 32

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

WIKI_API = "https://pt.wikipedia.org/w/api.php"

# ==========================
# LOAD MODELS
# ==========================

print("Loading models...")

nlp = spacy.load("pt_core_news_lg")
embedder = SentenceTransformer(MODEL_NAME)

nltk.download("stopwords")
nltk.download("punkt")

rake = Rake(language="portuguese")

# ==========================
# FIXED QUERIES
# ==========================

FIXED_QUERIES = [
    "identidade de gênero",
    "pessoas trans",
    "transexualidade",
    "transfobia",
    "direitos trans",
    "nome social",
]

# ==========================
# CLEAN TITLE / QUERY
# ==========================

def clean_video_title(title):
    if not isinstance(title, str):
        return ""
    return title.split("|")[0].strip()


def clean_query(q):
    q = q.lower()
    q = re.sub(r"[^a-zA-Z0-9áéíóúãõç ]", "", q)
    return q.strip()

# ==========================
# NLP
# ==========================

def extract_entities(text):

    if not text:
        return []

    doc = nlp(text[:500])

    entities = [
        e.text for e in doc.ents
        if e.label_ in ["PER","ORG","MISC"]
    ]

    return list(dict.fromkeys(entities))[:ENTITY_LIMIT]


def extract_keyphrases(text):

    rake.extract_keywords_from_text(text)

    phrases = rake.get_ranked_phrases()

    phrases = [p for p in phrases if 2 <= len(p.split()) <= 4]

    return phrases[:5]

# ==========================
# WIKIPEDIA SEARCH
# ==========================

def wikipedia_search(query):

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json"
    }

    headers = {"User-Agent": "research-bot"}

    time.sleep(REQUEST_DELAY)

    try:
        r = requests.get(WIKI_API, params=params, headers=headers)

        if r.status_code != 200:
            return []

        data = r.json()

        return [x["title"] for x in data["query"]["search"][:TOP_K_SEARCH]]

    except:
        return []

# ==========================
# FETCH TITLES
# ==========================

def fetch_wikipedia_titles(video_title, title_entities, keyphrases):

    queries = []

    queries.append(video_title)

    for ent in title_entities:
        queries.append(ent)
        queries.append(f'intitle:"{ent}"')

    queries += keyphrases
    queries += FIXED_QUERIES

    titles = set()

    for q in queries:

        q = clean_query(q)

        if not q:
            continue

        results = wikipedia_search(q)

        for r in results:
            titles.add(r)

    return list(titles)

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

    headers = {"User-Agent": "research-bot"}

    time.sleep(REQUEST_DELAY)

    try:
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

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(wikipedia_page, t): t
            for t in titles
        }

        for future in as_completed(futures):

            title = futures[future]

            try:
                text = future.result()

                if text and len(text) > 300:
                    articles[title] = text

            except:
                pass

    return articles

# ==========================
# CHUNK
# ==========================

def chunk_text(text):

    words = text.split()

    return [
        " ".join(words[i:i+CHUNK_SIZE])
        for i in range(0, len(words), CHUNK_SIZE)
    ]

# ==========================
# LOAD DATA
# ==========================

titles_df = pd.read_csv(TITLES_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(METADATA_PATH):
    with open(METADATA_PATH,"w",encoding="utf8") as f:
        json.dump([],f)

# ==========================
# LOAD CHECKPOINT
# ==========================

print("Loading checkpoint...")

with open(METADATA_PATH,"r",encoding="utf8") as f:
    metadata = json.load(f)

processed_videos = set(m["video_id"] for m in metadata)

# ==========================
# PIPELINE
# ==========================

for _, row in tqdm(titles_df.iterrows(), total=len(titles_df)):

    video_id = row["video_id"]

    if video_id in processed_videos:
        continue

    video_title = clean_video_title(row["video_title"])

    if not video_title:
        continue

    title_entities = extract_entities(video_title)
    keyphrases = extract_keyphrases(video_title)

    print("\nTITLE:", video_title)
    print("Entities:", title_entities)

    title_embedding = embedder.encode(video_title)

    titles = fetch_wikipedia_titles(
        video_title,
        title_entities,
        keyphrases
    )

    articles = fetch_wikipedia_articles(titles)

    docs = list(articles.values())
    doc_titles = list(articles.keys())

    if docs:

        doc_embeddings = embedder.encode(
            [d[:2000] for d in docs],
            batch_size=BATCH_SIZE
        )

        semantic_scores = [
            cosine_similarity([title_embedding],[emb])[0][0]
            for emb in doc_embeddings
        ]

        tokenized = [d.lower().split() for d in docs]
        bm25 = BM25Okapi(tokenized)

        q_tokens = " ".join(title_entities + keyphrases).lower().split()

        if q_tokens:
            lexical_scores = bm25.get_scores(q_tokens)
            lexical_scores = lexical_scores / (np.max(lexical_scores) + 1e-6)
        else:
            lexical_scores = np.zeros(len(docs))

        scores = []

        for i,title in enumerate(doc_titles):

            score = 0.7*semantic_scores[i] + 0.3*lexical_scores[i]

            scores.append((title,docs[i],score))

        scores.sort(key=lambda x:x[2], reverse=True)

        scores = scores[:MAX_ARTICLES_PER_VIDEO]

        for wiki_title,article,score in scores:

            if score < SIMILARITY_THRESHOLD:
                continue

            chunks = chunk_text(article)

            for i,chunk in enumerate(chunks):

                metadata.append({
                    "video_id": video_id,
                    "title": wiki_title,
                    "chunk_index": i,
                    "text": f"{wiki_title}. {chunk}",
                    "score": float(score)
                })

    # checkpoint incremental
    with open(METADATA_PATH,"w",encoding="utf8") as f:
        json.dump(metadata,f,ensure_ascii=False)

# ==========================
# BUILD FAISS
# ==========================

print("Building FAISS index...")

texts = [m["text"] for m in metadata]

if texts:

    embeddings = embedder.encode(texts, batch_size=BATCH_SIZE)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index,INDEX_PATH)

print("Wikipedia corpus created")
print("Chunks:",len(metadata))