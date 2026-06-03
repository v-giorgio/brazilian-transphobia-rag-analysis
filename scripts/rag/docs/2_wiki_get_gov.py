import os
import re
import json
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import faiss
import numpy as np
import pandas as pd
import requests
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
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40
SIMILARITY_THRESHOLD = 0.34
MAX_GOV_DOCS_PER_VIDEO = 2
MAX_DOCS_FROM_GOV = 24
MAX_PAGES_PER_SEED = 14
MAX_CRAWL_DEPTH = 1

NORMALIZE_EMBEDDINGS = True

USER_AGENT = "TransGovCollector/1.0"

SEED_URLS = [
    "https://www.gov.br/mdh/pt-br/navegue-por-temas/lgbt",
    "https://www.gov.br/mdh/pt-br/assuntos/noticias/2025/janeiro/direitos-humanos-anuncia-agenda-de-enfrentamento-a-violencia-contra-pessoas-lgbtqia",
    "https://www.gov.br/saude/pt-br/assuntos/noticias/2024/dezembro/ministerio-da-saude-apresenta-o-programa-de-atencao-a-saude-da-populacao-trans",
    "https://www.gov.br/saude/pt-br/assuntos/noticias/2026/janeiro/ministerio-da-saude-publica-orientacoes-para-garantir-o-uso-do-nome-social-e-a-seguranca-clinica-de-pessoas-trans-no-sus",
    "https://www.gov.br/saude/pt-br/composicao/saes/saips/manuais/manuais-cgae/orientacoes-para-habilitacao-processo-trans.pdf",
    "https://www.gov.br/pt-br/servicos/habilitar-se-para-realizar-processo-transexualizador",
    "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2016/decreto/d8727.htm",
    "https://www.gov.br/mdh/pt-br/assuntos/noticias/2024/janeiro/mdhc-destaca-cinco-perguntas-para-nao-fazer-a-uma-pessoa-trans",
    "https://www.gov.br/mulheres/pt-br/central-de-conteudos/publicacoes/cartilha-dignidade-e-cidadania-para-travestis-e-mulheres-trans-vf.pdf",
    "https://www.gov.br/mdh/pt-br/assuntos/noticias/2025/junho/mdhc-atua-pela-protecao-acolhimento-e-empregabilidade-da-populacao-lgbtqia-em-todos-os-territorios",
]

FIXED_QUERIES = [
    "nome social pessoas trans sus",
    "processo transexualizador sus",
    "violência contra pessoas lgbtqia política pública",
    "ambulatório população trans sus",
    "discriminação pessoas trans direitos humanos",
    "educação pessoas trans direitos",
    "empregabilidade população lgbtqia governo federal",
]

VIDEO_THEME_KEYWORDS = {
    "health": [
        "ambulatório", "ambulatorio", "sus", "saúde", "saude", "hormônio", "hormonio",
        "psicólogo", "psicologo", "psiquiatra", "atendimento", "hospital", "clínica", "clinica",
        "processo transexualizador", "nome social"
    ],
    "rights": [
        "direito", "direitos", "nome social", "identidade de gênero", "identidade de genero",
        "discriminação", "discriminacao", "retificação", "retificacao", "registro", "cartório", "cartorio",
        "decreto", "lei", "stf", "cidadania"
    ],
    "violence": [
        "violência", "violencia", "transfobia", "agressão", "agressao", "crime", "ódio", "odio",
        "ataque", "ameaça", "ameaca", "assassinato", "morte"
    ],
    "education_work": [
        "escola", "educação", "educacao", "universidade", "professor", "aluno",
        "trabalho", "emprego", "empregabilidade", "renda"
    ],
}

DOC_THEME_KEYWORDS = {
    "health": [
        "saúde", "saude", "sus", "ambulatório", "ambulatorial", "hospitalar", "hormonioterapia",
        "processo transexualizador", "autodeterminação", "autodeterminacao", "nome social"
    ],
    "rights": [
        "direitos humanos", "nome social", "identidade de gênero", "identidade de genero",
        "discriminação", "discriminacao", "retificação", "retificacao", "cidadania", "decreto"
    ],
    "violence": [
        "violência", "violencia", "transfobia", "enfrentamento à violência", "enfrentamento a violencia",
        "discriminação", "discriminacao", "proteção", "protecao", "acolhimento"
    ],
    "education_work": [
        "educação", "educacao", "escola", "trabalho", "empregabilidade", "renda", "trabalho digno"
    ],
}

STRONG_VIDEO_TERMS = [
    "ambulatório", "ambulatorio", "sus", "nome social", "processo transexualizador",
    "transfobia", "discriminação", "discriminacao", "violência", "violencia",
    "direitos", "retificação", "retificacao", "saúde", "saude", "educação", "educacao",
    "empregabilidade", "trabalho"
]

WEAK_OR_BROAD_VIDEO_TERMS = [
    "cantor", "atriz", "ator", "celebridade", "filme", "série", "serie", "disse", "falou",
    "opinião", "opiniao", "polêmica", "polemica", "debate", "viral", "treta"
]

GENERIC_INSTITUTIONAL_TERMS = [
    "página inicial", "pagina inicial", "gov br", "ministerio", "ministério", "assine", "compartilhe",
    "serviços", "servicos", "menu", "acesso à informação", "acesso a informacao"
]

SPECIFIC_TRANS_TERMS = [
    "pessoa trans", "pessoas trans", "populacao trans", "população trans", "homem trans",
    "mulher trans", "travesti", "travestis", "transexual", "transexuais", "transgenero",
    "transgênero", "nao binaria", "não binária", "nome social", "identidade de genero",
    "identidade de gênero", "processo transexualizador", "ambulatório trans", "ambulatorio trans",
    "hormonioterapia", "retificacao de nome", "retificação de nome", "transfobia"
]

FALSE_POSITIVE_TRANS_TERMS = [
    "transferencia", "transferencias", "transferências", "transatlântico", "transatlantico",
    "trânsito", "transito", "transparencia", "transparência", "transmissao", "transmissão",
    "transformacao digital", "transformação digital", "transicao energetica", "transição energética"
]

BLOCKED_GOV_TITLE_PHRASES = [
    "disque direitos humanos", "disque 100", "ouvidoria nacional de direitos humanos",
    "painel de dados da ouvidoria", "conectas direitos humanos", "transferências voluntárias",
    "transferencias voluntarias", "direitos humanos e cidadania", "memoria e verdade da escravidao",
    "memória e verdade da escravidão"
]

ALLOWED_DOMAINS = {
    "www.gov.br", "gov.br", "www.planalto.gov.br", "planalto.gov.br"
}

EXCLUDED_URL_PARTS = [
    "/search", "/busca", "/feed", "/tag/", "/category/", "/author/", "?amp", "@@download"
]

SOURCE_MAP = {
    "planalto.gov.br": "planalto",
    "www.planalto.gov.br": "planalto",
    "www.gov.br": "gov_br",
    "gov.br": "gov_br",
}

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})
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
    tmp_path = "/tmp/gov_temp.pdf"
    with open(tmp_path, "wb") as f:
        f.write(content)
    reader = PdfReader(tmp_path)
    texts = []
    for page in reader.pages[:50]:
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


def keyword_hits(text: str, keywords):
    norm = normalize_text(text)
    return sum(1 for kw in keywords if normalize_text(kw) in norm)


def detect_video_themes(video_title: str):
    detected = set()
    norm = normalize_text(video_title)
    for theme, keywords in VIDEO_THEME_KEYWORDS.items():
        if any(normalize_text(kw) in norm for kw in keywords):
            detected.add(theme)
    return detected


def should_query_government(video_title: str) -> bool:
    norm = normalize_text(video_title)
    strong_hits = sum(1 for term in STRONG_VIDEO_TERMS if normalize_text(term) in norm)
    broad_hits = sum(1 for term in WEAK_OR_BROAD_VIDEO_TERMS if normalize_text(term) in norm)
    has_theme = bool(detect_video_themes(video_title))
    has_trans = any(normalize_text(t) in norm for t in SPECIFIC_TRANS_TERMS) or ("trans" in tokenize(video_title) and any(normalize_text(t) in norm for t in STRONG_VIDEO_TERMS))

    if not has_trans:
        return False
    if has_theme and strong_hits >= 1:
        return True
    if strong_hits >= 2 and broad_hits == 0:
        return True
    return False


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc not in ALLOWED_DOMAINS:
        return False
    lowered = url.lower()
    if any(part in lowered for part in EXCLUDED_URL_PARTS):
        return False
    return True


def page_title_from_html(html: str, fallback: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title:
        return re.sub(r"\s+", " ", soup.title.get_text(" ", strip=True)).strip()[:250]
    h1 = soup.find("h1")
    if h1:
        return re.sub(r"\s+", " ", h1.get_text(" ", strip=True)).strip()[:250]
    return fallback


def infer_source_from_url(url: str) -> str:
    host = urlparse(url).netloc
    return SOURCE_MAP.get(host, "gov_br")


def is_noise_gov_title(title: str) -> bool:
    norm = normalize_text(title)
    return any(normalize_text(t) in norm for t in BLOCKED_GOV_TITLE_PHRASES)


def looks_trans_relevant(text: str) -> bool:
    norm = normalize_text(text)
    if any(normalize_text(t) in norm for t in FALSE_POSITIVE_TRANS_TERMS):
        return False
    return any(normalize_text(t) in norm for t in SPECIFIC_TRANS_TERMS)


def unique_records(records):
    seen = set()
    out = []
    for rec in records:
        key = (
            normalize_text(rec["title"]),
            rec["source"],
            normalize_text(rec.get("url", "")),
        )
        if key not in seen:
            seen.add(key)
            out.append(rec)
    return out


# ==========================
# DOC COLLECTION
# ==========================

def collect_gov_documents():
    records = []
    visited = set()

    for seed_url in SEED_URLS:
        queue = deque([(seed_url, 0)])
        collected_for_seed = 0

        while queue and collected_for_seed < MAX_PAGES_PER_SEED:
            url, depth = queue.popleft()
            if url in visited or not is_allowed_url(url):
                continue
            visited.add(url)

            try:
                r = safe_get(url)
            except Exception as e:
                print(f"[WARN] failed to fetch {url}: {e}")
                continue

            ctype = (r.headers.get("Content-Type") or "").lower()
            text = ""
            title = url.split("/")[-1] or url

            try:
                if "pdf" in ctype or url.lower().endswith(".pdf"):
                    text = extract_pdf_text(r.content)
                else:
                    title = page_title_from_html(r.text, fallback=title)
                    text = html_to_text(r.text)
            except Exception as e:
                print(f"[WARN] parse failed {url}: {e}")
                continue

            if text and len(text) > 300:
                preview = f"{title}. {text[:5000]}"
                if looks_trans_relevant(preview) and not is_noise_gov_title(title):
                    records.append({
                        "title": title,
                        "source": infer_source_from_url(url),
                        "url": url,
                        "text": text[:40000],
                    })
                    collected_for_seed += 1

            if depth < MAX_CRAWL_DEPTH and "pdf" not in ctype and not url.lower().endswith(".pdf"):
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.select("a[href]"):
                    next_url = urljoin(url, a["href"])
                    if is_allowed_url(next_url) and next_url not in visited:
                        anchor_text = a.get_text(" ", strip=True)
                        hint = normalize_text(anchor_text + " " + next_url)
                        if any(normalize_text(term) in hint for term in SPECIFIC_TRANS_TERMS + STRONG_VIDEO_TERMS):
                            queue.append((next_url, depth + 1))

    records = unique_records(records)
    print(f"[INFO] collected gov docs: {len(records)}")
    return records


# ==========================
# SCORING
# ==========================

def score_document_for_video(video_title: str, entities, doc, title_embedding):
    doc_title = doc["title"]
    doc_text = doc["text"]
    full_text = f"{doc_title}. {doc_text[:2500]}"

    doc_embedding = encode_texts([full_text])[0]
    semantic_score = semantic_similarity(title_embedding, doc_embedding)

    tokenized_doc = tokenize(full_text)
    bm25 = BM25Okapi([tokenized_doc])
    lexical_score = bm25.get_scores(tokenize(video_title))[0]
    lexical_score = float(lexical_score)
    lexical_score = lexical_score / (lexical_score + 1.0)

    base_score = 0.70 * semantic_score + 0.30 * lexical_score

    title_norm = normalize_text(doc_title)
    text_norm = normalize_text(doc_text[:5000])
    source = doc["source"]

    entity_hits = sum(1 for ent in entities if normalize_text(ent) and (normalize_text(ent) in title_norm or normalize_text(ent) in text_norm))
    query_hits = sum(1 for q in FIXED_QUERIES if normalize_text(q) in text_norm or sum(1 for tok in tokenize(q)[:4] if tok in text_norm) >= 2)

    video_themes = detect_video_themes(video_title)
    theme_hits = 0
    matched_themes = set()
    for theme in video_themes:
        hits = keyword_hits(doc_title + " " + doc_text[:4000], DOC_THEME_KEYWORDS[theme])
        if hits > 0:
            matched_themes.add(theme)
            theme_hits += hits

    generic_penalty = 0.0
    preview_text = doc_title + " " + doc_text[:1500]
    if keyword_hits(preview_text, GENERIC_INSTITUTIONAL_TERMS) >= 2:
        generic_penalty += 0.18
    if is_noise_gov_title(doc_title):
        generic_penalty += 0.45
    if any(normalize_text(t) in title_norm or normalize_text(t) in text_norm for t in FALSE_POSITIVE_TRANS_TERMS):
        generic_penalty += 0.50
    if len(tokenize(doc_text)) < 120:
        generic_penalty += 0.10
    if not looks_trans_relevant(preview_text):
        generic_penalty += 0.40

    source_boost = 0.10 if source == "planalto" else 0.06
    entity_boost = min(0.18, 0.06 * entity_hits)
    theme_boost = min(0.22, 0.05 * theme_hits)
    query_boost = min(0.10, 0.03 * query_hits)

    if video_themes and not matched_themes:
        generic_penalty += 0.20
    if video_themes and matched_themes and len(matched_themes) == 1 and "rights" in matched_themes and is_noise_gov_title(doc_title):
        generic_penalty += 0.20

    score = base_score + source_boost + entity_boost + theme_boost + query_boost - generic_penalty
    return float(score), matched_themes


# ==========================
# MERGE-SAFE LOAD / SAVE
# ==========================

def load_existing_metadata(path: str):
    if not os.path.exists(path):
        return [], set()

    rows = []
    seen_chunks = set()

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
                key = (
                    obj.get("video_id"),
                    normalize_text(obj.get("title", "")),
                    obj.get("source", ""),
                    obj.get("chunk_index", 0),
                )
                seen_chunks.add(key)
            except Exception:
                continue

    return rows, seen_chunks


def append_metadata(path: str, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rebuild_faiss_from_metadata(metadata_path: str, index_path: str):
    if not os.path.exists(metadata_path):
        return

    texts = []
    with open(metadata_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text")
                if text:
                    texts.append(text)
            except Exception:
                continue

    if not texts:
        return

    embeddings = encode_texts(texts)
    embeddings = np.array(embeddings).astype("float32")

    if NORMALIZE_EMBEDDINGS:
        index = faiss.IndexFlatIP(embeddings.shape[1])
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)


# ==========================
# PIPELINE
# ==========================

def main():
    titles_df = pd.read_csv(TITLES_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing_rows, seen_chunks = load_existing_metadata(METADATA_PATH)
    print(f"[INFO] existing metadata rows: {len(existing_rows)}")

    gov_docs = collect_gov_documents()
    if not gov_docs:
        print("[WARN] no government docs collected")
        return

    gov_docs = gov_docs[:MAX_DOCS_FROM_GOV]
    new_rows = []

    for _, row in titles_df.iterrows():
        video_id = row["video_id"]
        video_title = row["video_title"]

        if not should_query_government(video_title):
            continue

        entities = extract_entities_from_title(video_title)
        title_embedding = encode_texts([video_title])[0]

        scored = []
        for doc in gov_docs:
            score, matched_themes = score_document_for_video(video_title, entities, doc, title_embedding)
            if score >= SIMILARITY_THRESHOLD:
                scored.append((doc, score, matched_themes))

        scored.sort(key=lambda x: x[1], reverse=True)

        selected = []
        used_theme_sets = []
        for doc, score, matched_themes in scored:
            if len(selected) >= MAX_GOV_DOCS_PER_VIDEO:
                break
            normalized_title = normalize_text(doc["title"])
            if any(normalize_text(s[0]["title"]) == normalized_title for s in selected):
                continue
            if selected and matched_themes and any(matched_themes == ut for ut in used_theme_sets):
                continue
            selected.append((doc, score, matched_themes))
            used_theme_sets.append(matched_themes)

        if len(selected) < MAX_GOV_DOCS_PER_VIDEO:
            for doc, score, matched_themes in scored:
                if len(selected) >= MAX_GOV_DOCS_PER_VIDEO:
                    break
                normalized_title = normalize_text(doc["title"])
                if any(normalize_text(s[0]["title"]) == normalized_title for s in selected):
                    continue
                if score < max(SIMILARITY_THRESHOLD + 0.05, selected[0][1] - 0.08 if selected else SIMILARITY_THRESHOLD):
                    continue
                selected.append((doc, score, matched_themes))

        if not selected:
            continue

        for doc, score, _ in selected:
            chunks = chunk_text(doc["text"])
            for chunk_index, chunk in enumerate(chunks):
                entry = {
                    "video_id": video_id,
                    "title": doc["title"],
                    "source": doc["source"],
                    "url": doc["url"],
                    "chunk_index": chunk_index,
                    "text": f"{doc['title']}. {chunk}",
                    "score": float(score),
                }
                chunk_key = (video_id, normalize_text(doc["title"]), doc["source"], chunk_index)
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    new_rows.append(entry)

    append_metadata(METADATA_PATH, new_rows)
    print(f"[INFO] new gov rows appended: {len(new_rows)}")

    rebuild_faiss_from_metadata(METADATA_PATH, INDEX_PATH)
    print("[INFO] faiss rebuilt from full metadata.jsonl")


if __name__ == "__main__":
    main()
