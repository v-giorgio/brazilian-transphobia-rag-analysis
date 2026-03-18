import pandas as pd
import numpy as np
import faiss
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json
import os
import torch

# =========================
# CONFIG
# =========================

CSV_PATH = "./data/transcriptions/video_transcriptions_filtered.csv"
INDEX_PATH = "./data/base/faiss_index.bin"
METADATA_PATH = "./data/base/metadata.json"

# https://huggingface.co/intfloat/multilingual-e5-large-instruct
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"

SIM_THRESHOLD = 0.38
MAX_TOKENS = 350
MIN_CHUNK_TOKENS = 80
BATCH_SIZE = 16

DEVICE = "cpu"  # CPU é estável para indexação offline

# =========================
# LOAD MODELS
# =========================

print("Carregando modelo de embeddings...")
model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
model.eval()
torch.set_grad_enabled(False)

print("Carregando spaCy...")
nlp = spacy.load("pt_core_news_sm")


# =========================
# FUNÇÕES AUXILIARES
# =========================

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]


def count_tokens(text):
    return len(model.tokenizer(text)["input_ids"])


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_semantic_chunks(sentences, sentence_embeddings):
    chunks = []
    current_chunk = []
    current_embeddings = []

    for i, sentence in enumerate(sentences):
        emb = sentence_embeddings[i]

        if not current_chunk:
            current_chunk.append(sentence)
            current_embeddings.append(emb)
            continue

        mean_emb = np.mean(current_embeddings, axis=0)
        sim = cosine_sim(mean_emb, emb)

        current_text = " ".join(current_chunk)
        token_count = count_tokens(current_text)

        if sim >= SIM_THRESHOLD and token_count < MAX_TOKENS:
            current_chunk.append(sentence)
            current_embeddings.append(emb)
        else:
            final_text = " ".join(current_chunk)
            if count_tokens(final_text) >= MIN_CHUNK_TOKENS:
                chunks.append(final_text)

            current_chunk = [sentence]
            current_embeddings = [emb]

    # Último chunk
    if current_chunk:
        final_text = " ".join(current_chunk)
        if count_tokens(final_text) >= MIN_CHUNK_TOKENS:
            chunks.append(final_text)

    return chunks


# =========================
# PIPELINE
# =========================

print("Lendo CSV...")
df = pd.read_csv(CSV_PATH)

all_chunk_embeddings = []
metadata = []

chunk_global_id = 0

for _, row in tqdm(df.iterrows(), total=len(df)):

    video_id = row["video_id"]
    transcription = row["transcription"]

    if not isinstance(transcription, str) or len(transcription.strip()) == 0:
        continue

    sentences = split_sentences(transcription)

    if len(sentences) == 0:
        continue

    # Prefixo obrigatório para E5-instruct
    sentences_prefixed = [f"passage: {s}" for s in sentences]

    sentence_embeddings = model.encode(
        sentences_prefixed,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=False
    )

    chunks = build_semantic_chunks(sentences, sentence_embeddings)

    if len(chunks) == 0:
        continue

    chunks_prefixed = [f"passage: {c}" for c in chunks]

    chunk_embeddings = model.encode(
        chunks_prefixed,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=False
    )

    for i, chunk in enumerate(chunks):

        all_chunk_embeddings.append(chunk_embeddings[i])

        metadata.append({
            "chunk_id": chunk_global_id,
            "video_id": video_id,
            "chunk_index": i,
            "text": chunk,
            "token_count": count_tokens(chunk),
            "char_count": len(chunk)
        })

        chunk_global_id += 1


# =========================
# FAISS INDEX
# =========================

print("Criando índice FAISS...")

embedding_dim = len(all_chunk_embeddings[0])
index = faiss.IndexFlatIP(embedding_dim)

emb_array = np.array(all_chunk_embeddings).astype("float32")
index.add(emb_array)

print(f"{index.ntotal} vetores indexados.")

os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

faiss.write_index(index, INDEX_PATH)

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Indexação concluída com sucesso.")