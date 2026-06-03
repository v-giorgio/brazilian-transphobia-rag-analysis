import os
import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

METADATA_PATH = "./data/transcriptions/base/metadata.json"
INDEX_PATH = "./data/transcriptions/base/faiss_index.bin"

EMBEDDING_MODEL = "BAAI/bge-m3"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================

print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
model.eval()
torch.set_grad_enabled(False)

# =========================
# LOAD METADATA
# =========================

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

print("Loading metadata...")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

texts = [row["text"] for row in metadata if row.get("text")]

if not texts:
    raise ValueError("No texts found in metadata.")

print(f"Found {len(texts)} chunks in metadata.")

# =========================
# RE-EMBED CHUNKS
# =========================

print(f"Encoding chunks with {EMBEDDING_MODEL}...")
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
)

embeddings = np.asarray(embeddings, dtype="float32")

# =========================
# BUILD FAISS INDEX
# =========================

print("Building FAISS index...")
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings)

os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
faiss.write_index(index, INDEX_PATH)

print(f"FAISS rebuilt successfully with {index.ntotal} vectors.")
print(f"Index saved to: {INDEX_PATH}")