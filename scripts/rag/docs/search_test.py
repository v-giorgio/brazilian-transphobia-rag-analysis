import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# =========================
# CONFIG
# =========================

INDEX_PATH = "./data/base/faiss_index.bin"
METADATA_PATH = "./data/base/metadata.json"
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
DEVICE = "cpu"

# =========================
# LOAD
# =========================

print("Carregando modelo...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
model.eval()
torch.set_grad_enabled(False)

print("Carregando índice FAISS...")
index = faiss.read_index(INDEX_PATH)

print("Carregando metadados...")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# =========================
# SEARCH
# =========================

def search(query, top_k=5):
    """
    Realiza busca vetorial usando cosine similarity
    (IndexFlatIP + embeddings normalizadas)
    """

    # Prefixo obrigatório para E5-instruct
    formatted_query = f"query: {query}"

    query_emb = model.encode(
        [formatted_query],
        normalize_embeddings=True
    )

    query_emb = np.array(query_emb).astype("float32")

    scores, indices = index.search(query_emb, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):

        if idx == -1:
            continue

        chunk_meta = metadata[idx]

        results.append({
            "score": float(score),
            "chunk_id": chunk_meta["chunk_id"],
            "video_id": chunk_meta["video_id"],
            "chunk_index": chunk_meta["chunk_index"],
            "token_count": chunk_meta["token_count"],
            "text": chunk_meta["text"]
        })

    return results


# =========================
# EXEMPLO
# =========================

if __name__ == "__main__":
    query = "Nossa essa gente é doente, tem que internar mesmo"

    results = search(query, top_k=5)

    for r in results:
        print("\n" + "="*50)
        print(f"Score: {r['score']:.4f}")
        print(f"Video ID: {r['video_id']}")
        print(f"Chunk Index: {r['chunk_index']}")
        print(f"Tokens: {r['token_count']}")
        print("Texto:")
        print(r["text"])