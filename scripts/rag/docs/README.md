# Document Retrieval

## pt-br

Scripts para construir a base auxiliar usada no RAG: coleta em ANTRA, gov.br e Wikipédia, processamento de transcrições, chunking e criação de índices FAISS.

### Coleta de documentos

```bash
python scripts/rag/docs/1_wiki_get_antra.py
python scripts/rag/docs/2_wiki_get_gov.py
python scripts/rag/docs/3_wiki_get_wikipedia.py
```

### Transcrições e chunking

```bash
python scripts/rag/docs/utils/transcript.py
python scripts/rag/docs/4_transcript-chunking.py
```

### Índices FAISS

```bash
python scripts/rag/docs/5_split_and_build_faiss_indices.py
python scripts/rag/docs/rebuild_transcriptions_faiss_bge_m3.py
```

### Saídas

- `scripts/rag/docs/data/wiki/`: documentos e metadados.
- `scripts/rag/docs/data/retrieval_split/`: índices de entidades e explicações.
- `scripts/rag/docs/transcriptions/`: transcrições e índice de transcrições.

## en

Scripts for building the auxiliary RAG knowledge base: ANTRA, gov.br, and Wikipedia collection, transcript processing, chunking, and FAISS index creation.

### Document collection

```bash
python scripts/rag/docs/1_wiki_get_antra.py
python scripts/rag/docs/2_wiki_get_gov.py
python scripts/rag/docs/3_wiki_get_wikipedia.py
```

### Transcripts and chunking

```bash
python scripts/rag/docs/utils/transcript.py
python scripts/rag/docs/4_transcript-chunking.py
```

### FAISS indices

```bash
python scripts/rag/docs/5_split_and_build_faiss_indices.py
python scripts/rag/docs/rebuild_transcriptions_faiss_bge_m3.py
```

### Outputs

- `scripts/rag/docs/data/wiki/`: documents and metadata.
- `scripts/rag/docs/data/retrieval_split/`: entity and explanation indices.
- `scripts/rag/docs/transcriptions/`: transcripts and transcript index.
