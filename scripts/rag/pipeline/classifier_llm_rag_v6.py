import os
from pathlib import Path
import re
import json
import time
import ast
import argparse
from typing import Any, Dict, List, Set, Tuple, Optional

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

DEFAULT_INPUT_PATH = "./data/temp_dataset/dataset_unificado.csv"
DEFAULT_OUTPUT_DIR = "./data/temp_dataset/output"

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3:8b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_TRANSCRIPT_INDEX = "../docs/data/transcriptions/base/faiss_index.bin"
DEFAULT_TRANSCRIPT_METADATA = "../docs/data/transcriptions/base/metadata.json"

DEFAULT_WIKI_INDEX = "../docs/data/wiki/faiss_index.bin"
DEFAULT_WIKI_METADATA = "../docs/data/wiki/metadata.jsonl"

ALLOWED_SPEECH = {"HATEFUL", "SUPPORTIVE", "NEUTRAL"}
ALLOWED_OFFENSIVE = {"OFFENSIVE", "NON_OFFENSIVE"}
ALLOWED_ASPECTS = {
    "RELIGION",
    "POLITICS",
    "GENDER_DENIAL",
    "HARRASSMENT",
    "THREATENING",
    "PSEUDOSCIENCE",
}

GENERIC_WIKI_TITLES = {
    "transexualidade",
    "transgénero",
    "transgenero",
    "homem trans",
    "mulher trans",
    "travesti (identidade de gênero)",
    "travesti",
    "identidade de gênero",
    "pessoa trans",
    "transfobia",
    "disforia de gênero",
}

STOP_TERMS = {
    "isso", "essa", "esse", "tambem", "porque", "muito", "mais", "para", "com", "uma",
    "uns", "umas", "são", "ser", "tem", "temos", "sobre", "como", "qual", "quais",
    "depois", "antes", "mesmo", "ainda", "tudo", "todos", "todas", "dele", "dela",
}

SYSTEM_PROMPT = '''Você é uma assistente de anotação de comentários em português, especializada em classificação de discurso sobre pessoas trans.

Sua tarefa é classificar cada comentário usando APENAS os rótulos permitidos e retornar JSON válido.

Você deve produzir:
1. speech_class: uma entre HATEFUL, SUPPORTIVE, NEUTRAL
2. offensive_class: uma entre OFFENSIVE, NON_OFFENSIVE
3. aspects: zero ou mais rótulos dentre:
   RELIGION, POLITICS, GENDER_DENIAL, HARRASSMENT, THREATENING, PSEUDOSCIENCE
4. justification: justificativa curta em português
5. counterspeech: um contra-discurso curto em português

Definições:
- HATEFUL: comentário direcionado a pessoas trans ou identidades trans com teor discriminatório, excludente, desumanizante, de negação de identidade de gênero, ou de apoio a restrições, exclusão ou limitação de direitos.
- SUPPORTIVE: comentário direcionado a pessoas trans ou identidades trans com teor de apoio, defesa, acolhimento, empatia ou afirmação de direitos e identidades.
- NEUTRAL: comentário sem apoio ou hostilidade claros, apenas descritivo, ambíguo, irrelevante ou factual.
- OFFENSIVE: contém palavras explícitas de insulto, humilhação, agressividade verbal, xingamento ou linguagem degradante.
- NON_OFFENSIVE: não contém ofensa explícita.
- GENDER_DENIAL: o comentário explicitamente contém negação, invalidação ou ridicularização da identidade de gênero.
- RELIGION: o comentário explicitamente contém uso de religião, pecado, Deus, igreja, diabo, missão divina etc.
- POLITICS: o comentário explicitamente contém referência a política, governo, eleições, direita/esquerda, parlamentares, ideologia, legislação.
- HARRASSMENT: o comentário explicitamente contém ataque pessoal, zombaria, humilhação, bullying ou assédio verbal dirigido.
- THREATENING: o comentário explicitamente contém ameaça explícita ou implícita, intimidação, incentivo à violência ou desejo de dano.
- PSEUDOSCIENCE: o comentário explicitamente contém uso de falsa ciência, biologicismo raso, psicologização indevida ou pseudoexplicações para invalidar pessoas trans.

Regras:
- Use o título do vídeo apenas como contexto auxiliar, mas classifique o comentário em si.
- Use o contexto recuperado apenas como apoio adicional.
- Não invente rótulos fora da lista.
- Se nenhum aspecto se aplicar claramente, retorne lista vazia.
- A justificativa deve explicar por que os rótulos escolhidos fazem sentido.
- O counterspeech deve responder de forma construtiva e respeitosa.
- Mesmo para comentários NEUTRAL, forneça um counterspeech breve que estimule diálogo respeitoso.
- Retorne APENAS JSON válido, sem texto extra.
'''

USER_TEMPLATE = '''Classifique o comentário abaixo em esquema few-shot com RAG.

A seguir há exemplos anotados. Use-os apenas como referência de decisão e formato.

{few_shot_examples}

Agora veja o contexto recuperado.
Use esse contexto apenas como apoio. Se o contexto não ajudar, baseie-se no comentário e no título do vídeo.

Contexto recuperado:
{retrieved_context}

Agora classifique o comentário alvo.

Título do vídeo:
{video_title}

Comentário:
{comment}

Responda APENAS com JSON neste formato:
{{
  "speech_class": "HATEFUL | SUPPORTIVE | NEUTRAL",
  "offensive_class": "OFFENSIVE | NON_OFFENSIVE",
  "aspects": ["RELIGION", "POLITICS", "GENDER_DENIAL", "HARRASSMENT", "THREATENING", "PSEUDOSCIENCE"],
  "justification": "texto curto em português",
  "counterspeech": "texto curto em português"
}}
'''


def ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_aspects(aspects: Any) -> List[str]:
    if aspects is None:
        return []
    if isinstance(aspects, list):
        items = aspects
    elif isinstance(aspects, str):
        try:
            parsed = ast.literal_eval(aspects)
            items = parsed if isinstance(parsed, list) else [aspects]
        except Exception:
            items = [aspects]
    else:
        return []

    clean = []
    seen = set()
    for a in items:
        a = str(a).strip().upper()
        if a in ALLOWED_ASPECTS and a not in seen:
            clean.append(a)
            seen.add(a)
    return clean


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        snippet = match.group(0)
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Não foi possível extrair um JSON válido da saída do modelo.")


def validate_prediction(obj: Dict[str, Any]) -> Dict[str, Any]:
    speech_class = str(obj.get("speech_class", "")).strip().upper()
    offensive_class = str(obj.get("offensive_class", "")).strip().upper()
    aspects = normalize_aspects(obj.get("aspects", []))
    justification = str(obj.get("justification", "")).strip()
    counterspeech = str(obj.get("counterspeech", "")).strip()

    if speech_class not in ALLOWED_SPEECH:
        raise ValueError(f"speech_class inválida: {speech_class}")
    if offensive_class not in ALLOWED_OFFENSIVE:
        raise ValueError(f"offensive_class inválida: {offensive_class}")
    if not justification:
        raise ValueError("justification vazia")
    if not counterspeech:
        raise ValueError("counterspeech vazio")

    return {
        "speech_class": speech_class,
        "offensive_class": offensive_class,
        "aspects": aspects,
        "justification": justification,
        "counterspeech": counterspeech,
    }


def load_processed_ids(output_jsonl: str, id_col: str) -> set:
    processed = set()
    if not os.path.exists(output_jsonl):
        return processed

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if id_col in row:
                    processed.add(str(row[id_col]))
            except Exception:
                continue
    return processed


def create_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def build_example_text(row: pd.Series) -> str:
    aspects = normalize_aspects(row["aspects"])
    aspects_json = json.dumps(aspects, ensure_ascii=False)

    return f'''Exemplo
Título do vídeo:
{row["video_title"]}

Comentário:
{row["comment"]}

Resposta:
{{
  "speech_class": "{str(row["speech_class"]).strip().upper()}",
  "offensive_class": "{str(row["offensive_class"]).strip().upper()}",
  "aspects": {aspects_json},
  "justification": "Exemplo de referência; não reutilize este texto literalmente na resposta final.",
  "counterspeech": "Exemplo de referência; produza um counterspeech original para o comentário alvo."
}}'''


def select_examples(
    df: pd.DataFrame,
    target_id: str,
    id_col: str,
    speech_per_class: int,
    offensive_per_class: int,
    aspect_per_label: int,
) -> pd.DataFrame:
    pool = df[df[id_col].astype(str) != str(target_id)].copy()

    selected_ids: Set[str] = set()
    selected_rows: List[pd.Series] = []

    for label in ["HATEFUL", "SUPPORTIVE", "NEUTRAL"]:
        subset = pool[pool["speech_class"].astype(str).str.upper() == label]
        for _, row in subset.head(speech_per_class).iterrows():
            rid = str(row[id_col])
            if rid not in selected_ids:
                selected_ids.add(rid)
                selected_rows.append(row)

    for label in ["OFFENSIVE", "NON_OFFENSIVE"]:
        subset = pool[pool["offensive_class"].astype(str).str.upper() == label]
        for _, row in subset.head(offensive_per_class).iterrows():
            rid = str(row[id_col])
            if rid not in selected_ids:
                selected_ids.add(rid)
                selected_rows.append(row)

    for label in ["RELIGION", "POLITICS", "GENDER_DENIAL", "HARRASSMENT", "THREATENING", "PSEUDOSCIENCE"]:
        subset = pool[pool["aspects_list"].apply(lambda xs: label in xs)]
        for _, row in subset.head(aspect_per_label).iterrows():
            rid = str(row[id_col])
            if rid not in selected_ids:
                selected_ids.add(rid)
                selected_rows.append(row)

    if not selected_rows:
        return pool.head(0).copy()

    examples_df = pd.DataFrame(selected_rows)
    return examples_df.sort_values(by=id_col).copy()


def render_few_shot_examples(examples_df: pd.DataFrame) -> str:
    return "\n\n".join(build_example_text(row) for _, row in examples_df.iterrows())


def load_metadata(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata não encontrado: {path}")

    rows: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                rows = data
            else:
                raise ValueError("Formato de metadata não suportado.")
    return rows


def tokenize_portuguese(text: str) -> List[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", str(text).lower())


def extract_title_entity(video_title: str) -> str:
    title = str(video_title).strip()
    title = re.sub(r'["“”\']', "", title)

    title = re.sub(
        r"^(pré-candidato|candidato|deputado|deputada|senador|senadora|vereador|vereadora|governador|governadora|presidente)\s*,?\s*",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()

    m = re.search(
        r"\b([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+(?:\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+){0,2})\b\s+(?:diz|disse|fala|afirma|afirmou|lamenta|lamentou|discute|criticou|critica|rebate|comenta|declara|defende|ataca)\b",
        title,
    )
    if m:
        return m.group(1).strip()

    sequences = re.findall(
        r"\b[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+(?:\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+){0,2}\b",
        title,
    )
    blacklist = {
        "Pré", "Candidato", "Direita", "Esquerda", "Brasil", "Washington", "Rio", "Janeiro",
        "Mulher", "Homem", "Pessoas", "Trans", "Transgênero", "Transexualidade", "Política",
    }
    for seq in sequences:
        first = seq.split()[0]
        if first not in blacklist:
            return seq.strip()

    return ""


def comment_focus_terms(comment: str, max_terms: int = 6) -> List[str]:
    toks = tokenize_portuguese(comment)
    terms = []
    for t in toks:
        if len(t) < 4:
            continue
        if t in STOP_TERMS:
            continue
        terms.append(t)

    seen = set()
    out = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_terms:
            break
    return out


def lexical_overlap_score(text: str, query_terms: List[str]) -> float:
    if not query_terms:
        return 0.0
    toks = set(tokenize_portuguese(text))
    hits = sum(1 for t in query_terms if t in toks)
    return hits / len(query_terms)


def normalize_scores(items: List[Dict[str, Any]], field: str, out_field: str) -> None:
    vals = [float(x.get(field, 0.0)) for x in items]
    if not vals:
        return
    vmin = min(vals)
    vmax = max(vals)
    if abs(vmax - vmin) < 1e-9:
        for x in items:
            x[out_field] = 1.0 if vmax > 0 else 0.0
        return
    for x in items:
        x[out_field] = (float(x.get(field, 0.0)) - vmin) / (vmax - vmin)


def metadata_to_document(item: Dict[str, Any]) -> Document:
    page_content = " ".join(
        x for x in [
            str(item.get("title", "")),
            str(item.get("text", "")),
            str(item.get("video_title", "")),
            str(item.get("source", "")),
        ] if x
    ).strip()
    metadata = dict(item)
    metadata["_doc_text"] = page_content
    return Document(page_content=page_content, metadata=metadata)


class FaissRetriever:
    def __init__(self, index_path: str, metadata_path: str, embedder: SentenceTransformer, name: str = ""):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.name = name or Path(index_path).stem
        self.index = faiss.read_index(index_path)
        self.metadata = load_metadata(metadata_path)
        self.embedder = embedder

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Inconsistência entre índice e metadata em {self.name}: "
                f"index.ntotal={self.index.ntotal}, metadata={len(self.metadata)}"
            )

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        distances, indices = self.index.search(q_emb, top_k * 8)

        results = []
        for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            item["_faiss_score"] = float(score)
            item["_semantic_score"] = float(score)
            item["_retriever"] = self.name
            results.append(item)

        return results


class LangChainBM25Retriever:
    def __init__(self, metadata: List[Dict[str, Any]], name: str = "", k_default: int = 20):
        self.metadata = metadata
        self.name = name
        self.documents = [metadata_to_document(item) for item in metadata]
        self.retriever = BM25Retriever.from_documents(self.documents)
        self.k_default = k_default

    def search(self, query_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        self.retriever.k = top_k
        docs = self.retriever.invoke(query_text)
        results = []
        for rank, doc in enumerate(docs):
            item = dict(doc.metadata)
            item["_retriever"] = self.name
            item["_bm25_rank_score"] = float(max(top_k - rank, 1))
            results.append(item)
        return results


class TfidfMetadataRetriever:
    def __init__(self, metadata: List[Dict[str, Any]], name: str = ""):
        self.metadata = metadata
        self.name = name
        self.docs = [
            " ".join(
                x for x in [
                    str(item.get("title", "")),
                    str(item.get("text", "")),
                    str(item.get("video_title", "")),
                    str(item.get("source", "")),
                ] if x
            ).strip()
            for item in metadata
        ]
        self.tokenized_docs = [tokenize_portuguese(doc) for doc in self.docs]
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.docs)

    def search(self, query_text: str, query_terms: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        tfidf_query = self.tfidf.transform([query_text])
        tfidf_scores = linear_kernel(tfidf_query, self.tfidf_matrix).ravel()

        lexical_counts = np.array([
            sum(1 for t in query_terms if t and t in doc_toks)
            for doc_toks in self.tokenized_docs
        ], dtype=float)

        candidate_indices = np.argsort(-(tfidf_scores + lexical_counts))[: max(top_k * 6, 20)]

        results = []
        for idx in candidate_indices:
            item = dict(self.metadata[int(idx)])
            item["_retriever"] = self.name
            item["_tfidf_score"] = float(tfidf_scores[int(idx)])
            item["_lexical_count"] = float(lexical_counts[int(idx)])
            results.append(item)
        return results


def dedup_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dedup = []
    seen = set()
    for item in results:
        key = (
            str(item.get("title", "")).strip().lower(),
            str(item.get("text", "")).strip()[:300].lower(),
            str(item.get("_retriever", "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


def rerank_transcript_items(
    items: List[Dict[str, Any]],
    video_title: str,
    comment: str,
    entity: str,
    top_k: int,
    max_per_group: int = 1,
) -> List[Dict[str, Any]]:
    q_terms = comment_focus_terms(comment)

    normalize_scores(items, "_semantic_score", "_semantic_norm")
    normalize_scores(items, "_bm25_rank_score", "_bm25_norm")
    normalize_scores(items, "_tfidf_score", "_tfidf_norm")
    normalize_scores(items, "_lexical_count", "_lexical_norm")

    ranked = []
    for item in items:
        text = str(item.get("text", ""))
        score = 0.0
        score += 0.75 * float(item.get("_semantic_norm", 0.0))
        score += 1.00 * float(item.get("_bm25_norm", 0.0))
        score += 0.55 * float(item.get("_tfidf_norm", 0.0))
        score += 0.45 * float(item.get("_lexical_norm", 0.0))
        score += 0.35 * lexical_overlap_score(text, q_terms)

        if entity and entity.lower() in text.lower():
            score += 0.85

        item["_final_score"] = score
        item["_group_key"] = " ".join(tokenize_portuguese(text)[:20])
        ranked.append(item)

    ranked.sort(key=lambda x: x["_final_score"], reverse=True)

    out = []
    group_counts: Dict[str, int] = {}
    for item in ranked:
        gk = item["_group_key"]
        if group_counts.get(gk, 0) >= max_per_group:
            continue
        group_counts[gk] = group_counts.get(gk, 0) + 1
        out.append(item)
        if len(out) >= top_k:
            break
    return out


def rerank_wiki_items(
    items: List[Dict[str, Any]],
    video_title: str,
    comment: str,
    entity: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    q_terms = comment_focus_terms(comment)
    title_terms = tokenize_portuguese(video_title)

    normalize_scores(items, "_semantic_score", "_semantic_norm")
    normalize_scores(items, "_bm25_rank_score", "_bm25_norm")
    normalize_scores(items, "_tfidf_score", "_tfidf_norm")
    normalize_scores(items, "_lexical_count", "_lexical_norm")

    ranked = []
    for item in items:
        title = str(item.get("title", "")).strip()
        text = str(item.get("text", "")).strip()
        score = 0.0
        score += 0.60 * float(item.get("_semantic_norm", 0.0))
        score += 1.15 * float(item.get("_bm25_norm", 0.0))
        score += 0.75 * float(item.get("_tfidf_norm", 0.0))
        score += 0.60 * float(item.get("_lexical_norm", 0.0))
        score += 0.35 * lexical_overlap_score(text, q_terms)
        score += 0.20 * lexical_overlap_score(title, title_terms[:8])

        if entity and entity.lower() in title.lower():
            score += 1.60
        elif entity and entity.lower() in text.lower():
            score += 0.60

        if title.lower() in GENERIC_WIKI_TITLES:
            score -= 0.60

        item["_final_score"] = score
        ranked.append(item)

    ranked.sort(key=lambda x: x["_final_score"], reverse=True)

    out = []
    seen_titles = set()
    for item in ranked:
        title_key = str(item.get("title", "")).strip().lower()
        if title_key and title_key in seen_titles:
            continue
        if title_key:
            seen_titles.add(title_key)
        out.append(item)
        if len(out) >= top_k:
            break
    return out


def build_retrieval_queries(video_title: str, comment: str) -> Dict[str, str]:
    entity = extract_title_entity(video_title)
    terms = comment_focus_terms(comment, max_terms=6)

    comment_focus = comment
    entity_comment = f"{entity} {' '.join(terms)} {comment}".strip() if entity else comment
    title_comment = f"Título do vídeo: {video_title}\nComentário: {comment}"

    return {
        "entity": entity,
        "comment_focus": comment_focus,
        "entity_comment": entity_comment,
        "title_comment": title_comment,
    }


def safe_get_text(item: Dict[str, Any]) -> str:
    text = str(item.get("text", "")).strip()
    title = str(item.get("title", "")).strip()
    source = str(item.get("source", item.get("_retriever", ""))).strip()

    prefix = []
    if source:
        prefix.append(f"Fonte: {source}")
    if title:
        prefix.append(f"Título: {title}")
    prefix_text = " | ".join(prefix)
    return f"{prefix_text}\n{text}" if prefix_text else text


def retrieve_context(
    video_title: str,
    comment: str,
    mode: str,
    top_k_transcripts: int,
    top_k_wiki: int,
    transcripts_retriever: Optional[FaissRetriever],
    wiki_retriever: Optional[FaissRetriever],
    transcripts_bm25: Optional[LangChainBM25Retriever],
    wiki_bm25: Optional[LangChainBM25Retriever],
    transcripts_tfidf: Optional[TfidfMetadataRetriever],
    wiki_tfidf: Optional[TfidfMetadataRetriever],
) -> Tuple[str, List[Dict[str, Any]], Dict[str, str]]:
    queries = build_retrieval_queries(video_title, comment)
    entity = queries["entity"]
    focus_terms = comment_focus_terms(comment, max_terms=6)

    results: List[Dict[str, Any]] = []

    if mode in {"transcripts", "both"}:
        if transcripts_retriever is None or transcripts_bm25 is None or transcripts_tfidf is None:
            raise ValueError("Retriever de transcrições não foi carregado.")

        transcript_candidates = []
        transcript_candidates.extend(transcripts_retriever.search(queries["comment_focus"], top_k=top_k_transcripts))
        transcript_candidates.extend(transcripts_retriever.search(queries["entity_comment"], top_k=top_k_transcripts))
        transcript_candidates.extend(transcripts_retriever.search(queries["title_comment"], top_k=top_k_transcripts))

        transcript_candidates.extend(
            transcripts_bm25.search(
                query_text=queries["title_comment"],
                top_k=max(top_k_transcripts * 5, 20),
            )
        )

        lexical_terms = []
        if entity:
            lexical_terms.append(entity.lower())
        lexical_terms.extend(focus_terms)
        transcript_candidates.extend(
            transcripts_tfidf.search(
                query_text=queries["title_comment"],
                query_terms=lexical_terms,
                top_k=max(top_k_transcripts * 5, 20),
            )
        )

        transcript_candidates = dedup_results(transcript_candidates)
        transcript_candidates = rerank_transcript_items(
            transcript_candidates,
            video_title=video_title,
            comment=comment,
            entity=entity,
            top_k=top_k_transcripts,
        )
        results.extend(transcript_candidates)

    if mode in {"wiki", "both"}:
        if wiki_retriever is None or wiki_bm25 is None or wiki_tfidf is None:
            raise ValueError("Retriever de documentos wiki não foi carregado.")

        wiki_candidates = []
        wiki_candidates.extend(wiki_retriever.search(queries["entity_comment"], top_k=top_k_wiki))
        wiki_candidates.extend(wiki_retriever.search(queries["comment_focus"], top_k=top_k_wiki))
        wiki_candidates.extend(wiki_retriever.search(queries["title_comment"], top_k=top_k_wiki))

        wiki_candidates.extend(
            wiki_bm25.search(
                query_text=queries["title_comment"],
                top_k=max(top_k_wiki * 6, 24),
            )
        )

        lexical_terms = []
        if entity:
            lexical_terms.append(entity.lower())
        lexical_terms.extend(focus_terms)
        wiki_candidates.extend(
            wiki_tfidf.search(
                query_text=queries["title_comment"],
                query_terms=lexical_terms,
                top_k=max(top_k_wiki * 6, 24),
            )
        )

        wiki_candidates = dedup_results(wiki_candidates)
        wiki_candidates = rerank_wiki_items(
            wiki_candidates,
            video_title=video_title,
            comment=comment,
            entity=entity,
            top_k=top_k_wiki,
        )
        results.extend(wiki_candidates)

    results = dedup_results(results)
    results.sort(key=lambda x: x.get("_final_score", x.get("_faiss_score", 0.0)), reverse=True)

    context_blocks = []
    for i, item in enumerate(results, start=1):
        context_blocks.append(f"[Contexto {i}]\n{safe_get_text(item)}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "Nenhum contexto recuperado."
    return context_text, results, queries


def classify_comment(
    client: OpenAI,
    model: str,
    video_title: str,
    comment: str,
    few_shot_examples: str,
    retrieved_context: str,
    temperature: float,
    max_retries: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    user_prompt = USER_TEMPLATE.format(
        video_title=video_title,
        comment=comment,
        few_shot_examples=few_shot_examples,
        retrieved_context=retrieved_context,
    )
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            text = response.choices[0].message.content or ""
            parsed = extract_json_object(text)
            return validate_prediction(parsed)

        except Exception as e:
            last_error = e
            wait = sleep_seconds * (2 ** attempt)
            time.sleep(wait)

    raise RuntimeError(f"Falha após {max_retries} tentativas: {last_error}")


def default_output_paths(output_dir: str, rag_mode: str) -> Tuple[str, str]:
    base = os.path.join(output_dir, f"classifier_llm_rag_{rag_mode}")
    return f"{base}_predictions.csv", f"{base}_predictions.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Classificador few-shot + RAG híbrido (FAISS + LangChain BM25 + TF-IDF).")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Caminho do CSV de entrada")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Diretório base de saída")
    parser.add_argument("--output-csv", default=None, help="Caminho do CSV de saída")
    parser.add_argument("--output-jsonl", default=None, help="Caminho do JSONL incremental")
    parser.add_argument("--rag-mode", choices=["transcripts", "wiki", "both"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo Ollama/OpenAI-compatible")
    parser.add_argument("--api-key", default="ollama", help="Chave de API; para Ollama pode ser qualquer string")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL da API compatível com OpenAI")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Modelo de embeddings")
    parser.add_argument("--transcript-index", default=DEFAULT_TRANSCRIPT_INDEX)
    parser.add_argument("--transcript-metadata", default=DEFAULT_TRANSCRIPT_METADATA)
    parser.add_argument("--wiki-index", default=DEFAULT_WIKI_INDEX)
    parser.add_argument("--wiki-metadata", default=DEFAULT_WIKI_METADATA)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--id-col", default="dataset_comment_id")
    parser.add_argument("--speech-examples-per-class", type=int, default=10)
    parser.add_argument("--offensive-examples-per-class", type=int, default=10)
    parser.add_argument("--aspect-examples-per-label", type=int, default=5)
    parser.add_argument("--top-k-transcripts", type=int, default=4)
    parser.add_argument("--top-k-wiki", type=int, default=3)
    args = parser.parse_args()

    output_csv = args.output_csv
    output_jsonl = args.output_jsonl
    if output_csv is None or output_jsonl is None:
        default_csv, default_jsonl = default_output_paths(args.output_dir, args.rag_mode)
        output_csv = output_csv or default_csv
        output_jsonl = output_jsonl or default_jsonl

    ensure_dir(output_csv)
    ensure_dir(output_jsonl)

    df = pd.read_csv(args.input)

    if args.id_col not in df.columns:
        raise ValueError(f"Coluna '{args.id_col}' não encontrada.")
    if "video_title" not in df.columns or "comment" not in df.columns:
        raise ValueError("O CSV precisa conter as colunas 'video_title' e 'comment'.")
    for col in ["speech_class", "offensive_class", "aspects"]:
        if col not in df.columns:
            raise ValueError(f"O CSV precisa conter a coluna '{col}' para montar os exemplos few-shot.")

    df["speech_class"] = df["speech_class"].astype(str).str.strip().str.upper()
    df["offensive_class"] = df["offensive_class"].astype(str).str.strip().str.upper()
    df["aspects_list"] = df["aspects"].apply(normalize_aspects)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    processed_ids = load_processed_ids(output_jsonl, args.id_col)
    if processed_ids:
        df = df[~df[args.id_col].astype(str).isin(processed_ids)].copy()

    client = create_client(args.api_key, args.base_url)
    embedder = SentenceTransformer(args.embedding_model)

    transcripts_retriever = None
    wiki_retriever = None
    transcripts_bm25 = None
    wiki_bm25 = None
    transcripts_tfidf = None
    wiki_tfidf = None

    if args.rag_mode in {"transcripts", "both"}:
        transcripts_retriever = FaissRetriever(
            index_path=args.transcript_index,
            metadata_path=args.transcript_metadata,
            embedder=embedder,
            name="transcripts",
        )
        transcripts_bm25 = LangChainBM25Retriever(
            metadata=transcripts_retriever.metadata,
            name="transcripts",
        )
        transcripts_tfidf = TfidfMetadataRetriever(
            metadata=transcripts_retriever.metadata,
            name="transcripts",
        )

    if args.rag_mode in {"wiki", "both"}:
        wiki_retriever = FaissRetriever(
            index_path=args.wiki_index,
            metadata_path=args.wiki_metadata,
            embedder=embedder,
            name="wiki",
        )
        wiki_bm25 = LangChainBM25Retriever(
            metadata=wiki_retriever.metadata,
            name="wiki",
        )
        wiki_tfidf = TfidfMetadataRetriever(
            metadata=wiki_retriever.metadata,
            name="wiki",
        )

    rows_for_csv = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classificando"):
        comment_id = str(row[args.id_col])
        video_title = str(row["video_title"])
        comment = str(row["comment"])

        try:
            examples_df = select_examples(
                df=df,
                target_id=comment_id,
                id_col=args.id_col,
                speech_per_class=args.speech_examples_per_class,
                offensive_per_class=args.offensive_examples_per_class,
                aspect_per_label=args.aspect_examples_per_label,
            )
            few_shot_examples = render_few_shot_examples(examples_df)

            retrieved_context, retrieved_items, used_queries = retrieve_context(
                video_title=video_title,
                comment=comment,
                mode=args.rag_mode,
                top_k_transcripts=args.top_k_transcripts,
                top_k_wiki=args.top_k_wiki,
                transcripts_retriever=transcripts_retriever,
                wiki_retriever=wiki_retriever,
                transcripts_bm25=transcripts_bm25,
                wiki_bm25=wiki_bm25,
                transcripts_tfidf=transcripts_tfidf,
                wiki_tfidf=wiki_tfidf,
            )

            pred = classify_comment(
                client=client,
                model=args.model,
                video_title=video_title,
                comment=comment,
                few_shot_examples=few_shot_examples,
                retrieved_context=retrieved_context,
                temperature=args.temperature,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            )

            output_row = row.to_dict()
            output_row.pop("aspects_list", None)
            output_row.update({
                "pred_speech_class": pred["speech_class"],
                "pred_offensive_class": pred["offensive_class"],
                "pred_aspects": json.dumps(pred["aspects"], ensure_ascii=False),
                "pred_justification": pred["justification"],
                "pred_counterspeech": pred["counterspeech"],
                "prompt_strategy": f"rag_{args.rag_mode}",
                "model_name": args.model,
                "num_few_shot_examples": int(len(examples_df)),
                "rag_mode": args.rag_mode,
                "retrieval_query": json.dumps(used_queries, ensure_ascii=False),
                "retrieved_context": retrieved_context,
                "retrieved_titles": json.dumps([str(x.get("title", "")) for x in retrieved_items], ensure_ascii=False),
                "retrieved_sources": json.dumps([str(x.get("_retriever", "")) for x in retrieved_items], ensure_ascii=False),
                "num_retrieved_items": int(len(retrieved_items)),
                "error": None,
            })

        except Exception as e:
            output_row = row.to_dict()
            output_row.pop("aspects_list", None)
            output_row.update({
                "pred_speech_class": None,
                "pred_offensive_class": None,
                "pred_aspects": json.dumps([], ensure_ascii=False),
                "pred_justification": None,
                "pred_counterspeech": None,
                "prompt_strategy": f"rag_{args.rag_mode}",
                "model_name": args.model,
                "num_few_shot_examples": None,
                "rag_mode": args.rag_mode,
                "retrieval_query": None,
                "retrieved_context": None,
                "retrieved_titles": json.dumps([], ensure_ascii=False),
                "retrieved_sources": json.dumps([], ensure_ascii=False),
                "num_retrieved_items": 0,
                "error": str(e),
            })

        rows_for_csv.append(output_row)

        with open(output_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    if os.path.exists(output_csv):
        old_df = pd.read_csv(output_csv)
        new_df = pd.DataFrame(rows_for_csv)
        final_df = pd.concat([old_df, new_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=[args.id_col], keep="last")
    else:
        final_df = pd.DataFrame(rows_for_csv)

    final_df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"JSONL salvo em: {output_jsonl}")
    print(f"CSV salvo em: {output_csv}")


if __name__ == "__main__":
    main()
