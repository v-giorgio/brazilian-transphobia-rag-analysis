import os
from pathlib import Path
import re
import json
import time
import ast
import argparse
import unicodedata
from typing import Any, Dict, List, Set, Tuple, Optional

import faiss
import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

import prompt_constants as pc

# prompt sem counterspeech (para ativar, usar --full-prompt)
# limite de tokens gerados
# retrieval sem bm25
# cache de embeddings
# top_k menor
# few-shot exs menores
# no_think (ollama)
# com benchmark

DEFAULT_TEST_DATA_DIR = "./data/final/test_data"
DEFAULT_OUTPUT_DIR = "./data/final/output"

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen3:14b"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_TRANSCRIPT_INDEX = "../docs/data/transcriptions/base/faiss_index.bin"
DEFAULT_TRANSCRIPT_METADATA = "../docs/data/transcriptions/base/metadata.json"

DEFAULT_ENTITIES_INDEX = "../docs/data/retrieval_split/faiss_entities.bin"
DEFAULT_ENTITIES_METADATA = "../docs/data/retrieval_split/metadata_entities.jsonl"

DEFAULT_EXPLANATIONS_INDEX = "../docs/data/retrieval_split/faiss_explanations.bin"
DEFAULT_EXPLANATIONS_METADATA = "../docs/data/retrieval_split/metadata_explanations.jsonl"

GENERIC_TITLES = {
    "transexualidade",
    "transgénero",
    "transgenero",
    "homem trans",
    "mulher trans",
    "travesti (identidade de gênero)",
    "travesti",
    "identidade de gênero",
    "pessoa trans"
}

STOP_TERMS = {
    "isso", "essa", "esse", "tambem", "porque", "muito", "mais", "para", "com", "uma",
    "uns", "umas", "são", "ser", "tem", "temos", "sobre", "como", "qual", "quais",
    "depois", "antes", "mesmo", "ainda", "tudo", "todos", "todas", "dele", "dela",
    "ali", "aqui", "sua", "seu", "suas", "seus", "elas", "eles", "video", "vídeo",
    "coisa"
}

TITLE_STOPWORDS = {
    "quais", "qual", "como", "porque", "por", "que", "dos", "das", "do", "da",
    "os", "as", "na", "no", "nos", "nas", "um", "uma", "de", "e", "em", "para",
    "com", "sem", "sobre", "contra", "entre", "ao", "aos", "às", "ate", "até",
    "shorts",
}

ABSOLUTE_RELEVANCE_THRESHOLD = 2.2
RELATIVE_RELEVANCE_THRESHOLD = 0.72


def ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model).strip()).strip("_") or "model"


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text


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
        if a in pc.ALLOWED_ASPECTS and a not in seen:
            clean.append(a)
            seen.add(a)
    return clean


def _as_float_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))


def normalize_class_scores(scores: Any, labels: List[str], predicted_label: str) -> Dict[str, float]:
    if not isinstance(scores, dict):
        return {label: 1.0 if label == predicted_label else 0.0 for label in labels}

    clean = {label: _as_float_score(scores.get(label, scores.get(label.lower(), 0.0))) for label in labels}
    total = sum(clean.values())
    if total <= 0:
        return {label: 1.0 if label == predicted_label else 0.0 for label in labels}
    return {label: clean[label] / total for label in labels}


def normalize_aspect_scores(scores: Any, aspects: List[str]) -> Dict[str, float]:
    if not isinstance(scores, dict):
        return {label: 1.0 if label in aspects else 0.0 for label in pc.ASPECT_LABEL_ORDER}
    return {label: _as_float_score(scores.get(label, scores.get(label.lower(), 0.0))) for label in pc.ASPECT_LABEL_ORDER}


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


def validate_prediction(obj: Dict[str, Any], task: str, require_explanations: bool = True) -> Dict[str, Any]:
    task = pc.normalize_task(task)
    speech_class = str(obj.get("speech_class", "")).strip().upper()
    offensive_class = str(obj.get("offensive_class", "")).strip().upper()
    aspects = normalize_aspects(obj.get("aspects", []))
    speech_scores = normalize_class_scores(obj.get("speech_scores"), pc.SPEECH_LABEL_ORDER, speech_class)
    offensive_scores = normalize_class_scores(obj.get("offensive_scores"), pc.OFFENSIVE_LABEL_ORDER, offensive_class)
    aspect_scores = normalize_aspect_scores(obj.get("aspect_scores"), aspects)
    justification = str(obj.get("justification", "")).strip()
    counterspeech = str(obj.get("counterspeech", "")).strip()

    if task == "speech" and speech_class not in pc.ALLOWED_SPEECH:
        raise ValueError(f"speech_class inválida: {speech_class}")
    if task == "offensive" and offensive_class not in pc.ALLOWED_OFFENSIVE:
        raise ValueError(f"offensive_class inválida: {offensive_class}")

    if require_explanations:
        if not justification:
            raise ValueError("justification vazia")
        if not counterspeech:
            raise ValueError("counterspeech vazio")

    return {
        "speech_class": speech_class,
        "offensive_class": offensive_class,
        "aspects": aspects,
        "speech_scores": speech_scores,
        "offensive_scores": offensive_scores,
        "aspect_scores": aspect_scores,
        "justification": justification,
        "counterspeech": counterspeech,
    }


def prediction_output_fields(pred: Dict[str, Any], task: str) -> Dict[str, Any]:
    task = pc.normalize_task(task)
    if task == "speech":
        return {
            "pred_speech_class": pred["speech_class"],
            "pred_speech_scores": json.dumps(pred["speech_scores"], ensure_ascii=False),
        }
    if task == "offensive":
        return {
            "pred_offensive_class": pred["offensive_class"],
            "pred_offensive_scores": json.dumps(pred["offensive_scores"], ensure_ascii=False),
        }
    return {
        "pred_aspects": json.dumps(pred["aspects"], ensure_ascii=False),
        "pred_aspect_scores": json.dumps(pred["aspect_scores"], ensure_ascii=False),
    }


def empty_prediction_output_fields(task: str) -> Dict[str, Any]:
    task = pc.normalize_task(task)
    if task == "speech":
        return {
            "pred_speech_class": None,
            "pred_speech_scores": json.dumps({}, ensure_ascii=False),
        }
    if task == "offensive":
        return {
            "pred_offensive_class": None,
            "pred_offensive_scores": json.dumps({}, ensure_ascii=False),
        }
    return {
        "pred_aspects": json.dumps([], ensure_ascii=False),
        "pred_aspect_scores": json.dumps({}, ensure_ascii=False),
    }


def row_has_error(row: Dict[str, Any]) -> bool:
    return bool(str(row.get("error") or "").strip())


def load_processed_ids(output_jsonl: str, id_col: str, retry_errors: bool = False) -> set:
    latest_by_id: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(output_jsonl):
        return set()

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if id_col in row:
                    latest_by_id[str(row[id_col])] = row
            except Exception:
                continue

    if retry_errors:
        return {
            item_id
            for item_id, row in latest_by_id.items()
            if not row_has_error(row)
        }
    return set(latest_by_id.keys())


def create_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def get_ollama_native_base_url(base_url: str) -> str:
    base_url = str(base_url).strip().rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return base_url


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



def build_example_text_fast(row: pd.Series) -> str:
    aspects = normalize_aspects(row["aspects"])
    aspects_json = json.dumps(aspects, ensure_ascii=False)

    speech = str(row["speech_class"]).strip().upper()
    offensive = str(row["offensive_class"]).strip().upper()
    title = str(row["video_title"])
    comment = str(row["comment"])

    return (
        "Exemplo\n"
        f"Título: {title}\n"
        f"Comentário: {comment}\n"
        f"Resposta: {{\"speech_class\":\"{speech}\","
        f"\"offensive_class\":\"{offensive}\","
        f"\"aspects\":{aspects_json}}}"
    )

def build_example_text(row: pd.Series, task: str) -> str:
    task = pc.normalize_task(task)
    response = {}
    if task == "speech" and "speech_class" in row.index and pd.notna(row["speech_class"]):
        response["speech_class"] = str(row["speech_class"]).strip().upper()
    if task == "offensive" and "offensive_class" in row.index and pd.notna(row["offensive_class"]):
        response["offensive_class"] = str(row["offensive_class"]).strip().upper()
    if task == "aspects" and "aspects" in row.index:
        aspects_value = row["aspects"]
        if isinstance(aspects_value, list) or pd.notna(aspects_value):
            response["aspects"] = normalize_aspects(aspects_value)
    response.update({
        "justification": "Exemplo de referencia; nao reutilize este texto literalmente na resposta final.",
        "counterspeech": "Exemplo de referencia; produza um counterspeech original para o comentario alvo.",
    })
    response_json = json.dumps(response, ensure_ascii=False, indent=2)

    return f'''Exemplo
Titulo do video:
{row["video_title"]}

Comentario:
{row["comment"]}

Resposta:
{response_json}'''


def build_example_text_fast(row: pd.Series, task: str) -> str:
    task = pc.normalize_task(task)
    response = {}
    if task == "speech" and "speech_class" in row.index and pd.notna(row["speech_class"]):
        response["speech_class"] = str(row["speech_class"]).strip().upper()
    if task == "offensive" and "offensive_class" in row.index and pd.notna(row["offensive_class"]):
        response["offensive_class"] = str(row["offensive_class"]).strip().upper()
    if task == "aspects" and "aspects" in row.index:
        aspects_value = row["aspects"]
        if isinstance(aspects_value, list) or pd.notna(aspects_value):
            response["aspects"] = normalize_aspects(aspects_value)
    response_json = json.dumps(response, ensure_ascii=False, separators=(",", ":"))

    return (
        "Exemplo\n"
        f"Titulo: {row['video_title']}\n"
        f"Comentario: {row['comment']}\n"
        f"Resposta: {response_json}"
    )


def select_examples(
    df: pd.DataFrame,
    task: str,
    target_id: str,
    id_col: str,
    speech_per_class: int,
    offensive_per_class: int,
    aspect_per_label: int,
) -> pd.DataFrame:
    task = pc.normalize_task(task)
    pool = df[df[id_col].astype(str) != str(target_id)].copy()

    selected_ids: Set[str] = set()
    selected_rows: List[pd.Series] = []

    if task == "speech":
        for label in ["HATEFUL", "SUPPORTIVE", "NEUTRAL"]:
            subset = pool[pool["speech_class"].astype(str).str.upper() == label]
            for _, row in subset.head(speech_per_class).iterrows():
                rid = str(row[id_col])
                if rid not in selected_ids:
                    selected_ids.add(rid)
                    selected_rows.append(row)

    if task == "offensive" and "offensive_class" in pool.columns:
        for label in ["OFFENSIVE", "NON_OFFENSIVE"]:
            subset = pool[pool["offensive_class"].astype(str).str.upper() == label]
            for _, row in subset.head(offensive_per_class).iterrows():
                rid = str(row[id_col])
                if rid not in selected_ids:
                    selected_ids.add(rid)
                    selected_rows.append(row)

    if task == "aspects" and "aspects_list" in pool.columns:
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


def render_few_shot_examples(examples_df: pd.DataFrame, task: str, fast_prompt: bool = False) -> str:
    if fast_prompt:
        return "\n\n".join(build_example_text_fast(row, task) for _, row in examples_df.iterrows())
    return "\n\n".join(build_example_text(row, task) for _, row in examples_df.iterrows())


def _read_examples_csv(path: str, required_cols: List[str], id_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in [id_col, "video_title", "comment", *required_cols]:
        if col not in df.columns:
            raise ValueError(f"O CSV de exemplos '{path}' precisa conter a coluna '{col}'.")
    return df.drop_duplicates(subset=[id_col], keep="first").copy()


def _aspect_list_from_indicator_row(row: pd.Series) -> List[str]:
    labels = []
    for label in pc.ASPECT_LABEL_ORDER:
        value = row.get(label, 0)
        if pd.notna(value) and int(value) == 1:
            labels.append(label)
    return labels


def load_examples_dataframe(path: str, task: str, id_col: str) -> pd.DataFrame:
    task = pc.normalize_task(task)
    if task == "speech":
        df = _read_examples_csv(path, ["speech_class"], id_col)
        df["speech_class"] = df["speech_class"].astype(str).str.strip().str.upper()
        return df[[id_col, "video_title", "comment", "speech_class"]].copy()

    if task == "offensive":
        df = _read_examples_csv(path, ["offensive_class"], id_col)
        df["offensive_class"] = df["offensive_class"].astype(str).str.strip().str.upper()
        return df[[id_col, "video_title", "comment", "offensive_class"]].copy()

    df = _read_examples_csv(path, pc.ASPECT_LABEL_ORDER, id_col)
    df["aspects"] = df.apply(_aspect_list_from_indicator_row, axis=1)
    df["aspects_list"] = df["aspects"].apply(normalize_aspects)
    return df[[id_col, "video_title", "comment", "aspects", "aspects_list"]].copy()


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
    title_clean = re.sub(r'["“”\']', "", title)

    if title_clean.endswith("?") or "#shorts" in title_clean.lower():
        return ""

    m = re.search(
        r"\b([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+(?:\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+){0,2})\b\s+(?:diz|disse|fala|afirma|afirmou|lamenta|lamentou|discute|criticou|critica|rebate|comenta|declara|defende|ataca)\b",
        title_clean,
    )
    if m:
        return m.group(1).strip()

    sequences = re.findall(
        r"\b[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+(?:\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+){0,2}\b",
        title_clean,
    )
    blacklist = {
        "Pré", "Candidato", "Direita", "Esquerda", "Brasil", "Washington", "Rio", "Janeiro",
        "Mulher", "Homem", "Pessoas", "Trans", "Transgênero", "Transexualidade", "Política",
        "Quais", "Qual", "Como", "Consequências", "Consequencias", "Projetos",
    }
    for seq in sequences:
        if seq.split()[0] not in blacklist:
            return seq.strip()

    return ""


def comment_focus_terms(comment: str, max_terms: int = 6) -> List[str]:
    toks = tokenize_portuguese(comment)
    terms = []
    for t in toks:
        if len(t) < 4:
            continue
        if t in STOP_TERMS or t in {"vcs", "vc", "qto", "bozo", "linda"}:
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


def title_focus_terms(video_title: str, max_terms: int = 8) -> List[str]:
    toks = tokenize_portuguese(video_title)
    out = []
    seen = set()
    for t in toks:
        if len(t) < 4:
            continue
        if t in TITLE_STOPWORDS:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_terms:
            break
    return out


def title_priority_terms(video_title: str, max_terms: int = 6) -> List[str]:
    strong_terms = {
        "politica", "política", "cargo", "mandato", "parlamentar", "vereador", "deputado",
        "agressao", "agressão", "agressoes", "agressões", "ameaca", "ameaça",
        "ameacas", "ameaças", "violencia", "violência", "medo", "ataque", "ataques"
    }
    strong_norm = {normalize_text(x) for x in strong_terms}

    out = []
    for t in title_focus_terms(video_title, max_terms=12):
        if normalize_text(t) in strong_norm:
            out.append(t)

    return out[:max_terms]


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
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        embedder: SentenceTransformer,
        name: str = "",
        candidate_multiplier: int = 4,
        query_cache: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.name = name or Path(index_path).stem
        self.index = faiss.read_index(index_path)
        self.metadata = load_metadata(metadata_path)
        self.embedder = embedder
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self.query_cache = query_cache if query_cache is not None else {}

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Inconsistência entre índice e metadata em {self.name}: "
                f"index.ntotal={self.index.ntotal}, metadata={len(self.metadata)}"
            )

    def encode_query(self, query: str) -> np.ndarray:
        key = normalize_text(query)
        if key not in self.query_cache:
            self.query_cache[key] = np.asarray(
                self.embedder.encode([query], normalize_embeddings=True),
                dtype="float32",
            )
        return self.query_cache[key]

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        q_emb = self.encode_query(query)
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        distances, indices = self.index.search(q_emb, candidate_k)

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
    def __init__(self, metadata: List[Dict[str, Any]], name: str = ""):
        self.metadata = metadata
        self.name = name
        self.documents = [metadata_to_document(item) for item in metadata]
        self.retriever = BM25Retriever.from_documents(self.documents)

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
            normalize_text(item.get("source", item.get("_retriever", ""))),
            normalize_text(item.get("title", "")),
            int(item.get("chunk_index", 0)) if str(item.get("chunk_index", "")).strip() else -1,
            normalize_text(str(item.get("text", "")).strip()[:220]),
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


def filter_relevant_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return []

    scored = [x for x in results if "_final_score" in x]
    if not scored:
        return []

    best_score = max(float(x["_final_score"]) for x in scored)

    filtered = []
    for item in scored:
        score = float(item["_final_score"])
        if score >= ABSOLUTE_RELEVANCE_THRESHOLD or score >= best_score * RELATIVE_RELEVANCE_THRESHOLD:
            filtered.append(item)

    return filtered


def is_generic_title(title: str) -> bool:
    return normalize_text(title) in {normalize_text(x) for x in GENERIC_TITLES}


def transcript_is_relevant(item: Dict[str, Any], video_title: str, entity: str) -> bool:
    text = normalize_text(item.get("text", ""))
    title_terms = title_focus_terms(video_title, max_terms=6)
    title_overlap = lexical_overlap_score(text, title_terms)
    if entity and normalize_text(entity) in text:
        return True
    return title_overlap >= 0.15


def explanation_is_relevant(item: Dict[str, Any], video_title: str, comment: str, entity: str) -> bool:
    title = normalize_text(item.get("title", ""))
    text = normalize_text(item.get("text", ""))
    title_terms = title_focus_terms(video_title, max_terms=6)
    comment_terms = comment_focus_terms(comment, max_terms=6)
    priority_terms = title_priority_terms(video_title, max_terms=6)

    title_overlap = lexical_overlap_score(title + " " + text[:1200], title_terms)
    comment_overlap = lexical_overlap_score(title + " " + text[:1200], comment_terms)
    priority_overlap = lexical_overlap_score(title + " " + text[:1200], priority_terms)

    if entity and (normalize_text(entity) in title or normalize_text(entity) in text):
        return True
    if is_generic_title(title):
        return False

    if priority_terms:
        return priority_overlap >= 0.20 or title_overlap >= 0.22 or comment_overlap >= 0.22

    return title_overlap >= 0.18 or comment_overlap >= 0.20


def entity_is_relevant(item: Dict[str, Any], video_title: str, comment: str, entity: str) -> bool:
    title = normalize_text(item.get("title", ""))
    text = normalize_text(item.get("text", ""))
    if entity:
        ent = normalize_text(entity)
        return ent in title or ent in text
    title_terms = title_focus_terms(video_title, max_terms=6)
    comment_terms = comment_focus_terms(comment, max_terms=6)
    return lexical_overlap_score(title + " " + text, title_terms) >= 0.20 or lexical_overlap_score(title + " " + text, comment_terms) >= 0.20


def rerank_transcript_items(items: List[Dict[str, Any]], video_title: str, comment: str, entity: str, top_k: int, max_per_group: int = 1) -> List[Dict[str, Any]]:
    q_terms = comment_focus_terms(comment)
    title_terms = title_focus_terms(video_title, max_terms=6)

    items = [x for x in items if transcript_is_relevant(x, video_title, entity)]

    normalize_scores(items, "_semantic_score", "_semantic_norm")
    normalize_scores(items, "_bm25_rank_score", "_bm25_norm")
    normalize_scores(items, "_tfidf_score", "_tfidf_norm")
    normalize_scores(items, "_lexical_count", "_lexical_norm")

    ranked = []
    for item in items:
        text = str(item.get("text", ""))
        score = 0.0
        score += 0.70 * float(item.get("_semantic_norm", 0.0))
        score += 1.05 * float(item.get("_bm25_norm", 0.0))
        score += 0.45 * float(item.get("_tfidf_norm", 0.0))
        score += 0.35 * float(item.get("_lexical_norm", 0.0))
        score += 0.35 * lexical_overlap_score(text, q_terms)
        score += 0.65 * lexical_overlap_score(text, title_terms)

        if entity and normalize_text(entity) in normalize_text(text):
            score += 1.20

        item["_final_score"] = score
        item["_group_key"] = " ".join(tokenize_portuguese(text)[:20])
        ranked.append(item)

    ranked.sort(key=lambda x: x["_final_score"], reverse=True)
    ranked = filter_relevant_results(ranked)

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


def rerank_entity_items(items: List[Dict[str, Any]], video_title: str, comment: str, entity: str, top_k: int) -> List[Dict[str, Any]]:
    q_terms = comment_focus_terms(comment)
    title_terms = title_focus_terms(video_title)

    items = [x for x in items if entity_is_relevant(x, video_title, comment, entity)]

    normalize_scores(items, "_semantic_score", "_semantic_norm")
    normalize_scores(items, "_bm25_rank_score", "_bm25_norm")
    normalize_scores(items, "_tfidf_score", "_tfidf_norm")
    normalize_scores(items, "_lexical_count", "_lexical_norm")

    ranked = []
    for item in items:
        title = str(item.get("title", "")).strip()
        text = str(item.get("text", "")).strip()
        norm_title = normalize_text(title)

        score = 0.0
        score += 0.40 * float(item.get("_semantic_norm", 0.0))
        score += 1.10 * float(item.get("_bm25_norm", 0.0))
        score += 0.80 * float(item.get("_tfidf_norm", 0.0))
        score += 0.55 * float(item.get("_lexical_norm", 0.0))
        score += 0.30 * lexical_overlap_score(title + " " + text, q_terms)
        score += 0.70 * lexical_overlap_score(title + " " + text, title_terms)

        if entity and normalize_text(entity) in norm_title:
            score += 1.80
        elif entity and normalize_text(entity) in normalize_text(text):
            score += 0.80

        item["_final_score"] = score
        ranked.append(item)

    ranked.sort(key=lambda x: x["_final_score"], reverse=True)
    ranked = filter_relevant_results(ranked)

    out = []
    seen_titles = set()
    for item in ranked:
        title_key = normalize_text(item.get("title", ""))
        if title_key and title_key in seen_titles:
            continue
        if title_key:
            seen_titles.add(title_key)
        out.append(item)
        if len(out) >= top_k:
            break
    return out


def rerank_explanation_items(items: List[Dict[str, Any]], video_title: str, comment: str, entity: str, top_k: int) -> List[Dict[str, Any]]:
    q_terms = comment_focus_terms(comment)
    title_terms = title_focus_terms(video_title)
    priority_terms = title_priority_terms(video_title, max_terms=6)

    items = [x for x in items if explanation_is_relevant(x, video_title, comment, entity)]

    normalize_scores(items, "_semantic_score", "_semantic_norm")
    normalize_scores(items, "_bm25_rank_score", "_bm25_norm")
    normalize_scores(items, "_tfidf_score", "_tfidf_norm")
    normalize_scores(items, "_lexical_count", "_lexical_norm")

    ranked = []
    for item in items:
        title = str(item.get("title", "")).strip()
        text = str(item.get("text", "")).strip()
        norm_title = normalize_text(title)

        score = 0.0
        score += 0.40 * float(item.get("_semantic_norm", 0.0))
        score += 1.20 * float(item.get("_bm25_norm", 0.0))
        score += 0.85 * float(item.get("_tfidf_norm", 0.0))
        score += 0.65 * float(item.get("_lexical_norm", 0.0))
        score += 0.30 * lexical_overlap_score(text, q_terms)
        score += 0.75 * lexical_overlap_score(title + " " + text, title_terms[:8])
        score += 1.10 * lexical_overlap_score(title + " " + text, priority_terms)

        if entity and normalize_text(entity) in norm_title:
            score += 1.20
        elif entity and normalize_text(entity) in normalize_text(text):
            score += 0.50

        if is_generic_title(title):
            score -= 1.00

        item["_final_score"] = score
        ranked.append(item)

    ranked.sort(key=lambda x: x["_final_score"], reverse=True)
    ranked = filter_relevant_results(ranked)

    out = []
    seen_titles = set()
    for item in ranked:
        title_key = normalize_text(item.get("title", ""))
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
    comment_terms = comment_focus_terms(comment, max_terms=6)
    title_terms = title_focus_terms(video_title, max_terms=8)
    priority_terms = title_priority_terms(video_title, max_terms=6)

    title_comment = f"Título do vídeo: {video_title}\nComentário: {comment}"
    entity_comment = f"{entity} {video_title} {comment}".strip() if entity else title_comment
    constrained = " ".join(
        ([entity] if entity else []) + priority_terms[:4] + title_terms[:4] + comment_terms[:3]
    ).strip()

    return {
        "entity": entity,
        "title_comment": title_comment,
        "entity_comment": entity_comment,
        "constrained": constrained or title_comment,
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


def get_context_id(item: Dict[str, Any], index: int) -> str:
    for key in ("id", "chunk_id", "doc_id", "video_id", "url"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    source = str(item.get("source", item.get("_retriever", ""))).strip()
    title = str(item.get("title", "")).strip()
    return f"{source}:{title}:{index}" if source or title else f"context_{index}"


def ragas_context_payloads(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    contexts = []
    context_ids = []
    metadata = []

    for index, item in enumerate(items, start=1):
        context_id = get_context_id(item, index)
        context_ids.append(context_id)
        contexts.append(safe_get_text(item))
        metadata.append({
            "id": context_id,
            "title": str(item.get("title", "")),
            "source": str(item.get("source", item.get("_retriever", ""))),
            "retriever": str(item.get("_retriever", "")),
            "score": item.get("_final_score", item.get("_faiss_score")),
            "semantic_score": item.get("_semantic_score"),
            "bm25_score": item.get("_bm25_rank_score"),
            "tfidf_score": item.get("_tfidf_score"),
        })

    return {
        "retrieved_contexts": contexts,
        "retrieved_context_ids": context_ids,
        "retrieved_context_metadata": metadata,
    }


def retrieve_hybrid(
    query_text: str,
    fallback_query: str,
    top_k: int,
    faiss_retriever: FaissRetriever,
    bm25_retriever: LangChainBM25Retriever,
    tfidf_retriever: TfidfMetadataRetriever,
    lexical_terms: List[str],
    retrieval_level: str = "full",
) -> List[Dict[str, Any]]:
    candidates = []

    if retrieval_level == "semantic":
        candidates.extend(faiss_retriever.search(fallback_query or query_text, top_k=top_k))
        return dedup_results(candidates)

    if retrieval_level == "fast":
        candidates.extend(faiss_retriever.search(fallback_query or query_text, top_k=top_k))
        candidates.extend(
            tfidf_retriever.search(
                fallback_query or query_text,
                query_terms=lexical_terms,
                top_k=max(top_k * 2, 8),
            )
        )
        return dedup_results(candidates)

    candidates.extend(faiss_retriever.search(query_text, top_k=top_k))
    if fallback_query != query_text:
        candidates.extend(faiss_retriever.search(fallback_query, top_k=top_k))
    candidates.extend(bm25_retriever.search(query_text, top_k=max(top_k * 4, 12)))
    candidates.extend(tfidf_retriever.search(query_text, query_terms=lexical_terms, top_k=max(top_k * 4, 12)))
    return dedup_results(candidates)



def retrieve_context(video_title: str, comment: str, mode: str, top_k_transcripts: int, top_k_entities: int, top_k_explanations: int,
    transcripts_retriever: Optional[FaissRetriever], transcripts_bm25: Optional[LangChainBM25Retriever], transcripts_tfidf: Optional[TfidfMetadataRetriever],
    entities_retriever: Optional[FaissRetriever], entities_bm25: Optional[LangChainBM25Retriever], entities_tfidf: Optional[TfidfMetadataRetriever],
    explanations_retriever: Optional[FaissRetriever], explanations_bm25: Optional[LangChainBM25Retriever], explanations_tfidf: Optional[TfidfMetadataRetriever],
    retrieval_level: str = "full"
) -> Tuple[str, List[Dict[str, Any]], Dict[str, str]]:
    queries = build_retrieval_queries(video_title, comment)
    entity = queries["entity"]
    comment_terms = comment_focus_terms(comment, max_terms=6)
    title_terms = title_focus_terms(video_title, max_terms=8)
    lexical_terms = list(dict.fromkeys(([entity.lower()] if entity else []) + title_terms[:6] + comment_terms[:6]))

    results: List[Dict[str, Any]] = []

    if mode in {"transcripts", "both"}:
        if transcripts_retriever is None or transcripts_bm25 is None or transcripts_tfidf is None:
            raise ValueError("Retriever de transcrições não foi carregado.")

        transcript_candidates = retrieve_hybrid(
            query_text=queries["title_comment"],
            fallback_query=queries["constrained"],
            top_k=top_k_transcripts,
            faiss_retriever=transcripts_retriever,
            bm25_retriever=transcripts_bm25,
            tfidf_retriever=transcripts_tfidf,
            lexical_terms=lexical_terms,
            retrieval_level=retrieval_level,
        )
        if entity:
            transcript_candidates.extend(dedup_results(transcripts_retriever.search(queries["entity_comment"], top_k=top_k_transcripts)))

        transcript_candidates = dedup_results(transcript_candidates)
        transcript_candidates = rerank_transcript_items(
            transcript_candidates, video_title=video_title, comment=comment, entity=entity, top_k=top_k_transcripts
        )
        results.extend(transcript_candidates)

    if mode in {"wiki", "both"}:
        if explanations_retriever is None or explanations_bm25 is None or explanations_tfidf is None:
            raise ValueError("Retriever de explanations não foi carregado.")

        explanation_candidates = retrieve_hybrid(
            query_text=queries["title_comment"],
            fallback_query=queries["constrained"],
            top_k=top_k_explanations,
            faiss_retriever=explanations_retriever,
            bm25_retriever=explanations_bm25,
            tfidf_retriever=explanations_tfidf,
            lexical_terms=lexical_terms,
            retrieval_level=retrieval_level,
        )
        explanation_candidates = rerank_explanation_items(
            explanation_candidates, video_title=video_title, comment=comment, entity=entity, top_k=top_k_explanations
        )
        results.extend(explanation_candidates)

        if entity and entities_retriever and entities_bm25 and entities_tfidf:
            entity_candidates = retrieve_hybrid(
                query_text=queries["entity_comment"],
                fallback_query=queries["title_comment"],
                top_k=top_k_entities,
                faiss_retriever=entities_retriever,
                bm25_retriever=entities_bm25,
                tfidf_retriever=entities_tfidf,
                lexical_terms=lexical_terms,
                retrieval_level=retrieval_level,
            )
            entity_candidates = rerank_entity_items(
                entity_candidates, video_title=video_title, comment=comment, entity=entity, top_k=top_k_entities
            )
            results.extend(entity_candidates)

    results = dedup_results(results)
    results.sort(key=lambda x: x.get("_final_score", x.get("_faiss_score", 0.0)), reverse=True)
    results = filter_relevant_results(results)

    context_blocks = []
    for i, item in enumerate(results, start=1):
        context_blocks.append(f"[Contexto {i}]\n{safe_get_text(item)}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "Nenhum contexto recuperado."
    return context_text, results, queries


def classify_comment(
    client: OpenAI,
    model: str,
    task: str,
    video_title: str,
    comment: str,
    few_shot_examples: str,
    retrieved_context: str,
    temperature: float,
    max_retries: int,
    sleep_seconds: float,
    fast_prompt: bool = True,
    max_tokens: int = 256,
    base_url: str = DEFAULT_BASE_URL,
    use_ollama_native: bool = True,
    disable_thinking: bool = True,
    request_timeout: float = 180.0,
) -> Dict[str, Any]:
    system_prompt, user_template = pc.get_prompt_pair(task, strategy="rag", fast_prompt=fast_prompt)
    user_prompt = user_template.format(
        video_title=video_title,
        comment=comment,
        few_shot_examples=few_shot_examples,
        retrieved_context=retrieved_context,
    )

    last_error = None

    if use_ollama_native:
        native_base_url = get_ollama_native_base_url(base_url)
        url = f"{native_base_url}/api/chat"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "think": not disable_thinking,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=request_timeout)
                response.raise_for_status()
                data = response.json()

                text = data.get("message", {}).get("content", "") or ""
                parsed = extract_json_object(text)
                return validate_prediction(parsed, task=task, require_explanations=not fast_prompt)

            except Exception as e:
                last_error = e
                time.sleep(sleep_seconds * (2 ** attempt))

        raise RuntimeError(f"Falha após {max_retries} tentativas: {last_error}")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            text = response.choices[0].message.content or ""
            parsed = extract_json_object(text)
            return validate_prediction(parsed, task=task, require_explanations=not fast_prompt)

        except Exception as e:
            last_error = e
            time.sleep(sleep_seconds * (2 ** attempt))

    raise RuntimeError(f"Falha após {max_retries} tentativas: {last_error}")


def default_input_path(task: str) -> str:
    task = pc.normalize_task(task)
    return f"{DEFAULT_TEST_DATA_DIR}/{task}/test.csv"


def default_examples_input_path(task: str) -> str:
    task = pc.normalize_task(task)
    return f"{DEFAULT_TEST_DATA_DIR}/{task}/train.csv"


def default_output_paths(output_dir: str, task: str, rag_mode: str, model: str) -> Tuple[str, str]:
    task = pc.normalize_task(task)
    model_slug = sanitize_model_name(model)
    base = Path(output_dir) / "rag" / task / f"predictions_{model_slug}_{rag_mode}"
    return f"{base}.csv", f"{base}.jsonl"


def _mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _median(values: List[Optional[float]]) -> Optional[float]:
    clean = sorted(float(v) for v in values if v is not None and not pd.isna(v))
    if not clean:
        return None
    n = len(clean)
    mid = n // 2
    if n % 2 == 1:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2


def _sum(values: List[Any]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean:
        return None
    return sum(clean)


def _round_or_none(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def _value_or_none(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    return value


def benchmark_rows_from_dataframe(df: pd.DataFrame, id_col: str) -> List[Dict[str, Any]]:
    rows = []
    if df.empty or id_col not in df.columns:
        return rows

    for _, row in df.iterrows():
        error = row.get("error")
        has_error = bool(str(error).strip()) if pd.notna(error) else False
        rows.append({
            "dataset_comment_id": str(row[id_col]),
            "success": not has_error,
            "retrieval_time_seconds": row.get("retrieval_time_seconds"),
            "inference_time_seconds": row.get("inference_time_seconds"),
            "total_time_seconds": row.get("total_time_seconds"),
            "num_retrieved_items": _value_or_none(row.get("num_retrieved_items")),
        })
    return rows


def summarize_benchmark(
    benchmark_rows: List[Dict[str, Any]],
    total_run_seconds: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    attempted = len(benchmark_rows)
    successful = sum(1 for x in benchmark_rows if x.get("success"))
    failed = attempted - successful

    retrieval_times = [x.get("retrieval_time_seconds") for x in benchmark_rows]
    inference_times = [x.get("inference_time_seconds") for x in benchmark_rows]
    total_times = [x.get("total_time_seconds") for x in benchmark_rows]
    retrieved_counts = [x.get("num_retrieved_items") for x in benchmark_rows if x.get("num_retrieved_items") is not None]

    return {
        "model": args.model,
        "task": args.task,
        "rag_mode": args.rag_mode,
        "retrieval_level": args.retrieval_level,
        "fast_prompt": bool(args.fast_prompt),
        "ollama_native": bool(args.ollama_native),
        "thinking_disabled": bool(args.disable_thinking),
        "limit": args.limit,
        "attempted_classifications": attempted,
        "successful_classifications": successful,
        "failed_classifications": failed,
        "total_run_seconds": round(float(total_run_seconds), 6),
        "average_total_time_per_attempt_seconds": round(float(total_run_seconds / attempted), 6) if attempted else None,
        "average_total_time_per_success_seconds": round(float(total_run_seconds / successful), 6) if successful else None,
        "retrieval_time_seconds": {
            "mean": round(_mean(retrieval_times), 6) if _mean(retrieval_times) is not None else None,
            "median": round(_median(retrieval_times), 6) if _median(retrieval_times) is not None else None,
        },
        "inference_time_seconds": {
            "mean": round(_mean(inference_times), 6) if _mean(inference_times) is not None else None,
            "median": round(_median(inference_times), 6) if _median(inference_times) is not None else None,
        },
        "classification_total_time_seconds": {
            "mean": round(_mean(total_times), 6) if _mean(total_times) is not None else None,
            "median": round(_median(total_times), 6) if _median(total_times) is not None else None,
        },
        "retrieved_items": {
            "mean": round(_mean(retrieved_counts), 6) if _mean(retrieved_counts) is not None else None,
            "median": round(_median(retrieved_counts), 6) if _median(retrieved_counts) is not None else None,
        },
        "per_item": [
            {
                **row,
                "retrieval_time_seconds": _round_or_none(row.get("retrieval_time_seconds")),
                "inference_time_seconds": _round_or_none(row.get("inference_time_seconds")),
                "total_time_seconds": _round_or_none(row.get("total_time_seconds")),
            }
            for row in benchmark_rows
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Classificador few-shot + RAG híbrido com transcripts + entities + explanations.")
    parser.add_argument("--task", choices=pc.TASKS, default="speech", help="Tarefa a classificar")
    parser.add_argument("--input", default=None, help="Caminho do CSV de entrada")
    parser.add_argument("--examples-input", default=None, help="CSV usado como fonte dos exemplos da tarefa")
    parser.add_argument("--speech-examples-input", default=None, help="CSV usado como fonte dos exemplos de speech")
    parser.add_argument("--offensive-examples-input", default=None, help="CSV usado como fonte dos exemplos de offensive")
    parser.add_argument("--aspects-examples-input", default=None, help="CSV usado como fonte dos exemplos de aspects")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Diretório base de saída")
    parser.add_argument("--output-csv", default=None, help="Caminho do CSV de saída")
    parser.add_argument("--output-jsonl", default=None, help="Caminho do JSONL incremental")
    parser.add_argument("--benchmark-report", default=None, help="Caminho do JSON com métricas agregadas de execução")
    parser.add_argument("--rag-mode", choices=["transcripts", "wiki", "both"], required=True)
    parser.add_argument("--model", "--llm-model", dest="model", default=DEFAULT_MODEL, help="Modelo Ollama/OpenAI-compatible")
    parser.add_argument("--api-key", default="ollama", help="Chave de API; para Ollama pode ser qualquer string")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL da API compatível com OpenAI")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Modelo de embeddings")

    parser.add_argument("--transcript-index", default=DEFAULT_TRANSCRIPT_INDEX)
    parser.add_argument("--transcript-metadata", default=DEFAULT_TRANSCRIPT_METADATA)

    parser.add_argument("--entities-index", default=DEFAULT_ENTITIES_INDEX)
    parser.add_argument("--entities-metadata", default=DEFAULT_ENTITIES_METADATA)
    parser.add_argument("--explanations-index", default=DEFAULT_EXPLANATIONS_INDEX)
    parser.add_argument("--explanations-metadata", default=DEFAULT_EXPLANATIONS_METADATA)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--sleep-seconds", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--id-col", default="dataset_comment_id")
    parser.add_argument("--speech-examples-per-class", type=int, default=10)
    parser.add_argument("--offensive-examples-per-class", type=int, default=10)
    parser.add_argument("--aspect-examples-per-label", type=int, default=5)
    parser.add_argument("--top-k-transcripts", type=int, default=3)
    parser.add_argument("--top-k-entities", type=int, default=1)
    parser.add_argument("--top-k-explanations", type=int, default=2)
    parser.add_argument("--retrieval-level", choices=["semantic", "fast", "full"], default="full")
    parser.add_argument("--faiss-candidate-multiplier", type=int, default=4)
    parser.add_argument("--fast-prompt", action="store_true", default=False, help="Usa prompt curto sem justification/counterspeech.")
    parser.add_argument("--full-prompt", action="store_false", dest="fast_prompt", help="Usa prompt completo com justification/counterspeech.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--ollama-native", action="store_true", default=True, help="Usa a API nativa do Ollama /api/chat.")
    parser.add_argument("--openai-compatible", action="store_false", dest="ollama_native", help="Usa a API OpenAI-compatible /v1.")
    parser.add_argument("--disable-thinking", action="store_true", default=True, help="Envia think=false para modelos compatíveis, como Qwen3.")
    parser.add_argument("--enable-thinking", action="store_false", dest="disable_thinking", help="Permite thinking em modelos compatíveis.")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        default=False,
        help="Reclassifica registros cuja tentativa mais recente no JSONL possui error preenchido.",
    )
    args = parser.parse_args()
    args.task = pc.normalize_task(args.task)
    if args.input is None:
        args.input = default_input_path(args.task)
    if args.examples_input is None:
        task_specific_examples = getattr(args, f"{args.task}_examples_input", None)
        args.examples_input = task_specific_examples or default_examples_input_path(args.task)

    output_csv = args.output_csv
    output_jsonl = args.output_jsonl
    if output_csv is None or output_jsonl is None:
        default_csv, default_jsonl = default_output_paths(args.output_dir, args.task, args.rag_mode, args.model)
        output_csv = output_csv or default_csv
        output_jsonl = output_jsonl or default_jsonl

    ensure_dir(output_csv)
    ensure_dir(output_jsonl)

    df = pd.read_csv(args.input)

    if args.id_col not in df.columns:
        raise ValueError(f"Coluna '{args.id_col}' não encontrada.")
    if "video_title" not in df.columns or "comment" not in df.columns:
        raise ValueError("O CSV precisa conter as colunas 'video_title' e 'comment'.")
    if args.task == "aspects" and "aspects" not in df.columns and all(label in df.columns for label in pc.ASPECT_LABEL_ORDER):
        df["aspects"] = df.apply(_aspect_list_from_indicator_row, axis=1)

    true_col = pc.TASK_TO_TRUE_COLUMN[args.task]
    if true_col not in df.columns:
        raise ValueError(f"O CSV de entrada precisa conter a coluna '{true_col}'.")

    if "speech_class" in df.columns:
        df["speech_class"] = df["speech_class"].astype(str).str.strip().str.upper()
    if "offensive_class" in df.columns:
        df["offensive_class"] = df["offensive_class"].astype(str).str.strip().str.upper()
    if "aspects" in df.columns:
        df["aspects_list"] = df["aspects"].apply(normalize_aspects)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    processed_ids = load_processed_ids(output_jsonl, args.id_col, retry_errors=args.retry_errors)
    if processed_ids:
        df = df[~df[args.id_col].astype(str).isin(processed_ids)].copy()

    examples_source_df = load_examples_dataframe(
        path=args.examples_input,
        task=args.task,
        id_col=args.id_col,
    )

    eval_ids = set(df[args.id_col].astype(str))
    examples_source_df = examples_source_df[~examples_source_df[args.id_col].astype(str).isin(eval_ids)].copy()

    client = create_client(args.api_key, args.base_url)
    embedder = SentenceTransformer(args.embedding_model)
    query_embedding_cache: Dict[str, np.ndarray] = {}

    transcripts_retriever = transcripts_bm25 = transcripts_tfidf = None
    entities_retriever = entities_bm25 = entities_tfidf = None
    explanations_retriever = explanations_bm25 = explanations_tfidf = None

    if args.rag_mode in {"transcripts", "both"}:
        transcripts_retriever = FaissRetriever(
            index_path=args.transcript_index,
            metadata_path=args.transcript_metadata,
            embedder=embedder,
            name="transcripts",
            candidate_multiplier=args.faiss_candidate_multiplier,
            query_cache=query_embedding_cache,
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
        entities_retriever = FaissRetriever(
            index_path=args.entities_index,
            metadata_path=args.entities_metadata,
            embedder=embedder,
            name="entities",
            candidate_multiplier=args.faiss_candidate_multiplier,
            query_cache=query_embedding_cache,
        )
        entities_bm25 = LangChainBM25Retriever(
            metadata=entities_retriever.metadata,
            name="entities",
        )
        entities_tfidf = TfidfMetadataRetriever(
            metadata=entities_retriever.metadata,
            name="entities",
        )

        explanations_retriever = FaissRetriever(
            index_path=args.explanations_index,
            metadata_path=args.explanations_metadata,
            embedder=embedder,
            name="explanations",
            candidate_multiplier=args.faiss_candidate_multiplier,
            query_cache=query_embedding_cache,
        )
        explanations_bm25 = LangChainBM25Retriever(
            metadata=explanations_retriever.metadata,
            name="explanations",
        )
        explanations_tfidf = TfidfMetadataRetriever(
            metadata=explanations_retriever.metadata,
            name="explanations",
        )

    rows_for_csv = []
    benchmark_rows = []
    benchmark_run_start = time.perf_counter()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classificando"):
        comment_id = str(row[args.id_col])
        video_title = str(row["video_title"])
        comment = str(row["comment"])

        item_total_start = time.perf_counter()
        retrieval_time = None
        inference_time = None
        total_time = None

        try:
            examples_df = select_examples(
                df=examples_source_df,
                task=args.task,
                target_id=comment_id,
                id_col=args.id_col,
                speech_per_class=args.speech_examples_per_class,
                offensive_per_class=args.offensive_examples_per_class,
                aspect_per_label=args.aspect_examples_per_label,
            )
            few_shot_examples = render_few_shot_examples(examples_df, task=args.task, fast_prompt=args.fast_prompt)

            retrieval_start = time.perf_counter()
            retrieved_context, retrieved_items, used_queries = retrieve_context(
                video_title=video_title,
                comment=comment,
                mode=args.rag_mode,
                top_k_transcripts=args.top_k_transcripts,
                top_k_entities=args.top_k_entities,
                top_k_explanations=args.top_k_explanations,
                transcripts_retriever=transcripts_retriever,
                transcripts_bm25=transcripts_bm25,
                transcripts_tfidf=transcripts_tfidf,
                entities_retriever=entities_retriever,
                entities_bm25=entities_bm25,
                entities_tfidf=entities_tfidf,
                explanations_retriever=explanations_retriever,
                explanations_bm25=explanations_bm25,
                explanations_tfidf=explanations_tfidf,
                retrieval_level=args.retrieval_level,
            )
            retrieval_time = time.perf_counter() - retrieval_start
            ragas_payload = ragas_context_payloads(retrieved_items)

            inference_start = time.perf_counter()
            pred = classify_comment(
                client=client,
                model=args.model,
                task=args.task,
                video_title=video_title,
                comment=comment,
                few_shot_examples=few_shot_examples,
                retrieved_context=retrieved_context,
                temperature=args.temperature,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
                fast_prompt=args.fast_prompt,
                max_tokens=args.max_tokens,
                base_url=args.base_url,
                use_ollama_native=args.ollama_native,
                disable_thinking=args.disable_thinking,
                request_timeout=args.request_timeout,
            )
            inference_time = time.perf_counter() - inference_start
            total_time = time.perf_counter() - item_total_start

            output_row = row.to_dict()
            output_row.pop("aspects_list", None)
            output_row.update({
                **prediction_output_fields(pred, args.task),
                "pred_justification": pred.get("justification", ""),
                "pred_counterspeech": pred.get("counterspeech", ""),
                "prompt_task": args.task,
                "prompt_strategy": f"rag_{args.rag_mode}",
                "model_name": args.model,
                "ollama_native": bool(args.ollama_native),
                "thinking_disabled": bool(args.disable_thinking),
                "num_few_shot_examples": int(len(examples_df)),
                "rag_mode": args.rag_mode,
                "retrieval_query": json.dumps(used_queries, ensure_ascii=False),
                "retrieved_context": retrieved_context,
                "retrieved_contexts": json.dumps(ragas_payload["retrieved_contexts"], ensure_ascii=False),
                "retrieved_context_ids": json.dumps(ragas_payload["retrieved_context_ids"], ensure_ascii=False),
                "retrieved_context_metadata": json.dumps(ragas_payload["retrieved_context_metadata"], ensure_ascii=False),
                "retrieved_titles": json.dumps([str(x.get("title", "")) for x in retrieved_items], ensure_ascii=False),
                "retrieved_sources": json.dumps([str(x.get("_retriever", x.get("source", ""))) for x in retrieved_items], ensure_ascii=False),
                "num_retrieved_items": int(len(retrieved_items)),
                "retrieval_time_seconds": round(float(retrieval_time), 6) if retrieval_time is not None else None,
                "inference_time_seconds": round(float(inference_time), 6) if inference_time is not None else None,
                "total_time_seconds": round(float(total_time), 6) if total_time is not None else None,
                "error": None,
            })

        except Exception as e:
            total_time = time.perf_counter() - item_total_start
            output_row = row.to_dict()
            output_row.pop("aspects_list", None)
            output_row.update({
                **empty_prediction_output_fields(args.task),
                "pred_justification": None,
                "pred_counterspeech": None,
                "prompt_task": args.task,
                "prompt_strategy": f"rag_{args.rag_mode}",
                "model_name": args.model,
                "ollama_native": bool(args.ollama_native),
                "thinking_disabled": bool(args.disable_thinking),
                "num_few_shot_examples": None,
                "rag_mode": args.rag_mode,
                "retrieval_query": None,
                "retrieved_context": None,
                "retrieved_contexts": json.dumps([], ensure_ascii=False),
                "retrieved_context_ids": json.dumps([], ensure_ascii=False),
                "retrieved_context_metadata": json.dumps([], ensure_ascii=False),
                "retrieved_titles": json.dumps([], ensure_ascii=False),
                "retrieved_sources": json.dumps([], ensure_ascii=False),
                "num_retrieved_items": 0,
                "retrieval_time_seconds": round(float(retrieval_time), 6) if retrieval_time is not None else None,
                "inference_time_seconds": round(float(inference_time), 6) if inference_time is not None else None,
                "total_time_seconds": round(float(total_time), 6) if total_time is not None else None,
                "error": str(e),
            })

        rows_for_csv.append(output_row)
        benchmark_rows.append({
            "dataset_comment_id": comment_id,
            "success": output_row.get("error") is None,
            "retrieval_time_seconds": retrieval_time,
            "inference_time_seconds": inference_time,
            "total_time_seconds": total_time,
            "num_retrieved_items": output_row.get("num_retrieved_items", 0),
        })

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

    current_run_seconds = time.perf_counter() - benchmark_run_start
    final_benchmark_rows = benchmark_rows_from_dataframe(final_df, args.id_col)
    total_run_seconds = _sum([x.get("total_time_seconds") for x in final_benchmark_rows])
    if total_run_seconds is None:
        total_run_seconds = current_run_seconds
    benchmark_report = summarize_benchmark(
        benchmark_rows=final_benchmark_rows,
        total_run_seconds=total_run_seconds,
        args=args,
    )
    benchmark_report["current_run_seconds"] = round(float(current_run_seconds), 6)
    benchmark_report["retry_errors"] = bool(args.retry_errors)
    benchmark_report["current_run_attempted_classifications"] = len(benchmark_rows)

    benchmark_report_path = args.benchmark_report
    if benchmark_report_path is None:
        benchmark_report_path = str(Path(output_csv).with_suffix(".benchmark.json"))

    ensure_dir(benchmark_report_path)
    with open(benchmark_report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, ensure_ascii=False, indent=2)

    print(f"JSONL salvo em: {output_jsonl}")
    print(f"CSV salvo em: {output_csv}")
    print(f"Benchmark salvo em: {benchmark_report_path}")
    print(f"Tempo total: {benchmark_report['total_run_seconds']}s")
    print(f"Tempo médio retrieval: {benchmark_report['retrieval_time_seconds']['mean']}s")
    print(f"Tempo médio inferência: {benchmark_report['inference_time_seconds']['mean']}s")
    print(f"Tempo médio por classificação: {benchmark_report['classification_total_time_seconds']['mean']}s")


if __name__ == "__main__":
    main()
