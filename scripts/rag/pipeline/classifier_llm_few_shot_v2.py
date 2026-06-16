import os
from pathlib import Path
import re
import json
import time
import ast
import argparse
from typing import Any, Dict, List, Set

import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm

import prompt_constants as pc

DEFAULT_TEST_DATA_DIR = "./data/final/test_data"
DEFAULT_OUTPUT_DIR = "./data/final/output"

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen3:14b"
# llama3.1:70b
# mistral-small:24b
# gemma3:27b
# deepseek-r1:32b


def ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model).strip()).strip("_") or "model"


def default_input_path(task: str) -> str:
    task = pc.normalize_task(task)
    return f"{DEFAULT_TEST_DATA_DIR}/{task}/test.csv"


def default_examples_input_path(task: str) -> str:
    task = pc.normalize_task(task)
    return f"{DEFAULT_TEST_DATA_DIR}/{task}/train.csv"


def default_output_paths(output_dir: str, task: str, model: str) -> tuple[str, str]:
    task = pc.normalize_task(task)
    model_slug = sanitize_model_name(model)
    base = Path(output_dir) / "few_shot" / task / f"predictions_{model_slug}"
    return f"{base}.csv", f"{base}.jsonl"


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
    response_json = json.dumps(response, ensure_ascii=False, indent=4)

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
    builder = build_example_text_fast if fast_prompt else build_example_text
    blocks = []
    for _, row in examples_df.iterrows():
        blocks.append(builder(row, task))
    return "\n\n".join(blocks)


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


def classify_comment(
    client: OpenAI,
    model: str,
    task: str,
    video_title: str,
    comment: str,
    few_shot_examples: str,
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
    system_prompt, user_template = pc.get_prompt_pair(task, strategy="few_shot", fast_prompt=fast_prompt)
    user_prompt = user_template.format(
        video_title=video_title,
        comment=comment,
        few_shot_examples=few_shot_examples,
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
                wait = sleep_seconds * (2 ** attempt)
                time.sleep(wait)

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
            wait = sleep_seconds * (2 ** attempt)
            time.sleep(wait)

    raise RuntimeError(f"Falha após {max_retries} tentativas: {last_error}")


def _mean(values: List[Any]) -> Any:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _median(values: List[Any]) -> Any:
    clean = sorted(float(v) for v in values if v is not None and not pd.isna(v))
    if not clean:
        return None
    n = len(clean)
    mid = n // 2
    if n % 2 == 1:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2


def _sum(values: List[Any]) -> Any:
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
            "inference_time_seconds": row.get("inference_time_seconds"),
            "total_time_seconds": row.get("total_time_seconds"),
            "num_few_shot_examples": _value_or_none(row.get("num_few_shot_examples")),
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

    inference_times = [x.get("inference_time_seconds") for x in benchmark_rows]
    total_times = [x.get("total_time_seconds") for x in benchmark_rows]
    few_shot_counts = [x.get("num_few_shot_examples") for x in benchmark_rows if x.get("num_few_shot_examples") is not None]

    return {
        "model": args.model,
        "task": args.task,
        "prompt_strategy": "few_shot",
        "fast_prompt": bool(args.fast_prompt),
        "ollama_native": bool(args.ollama_native),
        "thinking_disabled": bool(args.disable_thinking),
        "limit": args.limit,
        "speech_examples_per_class": args.speech_examples_per_class,
        "offensive_examples_per_class": args.offensive_examples_per_class,
        "aspect_examples_per_label": args.aspect_examples_per_label,
        "attempted_classifications": attempted,
        "successful_classifications": successful,
        "failed_classifications": failed,
        "total_run_seconds": round(float(total_run_seconds), 6),
        "average_total_time_per_attempt_seconds": round(float(total_run_seconds / attempted), 6) if attempted else None,
        "average_total_time_per_success_seconds": round(float(total_run_seconds / successful), 6) if successful else None,
        "inference_time_seconds": {
            "mean": round(_mean(inference_times), 6) if _mean(inference_times) is not None else None,
            "median": round(_median(inference_times), 6) if _median(inference_times) is not None else None,
        },
        "classification_total_time_seconds": {
            "mean": round(_mean(total_times), 6) if _mean(total_times) is not None else None,
            "median": round(_median(total_times), 6) if _median(total_times) is not None else None,
        },
        "few_shot_examples": {
            "mean": round(_mean(few_shot_counts), 6) if _mean(few_shot_counts) is not None else None,
            "median": round(_median(few_shot_counts), 6) if _median(few_shot_counts) is not None else None,
        },
        "per_item": [
            {
                **row,
                "inference_time_seconds": _round_or_none(row.get("inference_time_seconds")),
                "total_time_seconds": _round_or_none(row.get("total_time_seconds")),
            }
            for row in benchmark_rows
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Classificador few-shot com LLM para comentários em português.")
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
    parser.add_argument("--model", "--llm-model", dest="model", default=DEFAULT_MODEL, help="Modelo Ollama/OpenAI-compatible")
    parser.add_argument("--api-key", default="ollama", help="Chave de API; para Ollama pode ser qualquer string")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL da API compatível com OpenAI")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperatura")
    parser.add_argument("--max-retries", type=int, default=6, help="Máximo de tentativas por comentário")
    parser.add_argument("--sleep-seconds", type=float, default=4.0, help="Espera base entre tentativas")
    parser.add_argument("--limit", type=int, default=None, help="Limite opcional de linhas")
    parser.add_argument("--id-col", default="dataset_comment_id", help="Nome da coluna de id")
    parser.add_argument("--speech-examples-per-class", type=int, default=10, help="Exemplos por classe de speech")
    parser.add_argument("--offensive-examples-per-class", type=int, default=10, help="Exemplos por classe de offensive")
    parser.add_argument("--aspect-examples-per-label", type=int, default=5, help="Exemplos por label de aspects")
    parser.add_argument("--fast-prompt", action="store_true", default=False, help="Usa prompt curto sem justification/counterspeech.")
    parser.add_argument("--full-prompt", action="store_false", dest="fast_prompt", help="Usa prompt completo com justification/counterspeech.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Máximo de tokens gerados")
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
        default_csv, default_jsonl = default_output_paths(args.output_dir, args.task, args.model)
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
    rows_for_csv = []
    benchmark_rows = []
    benchmark_run_start = time.perf_counter()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classificando"):
        comment_id = str(row[args.id_col])
        video_title = str(row["video_title"])
        comment = str(row["comment"])

        item_total_start = time.perf_counter()
        inference_time = None
        total_time = None
        num_few_shot_examples = None

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
            num_few_shot_examples = int(len(examples_df))

            inference_start = time.perf_counter()
            pred = classify_comment(
                client=client,
                model=args.model,
                task=args.task,
                video_title=video_title,
                comment=comment,
                few_shot_examples=few_shot_examples,
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
                "prompt_strategy": "few_shot",
                "model_name": args.model,
                "ollama_native": bool(args.ollama_native),
                "thinking_disabled": bool(args.disable_thinking),
                "num_few_shot_examples": num_few_shot_examples,
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
                "prompt_strategy": "few_shot",
                "model_name": args.model,
                "ollama_native": bool(args.ollama_native),
                "thinking_disabled": bool(args.disable_thinking),
                "num_few_shot_examples": num_few_shot_examples,
                "inference_time_seconds": round(float(inference_time), 6) if inference_time is not None else None,
                "total_time_seconds": round(float(total_time), 6) if total_time is not None else None,
                "error": str(e),
            })

        rows_for_csv.append(output_row)
        benchmark_rows.append({
            "dataset_comment_id": comment_id,
            "success": output_row.get("error") is None,
            "inference_time_seconds": inference_time,
            "total_time_seconds": total_time,
            "num_few_shot_examples": num_few_shot_examples,
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
    print(f"Tempo médio inferência: {benchmark_report['inference_time_seconds']['mean']}s")
    print(f"Tempo médio por classificação: {benchmark_report['classification_total_time_seconds']['mean']}s")


if __name__ == "__main__":
    main()
