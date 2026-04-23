import os
from pathlib import Path
import re
import json
import time
import argparse
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ==========================================================
# CONFIG PADRÃO
# ==========================================================

DEFAULT_INPUT_PATH = "./data/temp_dataset/dataset_unificado.csv"
DEFAULT_OUTPUT_CSV = "./data/temp_dataset/output/classifier_llm_zero_shot_predictions.csv"
DEFAULT_OUTPUT_JSONL = "./data/temp_dataset/output/classifier_llm_zero_shot_predictions.jsonl"

# Pensado para uso local com Ollama
DEFAULT_BASE_URL = "http://localhost:11434/v1"
# DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
DEFAULT_MODEL = "llama3:8b"
# DEFAULT_MODEL = "gemma3:12b"

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

SYSTEM_PROMPT = """Você é uma assistente de anotação de comentários em português, especializada em classificação de discurso sobre pessoas trans.

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
- Não invente rótulos fora da lista.
- Se nenhum aspecto se aplicar claramente, retorne lista vazia.
- A justificativa deve explicar por que os rótulos escolhidos fazem sentido.
- O counterspeech deve responder de forma construtiva e respeitosa.
- Mesmo para comentários NEUTRAL, forneça um counterspeech breve que estimule diálogo respeitoso.
- Retorne APENAS JSON válido, sem texto extra.
"""

USER_TEMPLATE = """Classifique o comentário abaixo em esquema zero-shot.

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
"""


def ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_aspects(aspects: Any) -> List[str]:
    if aspects is None:
        return []
    if isinstance(aspects, str):
        aspects = [aspects]
    if not isinstance(aspects, list):
        return []

    clean = []
    seen = set()
    for a in aspects:
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


def classify_comment(
    client: OpenAI,
    model: str,
    video_title: str,
    comment: str,
    temperature: float,
    max_retries: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    user_prompt = USER_TEMPLATE.format(video_title=video_title, comment=comment)
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


def main():
    parser = argparse.ArgumentParser(description="Classificador zero-shot com LLM para comentários em português.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Caminho do CSV de entrada")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Caminho do CSV de saída")
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL, help="Caminho do JSONL incremental")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo Ollama/OpenAI-compatible")
    parser.add_argument("--api-key", default="ollama", help="Chave de API; para Ollama pode ser qualquer string")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL da API compatível com OpenAI")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperatura")
    parser.add_argument("--max-retries", type=int, default=6, help="Máximo de tentativas por comentário")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Espera base entre tentativas")
    parser.add_argument("--limit", type=int, default=None, help="Limite opcional de linhas")
    parser.add_argument("--id-col", default="dataset_comment_id", help="Nome da coluna de id")
    args = parser.parse_args()

    ensure_dir(args.output_csv)
    ensure_dir(args.output_jsonl)

    df = pd.read_csv(args.input)

    if args.id_col not in df.columns:
        raise ValueError(f"Coluna '{args.id_col}' não encontrada.")
    if "video_title" not in df.columns or "comment" not in df.columns:
        raise ValueError("O CSV precisa conter as colunas 'video_title' e 'comment'.")

    if args.limit is not None:
        df = df.head(args.limit).copy()

    processed_ids = load_processed_ids(args.output_jsonl, args.id_col)
    if processed_ids:
        df = df[~df[args.id_col].astype(str).isin(processed_ids)].copy()

    client = create_client(args.api_key, args.base_url)
    rows_for_csv = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classificando"):
        comment_id = str(row[args.id_col])
        video_title = str(row["video_title"])
        comment = str(row["comment"])

        try:
            pred = classify_comment(
                client=client,
                model=args.model,
                video_title=video_title,
                comment=comment,
                temperature=args.temperature,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            )

            output_row = row.to_dict()
            output_row.update({
                "pred_speech_class": pred["speech_class"],
                "pred_offensive_class": pred["offensive_class"],
                "pred_aspects": json.dumps(pred["aspects"], ensure_ascii=False),
                "pred_justification": pred["justification"],
                "pred_counterspeech": pred["counterspeech"],
                "prompt_strategy": "zero_shot",
                "model_name": args.model,
                "error": None,
            })

        except Exception as e:
            output_row = row.to_dict()
            output_row.update({
                "pred_speech_class": None,
                "pred_offensive_class": None,
                "pred_aspects": json.dumps([], ensure_ascii=False),
                "pred_justification": None,
                "pred_counterspeech": None,
                "prompt_strategy": "zero_shot",
                "model_name": args.model,
                "error": str(e),
            })

        rows_for_csv.append(output_row)

        with open(args.output_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    if os.path.exists(args.output_csv):
        old_df = pd.read_csv(args.output_csv)
        new_df = pd.DataFrame(rows_for_csv)
        final_df = pd.concat([old_df, new_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=[args.id_col], keep="last")
    else:
        final_df = pd.DataFrame(rows_for_csv)

    final_df.to_csv(args.output_csv, index=False, encoding="utf-8")

    print(f"JSONL salvo em: {args.output_jsonl}")
    print(f"CSV salvo em: {args.output_csv}")


if __name__ == "__main__":
    main()
