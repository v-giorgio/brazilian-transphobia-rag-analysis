from typing import Dict, Tuple


TASKS = ("speech", "offensive", "aspects")

SPEECH_LABEL_ORDER = ["HATEFUL", "SUPPORTIVE", "NEUTRAL"]
OFFENSIVE_LABEL_ORDER = ["OFFENSIVE", "NON_OFFENSIVE"]
ASPECT_LABEL_ORDER = [
    "RELIGION",
    "POLITICS",
    "GENDER_DENIAL",
    "HARRASSMENT",
    "THREATENING",
    "PSEUDOSCIENCE",
]

ALLOWED_SPEECH = set(SPEECH_LABEL_ORDER)
ALLOWED_OFFENSIVE = set(OFFENSIVE_LABEL_ORDER)
ALLOWED_ASPECTS = set(ASPECT_LABEL_ORDER)

TASK_TO_TRUE_COLUMN = {
    "speech": "speech_class",
    "offensive": "offensive_class",
    "aspects": "aspects",
}

TASK_TO_PRED_COLUMN = {
    "speech": "pred_speech_class",
    "offensive": "pred_offensive_class",
    "aspects": "pred_aspects",
}


COMMON_RULES = """Regras:
- Use o título do video apenas como contexto auxiliar. Classifique principalmente o comentário.
- Não invente rótulos fora da lista.
- Retorne APENAS JSON valido, sem texto extra.
"""

RAG_RULES = """Regras:
- Use o título do video e o contexto recuperado apenas como apoio. Classifique principalmente o comentário.
- Se o contexto recuperado não ajudar, baseie-se no comentário e no título do video.
- Não invente rótulos fora da lista.
- Retorne APENAS JSON válido, sem texto extra.
"""

FULL_SUFFIX = """Inclua também:
- justification: justificativa curta em português
- counterspeech: um contra-discurso curto em português
"""


SYSTEM_PROMPTS: Dict[str, str] = {
    "speech": f"""Você é uma assistente de anotação de comentários em português sobre pessoas trans.

Tarefa: classificar a atitude discursiva do comentário.

Rótulos permitidos:
- HATEFUL: comentário direcionado a pessoas trans ou identidades trans com teor discriminatório, excludente, desumanizante, de negação de identidade de gênero, ou de apoio a restrições, exclusão ou limitação de direitos.
- SUPPORTIVE: comentário direcionado a pessoas trans ou identidades trans com teor de apoio, defesa, acolhimento, empatia ou afirmação de direitos e identidades.
- NEUTRAL: comentário sem apoio ou hostilidade claros, apenas descritivo, ambíguo, irrelevante ou factual.

{COMMON_RULES}
""",
    "offensive": f"""Você é uma assistente de anotação de comentários em português sobre pessoas trans.

Tarefa: classificar se o comentário contém ofensividade explícita.

Rótulos permitidos:
- OFFENSIVE: contém palavras explícitas de insulto, humilhação, agressividade verbal, xingamento ou linguagem degradante.
- NON_OFFENSIVE: não contém ofensa explícita.

{COMMON_RULES}
""",
    "aspects": f"""Você é uma assistente de anotação de comentários em português sobre pessoas trans.

Tarefa: identificar aspectos discursivos presentes no comentário. A tarefa é multi-label: retorne zero ou mais rótulos.

Rótulos permitidos:
- GENDER_DENIAL: o comentário explicitamente contém negação, invalidação ou ridicularização da identidade de gênero.
- RELIGION: o comentário explicitamente contém uso de religião, pecado, Deus, igreja, diabo, missão divina etc.
- POLITICS: o comentário explicitamente contém referência a política, governo, eleições, direita/esquerda, parlamentares, ideologia, legislação.
- HARRASSMENT: o comentário explicitamente contém ataque pessoal, zombaria, humilhação, bullying ou assédio verbal dirigido.
- THREATENING: o comentário explicitamente contém ameaça explícita ou implícita, intimidação, incentivo à violência ou desejo de dano.
- PSEUDOSCIENCE: o comentário explicitamente contém uso de falsa ciência, biologicismo raso, psicologização indevida ou pseudoexplicações para invalidar pessoas trans.

{COMMON_RULES}
Se nenhum aspecto se aplicar claramente, retorne lista vazia.
""",
}

RAG_SYSTEM_PROMPTS = {
    task: prompt.replace(COMMON_RULES, RAG_RULES)
    for task, prompt in SYSTEM_PROMPTS.items()
}


JSON_SCHEMAS = {
    "speech": '''{
  "speech_class": "HATEFUL | SUPPORTIVE | NEUTRAL",
  "speech_scores": {"HATEFUL": 0.0, "SUPPORTIVE": 0.0, "NEUTRAL": 0.0}
}''',
    "offensive": '''{
  "offensive_class": "OFFENSIVE | NON_OFFENSIVE",
  "offensive_scores": {"OFFENSIVE": 0.0, "NON_OFFENSIVE": 0.0}
}''',
    "aspects": '''{
  "aspects": ["RELIGION", "POLITICS", "GENDER_DENIAL", "HARRASSMENT", "THREATENING", "PSEUDOSCIENCE"],
  "aspect_scores": {"RELIGION": 0.0, "POLITICS": 0.0, "GENDER_DENIAL": 0.0, "HARRASSMENT": 0.0, "THREATENING": 0.0, "PSEUDOSCIENCE": 0.0}
}''',
}

FULL_JSON_SCHEMAS = {
    task: schema[:-2] + ''',
  "justification": "texto curto em português",
  "counterspeech": "texto curto em português"
}'''
    for task, schema in JSON_SCHEMAS.items()
}


ZERO_USER_TEMPLATE = """Classifique o comentário abaixo em esquema zero-shot.

título do video:
{video_title}

comentário:
{comment}

Responda APENAS com JSON neste formato:
{json_schema}
"""

FEW_USER_TEMPLATE = """Classifique o comentário abaixo em esquema few-shot.

A seguir há exemplos anotados. Use-os apenas como referência de decisão e formato.

{few_shot_examples}

Agora classifique o comentário alvo.

título do video:
{video_title}

comentário:
{comment}

Responda APENAS com JSON neste formato:
{json_schema}
"""

RAG_USER_TEMPLATE = """Classifique o comentário abaixo em esquema few-shot com RAG.

A seguir há exemplos anotados. Use-os apenas como referência de decisão e formato.

{few_shot_examples}

Agora veja o contexto recuperado.
Use esse contexto apenas como apoio. Se o contexto nao ajudar, baseie-se no comentário e no título do video.

Contexto recuperado:
{retrieved_context}

Agora classifique o comentário alvo.

título do video:
{video_title}

comentário:
{comment}

Responda APENAS com JSON neste formato:
{json_schema}
"""


def normalize_task(task: str) -> str:
    task = str(task).strip().lower()
    if task not in TASKS:
        raise ValueError(f"Tarefa invalida: {task}. Use: {', '.join(TASKS)}.")
    return task


def get_prompt_pair(task: str, strategy: str, fast_prompt: bool) -> Tuple[str, str]:
    task = normalize_task(task)
    strategy = str(strategy).strip().lower()
    system_prompts = RAG_SYSTEM_PROMPTS if strategy == "rag" else SYSTEM_PROMPTS
    system_prompt = system_prompts[task]

    if not fast_prompt:
        system_prompt = f"{system_prompt}\n{FULL_SUFFIX}"
        json_schema = FULL_JSON_SCHEMAS[task]
    else:
        json_schema = JSON_SCHEMAS[task]

    if strategy == "zero_shot":
        user_template = ZERO_USER_TEMPLATE
    elif strategy == "few_shot":
        user_template = FEW_USER_TEMPLATE
    elif strategy == "rag":
        user_template = RAG_USER_TEMPLATE
    else:
        raise ValueError(f"Estrategia invalida: {strategy}")

    escaped_schema = json_schema.replace("{", "{{").replace("}", "}}")
    return system_prompt, user_template.replace("{json_schema}", escaped_schema)
