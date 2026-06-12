# Guia da Pipeline LLM/RAG

Execute os comandos a partir de `scripts/rag/pipeline`.

```bash
cd scripts/rag/pipeline
```

As pipelines avaliam uma tarefa por execucao:

- `speech`: atitude discursiva (`HATEFUL`, `SUPPORTIVE`, `NEUTRAL`);
- `offensive`: ofensividade explicita (`OFFENSIVE`, `NON_OFFENSIVE`);
- `aspects`: aspectos discursivos multi-label.

## Entradas e saidas

Por padrao, cada tarefa usa:

```text
data/final/test_data/<task>/test.csv
data/final/test_data/<task>/train.csv
```

O `test.csv` e o conjunto avaliado. O `train.csv` e usado como fonte de exemplos nos modos `few-shot` e RAG.

As predicoes sao salvas em:

```text
data/zero_shot/<task>/predictions_<model_slug>.csv
data/zero_shot/<task>/predictions_<model_slug>.jsonl

data/few_shot/<task>/predictions_<model_slug>.csv
data/few_shot/<task>/predictions_<model_slug>.jsonl

data/rag/<task>/predictions_<model_slug>_<rag_mode>.csv
data/rag/<task>/predictions_<model_slug>_<rag_mode>.jsonl
```

Exemplo: `qwen3:14b` vira `qwen3_14b` no nome do arquivo.

```
$ ollama list
NAME                       ID              SIZE      MODIFIED           
llama3:8b                  365c0bd3c000    4.7 GB    2 minutes ago
qwen3:14b                  bdbd181c33f2    9.3 GB    4 weeks ago     
gemma3:27b                 a418f5838eaf    17 GB     6 minutes ago         
mistral-small:24b          8039dd90c113    14 GB     10 minutes ago          
```

## Zero-shot

```bash
python classifier_llm_zero_shot_v2.py \
  --task speech \
  --model mistral-small:24b
```

Para testar poucas linhas:

```bash
python classifier_llm_zero_shot_v2.py \
  --task speech \
  --model qwen3:14b \
  --limit 5
```

## Few-shot

```bash
python classifier_llm_few_shot_v2.py \
  --task offensive \
  --model gemma3:27b
```

Para sobrescrever a fonte de exemplos:

```bash
python classifier_llm_few_shot_v2.py \
  --task offensive \
  --model qwen3:14b \
  --examples-input ./data/final/test_data/offensive/train.csv
```

Usado no trabalho:

```bash
python classifier_llm_few_shot_v2.py \
  --task speech \
  --model qwen3:14b \
  --speech-examples-per-class 3 \
  --max-retries 2 

python classifier_llm_few_shot_v2.py \
  --task offensive \
  --model qwen3:14b \
  --offensive-examples-per-class 3 \
  --max-retries 2 

python classifier_llm_few_shot_v2.py \
  --task aspects \
  --model qwen3:14b \
  --aspect-examples-per-label 3 \
  --max-retries 2 
```

## RAG

Modos disponiveis:

- `transcripts`: transcricoes dos videos;
- `wiki`: documentos externos do dominio;
- `both`: transcricoes e documentos.

[x] qwen
  [x] speech
    [x] transcripts
    [x] wiki
    [x] both
  [x] offensive
    [x] transcripts
    [x] wiki
    [x] both
  [x] aspects
    [x] transcripts
    [x] wiki
    [x] both

[ ] llama
  [x] speech
    [x] transcripts
    [x] wiki - erro
    [x] both - erro
  [x] offensive
    [x] transcripts
    [x] wiki
    [x] both
  [x] aspects
    [x] transcripts
    [x] wiki
    [x] both

[x] gemma
  [x] speech
    [x] transcripts
    [x] wiki
    [x] both - erro
  [x] offensive
    [x] transcripts
    [x] wiki
    [x] both - erro
  [x] aspects
    [x] transcripts - erro
    [x] wiki - erro
    [x] both - erro

[x] mistral
  [x] speech
    [x] transcripts
    [x] wiki
    [x] both
  [x] offensive
    [x] transcripts
    [x] wiki
    [x] both
  [x] aspects
    [x] transcripts
    [x] wiki
    [x] both

```bash
python classifier_llm_rag_v12.py \
  --task speech \
  --rag-mode both \
  --model mistral-small:24b \
  --speech-examples-per-class 3 \
  --max-retries 2
```

```bash
python classifier_llm_rag_v12.py \
  --task offensive \
  --rag-mode both \
  --model mistral-small:24b \
  --offensive-examples-per-class 3 \
  --max-retries 2
```

```bash
python classifier_llm_rag_v12.py \
  --task aspects \
  --rag-mode both \
  --model mistral-small:24b \
  --aspect-examples-per-label 3 \
  --max-retries 2
```

## Avaliacao

Avaliar todas as estrategias de uma tarefa para um modelo:

```bash
python eval_classifier_llm_v6.py \
  --tasks speech \
  --strategy all \
  --model qwen3:14b
```

Avaliar uma estrategia especifica:

```bash
python eval_classifier_llm_v6.py \
  --tasks aspects \
  --strategy rag_wiki \
  --model qwen3:14b
```

Avaliar com RAGAS quando o arquivo contem campos de contexto recuperado:

```bash
python eval_classifier_llm_v6.py \
  --tasks speech \
  --strategy rag_wiki \
  --model qwen3:14b \
  --ragas
```

## Parametros uteis

```bash
--input caminho/customizado/test.csv
--examples-input caminho/customizado/train.csv
--output-csv caminho/customizado/predictions.csv
--output-jsonl caminho/customizado/predictions.jsonl
--base-url http://localhost:11434/v1
--full-prompt
--enable-thinking
--openai-compatible
```
