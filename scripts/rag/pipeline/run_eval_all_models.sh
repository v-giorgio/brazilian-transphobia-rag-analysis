#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_EXE="${PYTHON_EXE:-python}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-"$SCRIPT_DIR/data/final/output"}"
OUTPUT_DIR="${OUTPUT_DIR:-"$SCRIPT_DIR/data/final/metrics"}"
STRATEGY="${STRATEGY:-all}"
TASKS="${TASKS:-all}"
SKIP_MISSING="${SKIP_MISSING:-1}"
RAGAS="${RAGAS:-0}"
RAGAS_ALL="${RAGAS_ALL:-0}"
FULL_EVAL="${FULL_EVAL:-0}"
RAGAS_LIMIT="${RAGAS_LIMIT:-}"
RAGAS_METRICS="${RAGAS_METRICS:-faithfulness context_precision context_recall}"
RAGAS_PROVIDER="${RAGAS_PROVIDER:-ollama}"
RAGAS_OLLAMA_BASE_URL="${RAGAS_OLLAMA_BASE_URL:-http://localhost:11434}"
RAGAS_OLLAMA_LLM="${RAGAS_OLLAMA_LLM:-qwen3:14b}"
RAGAS_OLLAMA_EMBEDDINGS="${RAGAS_OLLAMA_EMBEDDINGS:-nomic-embed-text}"
RAGAS_BATCH_SIZE="${RAGAS_BATCH_SIZE:-10}"
RAGAS_INCREMENTAL="${RAGAS_INCREMENTAL:-1}"

if [[ "$RAGAS_ALL" == "1" || "$FULL_EVAL" == "1" ]]; then
  RAGAS="1"
fi

MODELS=(
  "llama3:8b"
  "qwen3:14b"
  "gemma3:27b"
  "mistral-small:24b"
)

EVAL_SCRIPT="$SCRIPT_DIR/eval_classifier_llm_v6.py"

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "Arquivo de avaliacao nao encontrado: $EVAL_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$PREDICTIONS_DIR" ]]; then
  echo "Diretorio de predicoes CSV nao encontrado: $PREDICTIONS_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Avaliador: $EVAL_SCRIPT"
echo "Predicoes CSV: $PREDICTIONS_DIR"
echo "Saida: $OUTPUT_DIR"
echo "FULL_EVAL: $FULL_EVAL"
echo "RAGAS: $RAGAS"
echo "RAGAS_ALL: $RAGAS_ALL"
if [[ "$RAGAS" == "1" ]]; then
  echo "Metricas RAGAS: $RAGAS_METRICS"
  echo "Provedor RAGAS: $RAGAS_PROVIDER"
  if [[ "$RAGAS_PROVIDER" == "ollama" ]]; then
    echo "Ollama base URL: $RAGAS_OLLAMA_BASE_URL"
    echo "Ollama LLM avaliador: $RAGAS_OLLAMA_LLM"
    echo "Ollama embeddings: $RAGAS_OLLAMA_EMBEDDINGS"
  fi
  echo "RAGAS incremental: $RAGAS_INCREMENTAL"
  echo "RAGAS batch size: $RAGAS_BATCH_SIZE"
  if [[ -n "$RAGAS_LIMIT" ]]; then
    echo "Limite RAGAS: $RAGAS_LIMIT"
  fi
fi
echo

run_eval() {
  local MODEL="$1"
  local CURRENT_STRATEGY="$2"
  local ENABLE_RAGAS="$3"

  echo "==> Avaliando modelo: $MODEL | estrategia: $CURRENT_STRATEGY | ragas: $ENABLE_RAGAS"

  CMD=(
    "$PYTHON_EXE" "$EVAL_SCRIPT"
    --model "$MODEL"
    --predictions-dir "$PREDICTIONS_DIR"
    --output-dir "$OUTPUT_DIR"
    --strategy "$CURRENT_STRATEGY"
    --tasks $TASKS
  )

  if [[ "$SKIP_MISSING" == "1" ]]; then
    CMD+=(--skip-missing)
  fi

  if [[ "$ENABLE_RAGAS" == "1" ]]; then
    CMD+=(
      --ragas
      --ragas-metrics $RAGAS_METRICS
      --ragas-provider "$RAGAS_PROVIDER"
      --ragas-batch-size "$RAGAS_BATCH_SIZE"
    )

    if [[ "$RAGAS_INCREMENTAL" != "1" ]]; then
      CMD+=(--no-ragas-incremental)
    fi

    if [[ "$RAGAS_PROVIDER" == "ollama" ]]; then
      CMD+=(
        --ragas-ollama-base-url "$RAGAS_OLLAMA_BASE_URL"
        --ragas-ollama-llm "$RAGAS_OLLAMA_LLM"
        --ragas-ollama-embeddings "$RAGAS_OLLAMA_EMBEDDINGS"
      )
    fi

    if [[ -n "$RAGAS_LIMIT" ]]; then
      CMD+=(--ragas-limit "$RAGAS_LIMIT")
    fi
  fi

  "${CMD[@]}"
  echo
}

RAG_STRATEGIES=(rag_transcripts rag_wiki rag_both)

if [[ "$FULL_EVAL" == "1" ]]; then
  echo "==> Etapa 1/2: metricas classificatorias para todas as estrategias"
  for MODEL in "${MODELS[@]}"; do
    run_eval "$MODEL" "all" "0"
  done

  echo "==> Etapa 2/2: metricas RAGAS para estrategias RAG"
  for MODEL in "${MODELS[@]}"; do
    for CURRENT_STRATEGY in "${RAG_STRATEGIES[@]}"; do
      run_eval "$MODEL" "$CURRENT_STRATEGY" "1"
    done
  done
elif [[ "$RAGAS_ALL" == "1" ]]; then
  for MODEL in "${MODELS[@]}"; do
    for CURRENT_STRATEGY in "${RAG_STRATEGIES[@]}"; do
      run_eval "$MODEL" "$CURRENT_STRATEGY" "1"
    done
  done
else
  ENABLE_RAGAS="$RAGAS"
  for MODEL in "${MODELS[@]}"; do
    run_eval "$MODEL" "$STRATEGY" "$ENABLE_RAGAS"
  done
fi

echo "Avaliacoes concluidas."
