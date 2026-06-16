#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_EXE="${PYTHON_EXE:-python}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-"$SCRIPT_DIR/data/final/output"}"
OUTPUT_DIR="${OUTPUT_DIR:-"$SCRIPT_DIR/data/final/metrics"}"
STRATEGY="${STRATEGY:-all}"
TASKS="${TASKS:-all}"
SKIP_MISSING="${SKIP_MISSING:-1}"

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
echo

for MODEL in "${MODELS[@]}"; do
  echo "==> Avaliando modelo: $MODEL"

  CMD=(
    "$PYTHON_EXE" "$EVAL_SCRIPT"
    --model "$MODEL"
    --predictions-dir "$PREDICTIONS_DIR"
    --output-dir "$OUTPUT_DIR"
    --strategy "$STRATEGY"
    --tasks $TASKS
  )

  if [[ "$SKIP_MISSING" == "1" ]]; then
    CMD+=(--skip-missing)
  fi

  "${CMD[@]}"
  echo
done

echo "Avaliacoes concluidas."
