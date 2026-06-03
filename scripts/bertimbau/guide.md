# Guia da pipeline BERTimbau

Este diretório contém os scripts para preparar os dados e executar os baselines com BERTimbau. Os modelos treinados e os arquivos de saída em `pipeline/models/` e `pipeline/outputs/` podem ser regenerados e não devem ser versionados.

Execute os comandos a partir deste diretório:

```bash
cd scripts/bertimbau
```

## 1. Preparar os dados

Este passo lê `data/dataset_final.csv`, valida os rótulos e gera os splits de treino, validação e teste para `speech`, `offensive` e `aspects`.

```bash
python 1_prepare_bertimbau_data.py
```

Saídas principais:

- `pipeline/data/master/clean_dataset.csv`
- `pipeline/data/speech/{train,val,test}.csv`
- `pipeline/data/offensive/{train,val,test}.csv`
- `pipeline/data/aspects/{train,val,test}.csv`
- `pipeline/data/metadata/*.json`

## 2. BERTimbau congelado + SVM

Este baseline usa o BERTimbau apenas como extrator de embeddings e treina um
SVM linear calibrado sobre as representações. Não há fine-tuning do BERTimbau.

Argumentos principais:

- `--task`: `speech`, `offensive` ou `aspects`
- `--input-format`: `comment_only` ou `title_comment`

```bash
python 2_train_bertimbau_frozen_baseline.py --task speech --input-format comment_only
python 2_train_bertimbau_frozen_baseline.py --task speech --input-format title_comment
```

```bash
python 2_train_bertimbau_frozen_baseline.py --task offensive --input-format comment_only
python 2_train_bertimbau_frozen_baseline.py --task offensive --input-format title_comment
```

```bash
python 2_train_bertimbau_frozen_baseline.py --task aspects --input-format comment_only
python 2_train_bertimbau_frozen_baseline.py --task aspects --input-format title_comment
```

## 3. Fine-tuning para atitude discursiva

Treina BERTimbau para `speech_class` (`HATEFUL`, `SUPPORTIVE`, `NEUTRAL`).

```bash
python 3_train_bertimbau_speech_v2.py `
  --input-format comment_only `
  --learning-rate 2e-5 `
  --num-epochs 5 `
  --train-batch-size 16 `
  --eval-batch-size 16 `
  --class-weight-mode balanced
```

```bash
python 3_train_bertimbau_speech_v2.py `
  --input-format title_comment `
  --learning-rate 2e-5 `
  --num-epochs 5 `
  --train-batch-size 16 `
  --eval-batch-size 16 `
  --class-weight-mode balanced
```

## 4. Fine-tuning para ofensividade

Treina BERTimbau para `offensive_class` (`OFFENSIVE`, `NON_OFFENSIVE`).

```bash
python 4_train_bertimbau_offensive_v3.py `
  --input-format comment_only `
  --learning-rate 2e-5 `
  --num-epochs 5 `
  --train-batch-size 16 `
  --eval-batch-size 16 `
  --class-weight-mode balanced
```

```bash
python 4_train_bertimbau_offensive_v3.py `
  --input-format title_comment `
  --learning-rate 2e-5 `
  --num-epochs 5 `
  --train-batch-size 16 `
  --eval-batch-size 16 `
  --class-weight-mode balanced
```

## 5. Fine-tuning para aspectos discursivos

Treina BERTimbau para `aspects`, uma tarefa multi-label. Os thresholds são
calibrados por rótulo no conjunto de validação.

```bash
python 5_train_bertimbau_aspects_v3.py `
  --input-format comment_only `
  --learning-rate 2e-5 `
  --num-epochs 5 `
  --train-batch-size 16 `
  --eval-batch-size 16
```

```bash
python 5_train_bertimbau_aspects_v3.py `
  --input-format title_comment `
  --learning-rate 2e-5 `
  --num-epochs 5 `
  --train-batch-size 16 `
  --eval-batch-size 16
```

Se o treinamento já terminou e falta apenas regenerar predições e métricas,
use `--predict-only`:

```bash
python 5_train_bertimbau_aspects_v3.py --input-format comment_only --predict-only
python 5_train_bertimbau_aspects_v3.py --input-format title_comment --predict-only
```

## Saídas esperadas

Cada execução cria uma pasta específica em `pipeline/outputs/`, por exemplo:

- `speech_frozen_comment_only`
- `speech_finetuned_title_comment`
- `offensive_finetuned_comment_only`
- `aspects_finetuned_title_comment`

Arquivos comuns de saída:

- `metrics_val.json`
- `metrics_test.json`
- `val_predictions.csv`
- `test_predictions.csv`
- `metrics_val_predictions.json`
- `metrics_test_predictions.json`
- `confusion_matrix_test.csv`
- `training_summary.json`

Para `aspects`, a matriz de confusão é salva por rótulo, com colunas `tn`,
`fp`, `fn` e `tp`.
