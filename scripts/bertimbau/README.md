# Supervised Learning

## pt-br

Scripts para preparar os dados e treinar modelos supervisionados: baseline com BERTimbau congelado + SVM e fine-tuning de BERTimbau.

### Preparação dos dados

```bash
python scripts/bertimbau/1_prepare_bertimbau_data.py
```

### Baseline SVM com BERTimbau congelado

```bash
python scripts/bertimbau/2_train_bertimbau_frozen_baseline.py --task speech --input-format title_comment
python scripts/bertimbau/2_train_bertimbau_frozen_baseline.py --task offensive --input-format title_comment
python scripts/bertimbau/2_train_bertimbau_frozen_baseline.py --task aspects --input-format title_comment
```

### Fine-tuning BERTimbau

```bash
python scripts/bertimbau/3_train_bertimbau_speech_v2.py --input-format title_comment
python scripts/bertimbau/4_train_bertimbau_offensive_v3.py --input-format title_comment
python scripts/bertimbau/5_train_bertimbau_aspects_v3.py --input-format title_comment
```

### Saídas

- `scripts/bertimbau/pipeline/data/`: divisões de treino, validação e teste.
- `scripts/bertimbau/pipeline/outputs/`: métricas e predições.
- `scripts/bertimbau/pipeline/models/`: modelos salvos quando gerados localmente.

## en

Scripts for preparing data and training supervised models: frozen BERTimbau + SVM baseline and BERTimbau fine-tuning.

### Data preparation

```bash
python scripts/bertimbau/1_prepare_bertimbau_data.py
```

### Frozen BERTimbau + SVM baseline

```bash
python scripts/bertimbau/2_train_bertimbau_frozen_baseline.py --task speech --input-format title_comment
python scripts/bertimbau/2_train_bertimbau_frozen_baseline.py --task offensive --input-format title_comment
python scripts/bertimbau/2_train_bertimbau_frozen_baseline.py --task aspects --input-format title_comment
```

### BERTimbau fine-tuning

```bash
python scripts/bertimbau/3_train_bertimbau_speech_v2.py --input-format title_comment
python scripts/bertimbau/4_train_bertimbau_offensive_v3.py --input-format title_comment
python scripts/bertimbau/5_train_bertimbau_aspects_v3.py --input-format title_comment
```

### Outputs

- `scripts/bertimbau/pipeline/data/`: train, validation, and test splits.
- `scripts/bertimbau/pipeline/outputs/`: metrics and predictions.
- `scripts/bertimbau/pipeline/models/`: locally generated saved models.
