# Data Processing

## pt-br

Scripts para coleta de comentários via YouTube Data API, limpeza textual e contagem por canal.

### Configuração

Na raiz do repositório, crie `.env` a partir de `.env.example`:

```bash
YOUTUBE_API_KEY=your_youtube_data_api_key
```

### Execução

```bash
python scripts/etl/extract/collect_data.py
python scripts/etl/transform/clean_data.py
python scripts/etl/count_records_by_channel.py
```

### Saídas

- `data/raw_comments/`: comentários coletados.
- `data/manual_selection/`: amostra selecionada para curadoria.
- `data/cleaned_comments/`: comentários limpos.
- `scripts/etl/output/`: contagens por canal.

## en

Scripts for collecting comments through the YouTube Data API, cleaning text, and counting records by channel.

### Configuration

At the repository root, create `.env` from `.env.example`:

```bash
YOUTUBE_API_KEY=your_youtube_data_api_key
```

### Run

```bash
python scripts/etl/extract/collect_data.py
python scripts/etl/transform/clean_data.py
python scripts/etl/count_records_by_channel.py
```

### Outputs

- `data/raw_comments/`: collected comments.
- `data/manual_selection/`: selected sample for curation.
- `data/cleaned_comments/`: cleaned comments.
- `scripts/etl/output/`: channel-level counts.
