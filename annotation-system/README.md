# Annotation Tool

## pt-br

Ferramenta web para anotação manual de comentários. O frontend usa React/Vite e as rotas serverless usam Supabase.

### Configuração

Crie `annotation-system/.env.local` a partir de `.env.example`:

```bash
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

### Execução

```bash
cd annotation-system
npm install
npm run dev
```

### Endpoints

```text
GET  /api/next?token=<annotator_token>
GET  /api/progress?token=<annotator_token>
POST /api/submit
```

## en

Web tool for manual comment annotation. The frontend uses React/Vite and the serverless routes use Supabase.

### Configuration

Create `annotation-system/.env.local` from `.env.example`:

```bash
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

### Run

```bash
cd annotation-system
npm install
npm run dev
```

### Endpoints

```text
GET  /api/next?token=<annotator_token>
GET  /api/progress?token=<annotator_token>
POST /api/submit
```
