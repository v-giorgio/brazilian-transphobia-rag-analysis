## Endpoints

```bash
GET /api/health
GET /api/next?token=XXXX
POST /api/submit
GET /api/progress?token=XXXX
```

## Deploy vercel
```bash
npm install
vercel login
vercel
```


curl -X POST https://annotation-system-wine.vercel.app/api/submit \
  -H "Content-Type: application/json" \
  -d '{
    "token": "TKN003",
    "assignment_id": 1210,
    "comment_id": 1,
    "speech_class": "HATEFUL",
    "offensive_class": "NON_OFFENSIVE",
    "aspects": ["POLITICS"]
  }'

curl -X GET https://annotation-system-wine.vercel.app/api/next?token=TKN003