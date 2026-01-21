from dotenv import load_dotenv
import requests
import pandas as pd
import time
import os
from pathlib import Path

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ================= CONFIG =================

BASE_URL = "https://www.googleapis.com/youtube/v3"

OUTPUT_DIR = Path("../../../data/raw_comments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optou-se por coletar um número limitado de comentários por vídeo, 
# priorizando a diversidade de vídeos analisados, de forma a reduzir viés de correlação entre comentários e 
# aumentar a variabilidade temática do conjunto de dados.
MAX_VIDEOS_PER_CHANNEL = 50
MAX_COMMENTS_PER_VIDEO = 100

QUERY_TERMS = [
    # Termos identitários
    "trans",
    "transexual",
    "transgênero",
    "transgenero",
    "travesti",
    "mulher trans",
    "homem trans",
    "pessoa trans",
    "pessoas trans",

    # Direitos e reconhecimento
    "identidade de gênero",
    "nome social",

    # Discurso público e político
    "ideologia de gênero",
    "lei trans",
    "cota trans",

    # Termos relacionados a preconceito
    "transfobia",
    "transfóbico",
    "transfobico",
]


CHANNEL_IDS = {
    # "g1globo": "UCaGmdJSSiR7fkh2A-c6emsA",
    # "cnn": "UCvdwhh_fDyWccR42-rReZLw",
    # "sbt": "UC376n347Ob5Lwzq2WGzF1AA",
    # "folha": "UCmYuiV9wAR3i8p19Ygc7izg",
    # "bbc": "UCthbIFAxbXTTQEC7EcQvP1Q",
    # "metropoles": "UC9mdw2mmn49ZuqGOpSri7Fw",
    # "uol": "UCVFbXI6Gu8U2f9Gjtxw4A-Q",
}

# ================= API HELPERS =================

def search_videos(channel_id):
    """
    Busca vídeos do canal relacionados aos termos,
    retorna lista única de video_ids (limitada).
    """
    video_ids = []
    seen = set()

    for query in QUERY_TERMS:
        next_page = None

        while True:
            params = {
                "part": "snippet",
                "q": query,
                "channelId": channel_id,
                "type": "video",
                "maxResults": 50,
                "relevanceLanguage": "pt",
                "pageToken": next_page,
                "key": YOUTUBE_API_KEY,
            }

            r = requests.get(f"{BASE_URL}/search", params=params)
            if r.status_code == 403:
                print("Quota atingida durante search.")
                return video_ids

            r.raise_for_status()
            data = r.json()

            for item in data.get("items", []):
                vid = item["id"]["videoId"]
                if vid not in seen:
                    seen.add(vid)
                    video_ids.append(vid)

                if len(video_ids) >= MAX_VIDEOS_PER_CHANNEL:
                    return video_ids

            next_page = data.get("nextPageToken")
            if not next_page:
                break

            time.sleep(0.1)

    return video_ids


def get_video_details(video_ids):
    videos = []

    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]

        params = {
            "part": "snippet,statistics",
            "id": ",".join(chunk),
            "key": YOUTUBE_API_KEY,
        }

        r = requests.get(f"{BASE_URL}/videos", params=params)
        if r.status_code == 403:
            print("Quota atingida durante videos.list.")
            return videos

        r.raise_for_status()

        for v in r.json().get("items", []):
            videos.append({
                "video_id": v["id"],
                "video_title": v["snippet"]["title"],
                "published_at": v["snippet"]["publishedAt"],
                "view_count": int(v["statistics"].get("viewCount", 0)),
                "comment_count": int(v["statistics"].get("commentCount", 0)),
            })

        time.sleep(0.1)

    return videos


def get_comments(video_id):
    comments = []
    next_page = None

    while len(comments) < MAX_COMMENTS_PER_VIDEO:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "pageToken": next_page,
            "textFormat": "plainText",
            "key": YOUTUBE_API_KEY,
        }

        r = requests.get(f"{BASE_URL}/commentThreads", params=params)
        if r.status_code == 403:
            print("Quota atingida durante comments.")
            break

        r.raise_for_status()
        data = r.json()

        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(snippet["textDisplay"])
            if len(comments) >= MAX_COMMENTS_PER_VIDEO:
                break

        next_page = data.get("nextPageToken")
        if not next_page:
            break

        time.sleep(0.1)

    return comments


# ================= PIPELINE =================

def process_channel(name, channel_id):
    print(f"\nCanal: {name}")

    video_ids = search_videos(channel_id)
    print(f"{len(video_ids)} vídeos encontrados")

    videos = get_video_details(video_ids)

    rows = []

    for v in videos:
        comments = get_comments(v["video_id"])
        for c in comments:
            rows.append({
                "source": name,
                "video_id": v["video_id"],
                "video_title": v["video_title"],
                "comment": c,
            })

    df = pd.DataFrame(rows)

    output_path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"{len(df)} comentários salvos em {output_path}")


# ================= MAIN =================

if __name__ == "__main__":
    for name, channel_id in CHANNEL_IDS.items():
        process_channel(name, channel_id)
