import os
import glob
import pandas as pd
import subprocess
import whisper
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_DIR = "../../data/cleaned_comments"
OUTPUT_FILE = "./data/transcriptions/video_transcriptions.csv"
TEMP_AUDIO_DIR = "./data/transcriptions/temp_audio"
WHISPER_MODEL = "medium"  # use "large" se tiver GPU forte
LANGUAGE = "pt"

os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
os.makedirs("./data", exist_ok=True)

# =========================
# 1. COLETAR VIDEO_IDS ÚNICOS
# =========================
print("Coletando video_ids únicos...")

all_video_ids = set()
csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

for file in csv_files:
    df = pd.read_csv(file)
    if "video_id" in df.columns:
        all_video_ids.update(df["video_id"].dropna().unique())

video_ids = list(all_video_ids)
print(f"Total de vídeos únicos encontrados: {len(video_ids)}")

# =========================
# 2. CARREGAR TRANSCRIÇÕES JÁ EXISTENTES
# =========================
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    processed_ids = set(existing_df["video_id"].unique())
    print(f"Já existem {len(processed_ids)} vídeos processados.")
else:
    existing_df = pd.DataFrame(columns=["video_id", "transcription"])
    processed_ids = set()

# Filtrar apenas vídeos ainda não processados
video_ids_to_process = [vid for vid in video_ids if vid not in processed_ids]

print(f"Vídeos restantes para processar: {len(video_ids_to_process)}")

# =========================
# 3. CARREGAR WHISPER
# =========================
print("Carregando modelo Whisper...")
model = whisper.load_model(WHISPER_MODEL)

# =========================
# 4. PROCESSAR VÍDEOS
# =========================
for video_id in tqdm(video_ids_to_process):

    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        output_template = os.path.join(TEMP_AUDIO_DIR, f"{video_id}.%(ext)s")

        # Baixar e converter para wav 16k mono
        subprocess.run([
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--output", output_template,
            video_url
        ], check=True)

        audio_path = os.path.join(TEMP_AUDIO_DIR, f"{video_id}.wav")

        # Transcrever
        result = model.transcribe(audio_path, language=LANGUAGE)
        transcription = result["text"].strip()

        # Salvar imediatamente
        new_row = pd.DataFrame([{
            "video_id": video_id,
            "transcription": transcription
        }])

        new_row.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=not os.path.exists(OUTPUT_FILE),
            index=False
        )

        # Remover áudio temporário
        os.remove(audio_path)

    except Exception as e:
        print(f"\nErro no vídeo {video_id}: {e}")
        continue

print("\nProcessamento concluído.")