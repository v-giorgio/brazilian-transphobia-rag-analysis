import os
import glob
import pandas as pd

INPUT_DIR = "../../../data/cleaned_comments"
OUTPUT_PATH = "./data/titles/video_titles.csv"

os.makedirs("./data/titles", exist_ok=True)

all_data = []

for file in glob.glob(os.path.join(INPUT_DIR, "*.csv")):
    df = pd.read_csv(file, usecols=["video_id", "video_title"])
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

df_unique = df_all.drop_duplicates(subset=["video_id", "video_title"])

df_unique.to_csv(OUTPUT_PATH, index=False)

print(f"Arquivo salvo em: {OUTPUT_PATH}")
print(f"Total de vídeos únicos: {len(df_unique)}")

# + remoção manual de partes desnecessárias dos títulos, como " | CNN NO PLURAL", " - #Shorts", etc.