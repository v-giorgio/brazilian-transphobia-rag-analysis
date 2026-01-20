from pathlib import Path
import pandas as pd

CSV_DIR = Path("../../data/raw_comments")

total_geral = 0

print("Contagem de registros por arquivo:\n")

for csv_file in CSV_DIR.glob("*.csv"):
    try:
        df = pd.read_csv(csv_file)
        count = len(df)
        total_geral += count
        print(f" - {csv_file.name}: {count} registros")
    except Exception as e:
        print(f"Erro ao ler {csv_file.name}: {e}")

print("\n==============================")
print(f"TOTAL GERAL: {total_geral} registros")
