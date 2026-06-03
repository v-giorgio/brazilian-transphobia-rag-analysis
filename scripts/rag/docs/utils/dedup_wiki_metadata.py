import json
from pathlib import Path

INPUT_PATH = Path("../data/wiki/metadata-orig.jsonl")
OUTPUT_PATH = Path("../data/wiki/metadata.jsonl")

def normalize_key_part(value):
    return str(value).strip().lower()

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_PATH}")

    seen = set()
    kept = 0
    skipped = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception as e:
                print(f"[WARN] Linha {line_num} inválida, ignorando: {e}")
                skipped += 1
                continue

            key = (
                normalize_key_part(row.get("title", "")),
                normalize_key_part(row.get("source", "")),
                int(row.get("chunk_index", 0)),
            )

            if key in seen:
                skipped += 1
                continue

            seen.add(key)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print("Concluído.")
    print(f"Arquivo de entrada: {INPUT_PATH}")
    print(f"Arquivo de saída:   {OUTPUT_PATH}")
    print(f"Registros mantidos: {kept}")
    print(f"Registros removidos/ignorados: {skipped}")

if __name__ == "__main__":
    main()
