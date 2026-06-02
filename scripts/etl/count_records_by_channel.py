import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_DATASETS = [
    "../../data/raw_comments",
    "../../data/manual_selection",
    "../../data/cleaned_comments",
]


def infer_channel_from_filename(path: Path) -> str:
    name = path.stem
    if name.endswith("_comments"):
        name = name[: -len("_comments")]
    return name


def count_csv(path: Path) -> Tuple[int, Dict[str, int]]:
    counts: Dict[str, int] = defaultdict(int)
    total = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return 0, {}

        has_source = "source" in reader.fieldnames
        fallback_channel = infer_channel_from_filename(path)

        for row in reader:
            total += 1
            source = str(row.get("source", "")).strip() if has_source else ""
            channel = source or fallback_channel
            counts[channel] += 1

    return total, dict(counts)


def iter_csv_files(dataset_dir: Path, comments_only: bool) -> Iterable[Path]:
    pattern = "*_comments.csv" if comments_only else "*.csv"
    return sorted(dataset_dir.glob(pattern))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conta registros por canal em diretorios de comentarios."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Diretorios a analisar.",
    )
    parser.add_argument(
        "--comments-only",
        action="store_true",
        help="Conta apenas arquivos *_comments.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Diretorio para salvar os CSVs de resumo.",
    )
    args = parser.parse_args()

    detail_rows: List[Dict[str, object]] = []
    summary_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for dataset in args.datasets:
        dataset_dir = Path(dataset)
        stage = dataset_dir.name

        if not dataset_dir.exists():
            detail_rows.append({
                "stage": stage,
                "file": "",
                "channel": "",
                "records": 0,
                "status": "missing_dir",
            })
            continue

        csv_files = list(iter_csv_files(dataset_dir, args.comments_only))
        if not csv_files:
            detail_rows.append({
                "stage": stage,
                "file": "",
                "channel": "",
                "records": 0,
                "status": "no_csv_files",
            })
            continue

        for csv_path in csv_files:
            try:
                _, file_counts = count_csv(csv_path)
                for channel, records in sorted(file_counts.items()):
                    detail_rows.append({
                        "stage": stage,
                        "file": str(csv_path),
                        "channel": channel,
                        "records": records,
                        "status": "ok",
                    })
                    summary_counts[(stage, channel)] += records
            except Exception as exc:
                detail_rows.append({
                    "stage": stage,
                    "file": str(csv_path),
                    "channel": infer_channel_from_filename(csv_path),
                    "records": 0,
                    "status": f"error: {exc}",
                })

    summary_rows = [
        {"stage": stage, "channel": channel, "records": records}
        for (stage, channel), records in sorted(summary_counts.items())
    ]

    output_dir = Path(args.output_dir)
    summary_path = output_dir / "records_by_channel_summary.csv"
    detail_path = output_dir / "records_by_channel_details.csv"

    write_csv(summary_path, summary_rows, ["stage", "channel", "records"])
    write_csv(detail_path, detail_rows, ["stage", "file", "channel", "records", "status"])

    print(f"Resumo salvo em: {summary_path}")
    print(f"Detalhes salvos em: {detail_path}")
    print()
    print("stage,channel,records")
    for row in summary_rows:
        print(f"{row['stage']},{row['channel']},{row['records']}")


if __name__ == "__main__":
    main()
