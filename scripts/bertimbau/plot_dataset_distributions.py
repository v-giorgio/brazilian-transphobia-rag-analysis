import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
METADATA_DIR = BASE_DIR / "pipeline" / "data" / "metadata"
OUT_DIR = BASE_DIR / "figuras"


def load_metadata(name: str) -> dict:
    with open(METADATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def annotate_heatmap(ax, values: np.ndarray, counts: np.ndarray) -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > values.max() * 0.55 else "#111827"
            ax.text(
                j,
                i,
                f"{counts[i, j]}\n{values[i, j]:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold",
            )


def plot_single_label_heatmap(meta: dict, labels: list[str], title: str, filename: str) -> None:
    split_names = ["Treino", "Validação", "Teste"]
    split_keys = ["train_distribution", "val_distribution", "test_distribution"]

    counts = np.array(
        [[meta[split_key].get(label, 0) for label in labels] for split_key in split_keys],
        dtype=float,
    )
    percentages = counts / counts.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(percentages, cmap="Blues", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(split_names)), split_names)
    ax.set_title(title, pad=14)
    ax.set_xlabel("Classe")
    ax.set_ylabel("Subconjunto")

    annotate_heatmap(ax, percentages, counts.astype(int))

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Proporção no subconjunto (%)")

    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(split_names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_aspects_heatmap(meta: dict) -> None:
    labels = meta["label_order"]
    split_names = ["Total", "Treino", "Validação", "Teste"]
    distributions = [
        meta["label_distribution"],
        meta["train_distribution"],
        meta["val_distribution"],
        meta["test_distribution"],
    ]
    denominators = [
        meta["n_total"],
        meta["split_sizes"]["train"],
        meta["split_sizes"]["val"],
        meta["split_sizes"]["test"],
    ]

    counts = np.array(
        [[dist.get(label, 0) for label in labels] for dist in distributions],
        dtype=float,
    )
    percentages = counts / np.array(denominators, dtype=float).reshape(-1, 1) * 100

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    im = ax.imshow(percentages, cmap="Purples", aspect="auto")

    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(split_names)), split_names)
    ax.set_title("Distribuição dos aspectos discursivos por subconjunto", pad=14)
    ax.set_xlabel("Aspecto")
    ax.set_ylabel("Subconjunto")

    annotate_heatmap(ax, percentages, counts.astype(int))

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Proporção no subconjunto (%)")

    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(split_names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_aspects_splits.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    speech = load_metadata("speech_metadata.json")
    offensive = load_metadata("offensive_metadata.json")
    aspects = load_metadata("aspects_metadata.json")

    plot_single_label_heatmap(
        speech,
        ["HATEFUL", "NEUTRAL", "SUPPORTIVE"],
        "Distribuição de atitude discursiva por subconjunto",
        "heatmap_speech_splits.png",
    )
    plot_single_label_heatmap(
        offensive,
        ["NON_OFFENSIVE", "OFFENSIVE"],
        "Distribuição de ofensividade por subconjunto",
        "heatmap_offensive_splits.png",
    )
    plot_aspects_heatmap(aspects)


if __name__ == "__main__":
    main()
